"""Flow Regime Identification web app.

The server is stateless: /upload validates a file and returns the cleaned
arrays to the browser, and every /cluster request carries the data it should
cluster. This keeps students' datasets separate when many people use the app
at once (the previous version stored one global dataset for everybody).
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import traceback
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import auxiliary_functions as cf  # noqa: F401  (kept for parity with the scripts)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB uploads

REQUIRED_COLUMNS = ['lndt', 'dp', 'dp_dlndt']
MAX_POINTS = 20000

class DataError(ValueError):
    """A problem with the user's data that should be shown to them as-is."""


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------
def _is_csv(filename):
    return (filename or '').lower().endswith('.csv')


def read_table(file_storage, sheet_name=None):
    name = file_storage.filename or ''
    try:
        if _is_csv(name):
            return pd.read_csv(file_storage)
        return pd.read_excel(file_storage, sheet_name=sheet_name or 0)
    except ValueError as e:
        raise DataError(f"Could not read sheet '{sheet_name}': {e}")
    except Exception as e:
        raise DataError(f"Could not read the file ({type(e).__name__}). Is it a valid .xlsx or .csv file?")


def clean_dataframe(df):
    """Validate columns and values; return arrays plus a report of what changed."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        found = ', '.join(df.columns[:10]) or 'none'
        raise DataError(f"Missing column(s): {', '.join(missing)}. Columns found: {found}.")

    report = {'rows_read': int(len(df))}
    sub = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors='coerce')
    finite = np.isfinite(sub.to_numpy(dtype=float)).all(axis=1)
    report['dropped_non_numeric'] = int((~finite).sum())
    sub = sub[finite]

    positive = (sub['dp'] > 0) & (sub['dp_dlndt'] > 0)
    report['dropped_non_positive'] = int((~positive).sum())
    sub = sub[positive]

    report['sorted'] = False
    if not sub['lndt'].is_monotonic_increasing:
        sub = sub.sort_values('lndt', kind='mergesort')
        report['sorted'] = True
    report['duplicate_times'] = int(sub['lndt'].duplicated().sum())

    if len(sub) < 4:
        raise DataError(f"Only {len(sub)} usable rows after cleaning; at least 4 are needed.")
    if len(sub) > MAX_POINTS:
        raise DataError(f"{len(sub)} rows is more than the {MAX_POINTS} this app accepts.")
    report['rows_used'] = int(len(sub))

    data = {c: sub[c].astype(float).round(10).tolist() for c in REQUIRED_COLUMNS}
    return data, report


def arrays_from_payload(data):
    if not isinstance(data, dict):
        raise DataError("No data supplied. Upload a file first.")
    try:
        lndt = np.asarray(data['lndt'], dtype=float)
        dp = np.asarray(data['dp'], dtype=float)
        dpd = np.asarray(data['dp_dlndt'], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise DataError("Malformed data. Please upload the file again.")
    if not (len(lndt) == len(dp) == len(dpd)) or len(lndt) < 4:
        raise DataError("Malformed data. Please upload the file again.")
    if len(lndt) > MAX_POINTS:
        raise DataError("Too many points.")
    if not (np.isfinite(lndt).all() and (dp > 0).all() and (dpd > 0).all()):
        raise DataError("Data contains non-positive or non-finite values. Please upload the file again.")
    return lndt, dp, dpd


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def minmax(v):
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi == lo:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo) * 2.0 - 1.0


def create_windows(data_norm, data_real, window_size):
    """Non-overlapping windows; leftover points at the end are not clustered."""
    windows = []
    n = data_norm.shape[0]
    for i in range(0, n - window_size + 1, window_size):
        wn = data_norm[i:i + window_size]
        wr = data_real[i:i + window_size]
        if np.ptp(wr[:, 0]) != 0:
            slope, intercept = np.polyfit(wr[:, 0], wr[:, 1], 1)
        else:
            slope, intercept = 0.0, float(np.mean(wr[:, 1]))
        windows.append({
            'data_norm': wn,
            'data_real': wr,
            'median_norm': np.median(wn, axis=0),
            'median_real': np.median(wr, axis=0),
            'slope': float(slope),
            'intercept': float(intercept),
            'index': len(windows),
            'start': i,
            'end': i + window_size,  # exclusive
            'inverted_block': False,
        })
    return windows


def assign_inverted_v_block(windows, p, early_time_index=None):
    """Mark every block of p windows that contains an inverted V (slope + then -).

    early_time_index controls how far back the storage (WBS) block reaches:
      None   - only the block itself (original web-app behaviour);
      int n  - same logic as clustering_with_custom_dissimilarity_metric.py: if the
               inverted V is found in a block starting at window i <= n, the flag is
               extended back to window 0;
      'auto' - the flag is extended back over the unbroken run of rising windows
               (slope > 0) before the block, however long it is. No constant, and it
               does not depend on where the hump happens to fall.
    """
    for w in windows:
        w['inverted_block'] = False
    for i in range(len(windows) - p + 1):
        block = windows[i:i + p]
        if any(block[j]['slope'] > 0 and block[j + 1]['slope'] < 0 for j in range(p - 1)):
            if early_time_index == 'auto':
                start = i
                while start > 0 and windows[start - 1]['slope'] > 0:
                    start -= 1
            elif early_time_index is not None and i <= early_time_index:
                start = 0
            else:
                start = i
            for j in range(start, i + p):
                windows[j]['inverted_block'] = True
    return windows


# Thinning by curve shape. Points are spaced evenly along the derivative curve in the
# same [-1, 1] x [-1, 1] box the model uses, with extra density where the curve bends,
# so long straight stretches (e.g. a unit-slope storage line) keep fewer points and the
# point density no longer depends on how the data were recorded. Only measured points
# are kept (the nearest one to each target position). The step keeps about 97 of the
# 309 points of the original KAPPA curves, the point count of the PanSystem example.
# Tested on the five PanSystem/KAPPA curves and the three datafile.xlsx sheets: steps of
# 0.062-0.064 give the same regime counts; 0.063 sits in the middle.
THIN_STEP = 0.063
THIN_BEND_WEIGHT = 1.0


def thin_by_curve(lndt, ln_dpd, step=THIN_STEP, bend_weight=THIN_BEND_WEIGHT):
    """Indices of the points kept when thinning by curve length plus bend weight."""
    X, Y = minmax(lndt), minmax(ln_dpd)
    ds = np.hypot(np.diff(X), np.diff(Y))
    if bend_weight > 0 and len(ds) >= 2:
        theta = np.arctan2(np.diff(Y), np.diff(X))
        dth = np.abs(np.diff(theta))
        dth = np.minimum(dth, 2 * np.pi - dth)
        turn = np.r_[dth[0], (dth[:-1] + dth[1:]) / 2, dth[-1]] / np.maximum(ds, 1e-12)
        turn = np.convolve(turn, np.ones(5) / 5, mode='same')
        if np.mean(turn) > 0:
            ds = ds * (1 + bend_weight * turn / np.mean(turn))
    s = np.r_[0.0, np.cumsum(ds)]
    targets = np.arange(0.0, s[-1] + 1e-12, step)
    return np.array(sorted({int(np.abs(s - t).argmin()) for t in targets}))


def marginal_gain_k(scores, tau):
    """Largest k whose added cluster still removes at least tau of the total cost J(1).

    scores[k-1] = J(k) for k = 1..K. Returns (k_opt, gains) with gains[k] for k = 2..K.
    """
    j1 = scores[0]
    gains = {k: ((scores[k - 2] - scores[k - 1]) / j1 if j1 > 0 else 0.0) for k in range(2, len(scores) + 1)}
    ks = [k for k, g in gains.items() if g >= tau]
    return (max(ks) if ks else 1), gains


def dissimilarity_terms(windows, le, lp, b, gb, delta, threshold):
    """Vectorised version of the composite K-Medoids dissimilarity.

    Returns the total matrix, each weighted term and the near-flat override mask.
    Same rules as the original pairwise function:
      total = max(lE*dE + lT*dT + ltau*dtau + concave + block, 0), except that two
      adjacent windows with |m| < threshold get 0.1*lE*dE.
    """
    n = len(windows)
    flat = np.array([w['data_norm'].ravel() for w in windows])
    diff = flat[:, None, :] - flat[None, :, :]
    e = np.sqrt((diff ** 2).sum(axis=2))
    d_max = e.max()
    norm_e = e / d_max if d_max > 0 else e

    slopes = np.array([w['slope'] for w in windows])
    ang = np.degrees(np.arctan(slopes))
    a = np.abs(ang[:, None] - ang[None, :])
    a = np.where(a > 90, 180 - a, a)
    norm_p = a / 90.0

    idx = np.arange(n)
    idx_diff = np.abs(idx[:, None] - idx[None, :])
    t_max = n - 1
    norm_t = np.maximum(0, idx_diff - 1) / t_max if t_max > 0 else np.maximum(0, idx_diff - 1).astype(float)

    concave = np.zeros((n, n))
    override = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        w1, w2 = windows[i], windows[i + 1]
        if abs(w1['slope']) < threshold and abs(w2['slope']) < threshold:
            override[i, i + 1] = override[i + 1, i] = True
        dx = abs(w1['median_norm'][0] - w2['median_norm'][0])
        if dx > 0:
            m1, m2 = w1['slope'], w2['slope']
            m_avg = (m1 + m2) / 2.0
            y_dd = (m2 - m1) / dx
            r = float('inf') if abs(y_dd) < 1e-6 else (1 + m_avg ** 2) ** 1.5 / abs(y_dd)
            if r > 2 * dx:
                concave[i, i + 1] = concave[i + 1, i] = -delta

    flags = np.array([bool(w['inverted_block']) for w in windows])
    block = np.where(flags[:, None] & flags[None, :], -gb, 0.0)

    E, P, T = le * norm_e, lp * norm_p, b * norm_t
    total = np.maximum(E + P + T + concave + block, 0.0)
    total = np.where(override, le * norm_e * 0.1, total)
    np.fill_diagonal(total, 0.0)
    return total, {'E': E, 'P': P, 'T': T, 'C': concave, 'B': block}, override


def kmeans_features(windows, le, lp, b):
    """Standardise each feature first, then apply the weights.

    (Weighting before StandardScaler cancels any positive weight, which made the
    sliders have no effect on K-Means.)
    """
    raw = np.array([[w['median_norm'][0], w['median_norm'][1], w['slope'], w['index']] for w in windows], dtype=float)
    std = raw.std(axis=0)
    scaled = np.zeros_like(raw)
    ok = std > 0
    scaled[:, ok] = StandardScaler().fit_transform(raw[:, ok])
    return scaled * np.array([le, le, lp, b])


def fit_model(kind, k, dist_mat, feats):
    if kind == 'kmedoids':
        return KMedoids(n_clusters=k, metric='precomputed', method='pam', random_state=42).fit(dist_mat)
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit(feats)


def contiguous_runs(labels):
    runs = {}
    prev = None
    for lab in labels:
        if lab != prev:
            runs[lab] = runs.get(lab, 0) + 1
        prev = lab
    return runs


def r(x, nd=4):
    return None if x is None else round(float(x), nd)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sheets', methods=['POST'])
def list_sheets():
    file = request.files.get('file')
    if file is None or not file.filename:
        return jsonify({'error': 'No file received.'}), 400
    if _is_csv(file.filename):
        return jsonify({'sheets': [], 'csv': True})
    try:
        sheets = pd.ExcelFile(file).sheet_names
    except Exception:
        return jsonify({'error': 'Could not open this workbook. Is it a valid .xlsx file?'}), 400
    return jsonify({'sheets': sheets, 'csv': False})


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file is None or not file.filename:
        return jsonify({'error': 'No file received.'}), 400
    sheet = request.form.get('sheet_name') or None
    try:
        df = read_table(file, sheet)
        data, report = clean_dataframe(df)
    except DataError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error while reading the file: {e}'}), 500
    label = file.filename + ('' if _is_csv(file.filename) or not sheet else f' [{sheet}]')
    return jsonify({'data': data, 'report': report, 'name': label})


@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        body = request.get_json(silent=True) or {}
        params = body.get('params') or {}
        lndt, dp, dpd = arrays_from_payload(body.get('data'))

        method = params.get('method', 'kmeans')
        if method not in ('kmeans', 'kmedoids', 'semi_automated'):
            raise DataError(f"Unknown method '{method}'.")
        backbone = params.get('backbone_method', 'kmeans') if method == 'semi_automated' else method
        if backbone not in ('kmeans', 'kmedoids'):
            raise DataError(f"Unknown backbone '{backbone}'.")

        le = float(params.get('lambda_e', 1.0))
        lp = float(params.get('lambda_p', 1.0))
        b = float(params.get('beta', 0.5))
        gb = float(params.get('gamma_block', 1.0))
        delta = float(params.get('delta', 0.1))
        threshold = float(params.get('threshold', 0.1))
        threshold_elbow = float(params.get('threshold_elbow', 0.15))
        k_rule = params.get('k_rule', 'elbow')          # 'elbow' (slope ratio) or 'marginal_gain'
        if k_rule not in ('elbow', 'marginal_gain'):
            raise DataError(f"Unknown k selection rule '{k_rule}'.")
        tau = float(params.get('tau', 0.04))
        early = params.get('early_time_index', None)     # None, '' or a negative value = off; 'auto'
        if early != 'auto':
            early = None if early in (None, '') or int(early) < 0 else int(early)
        thin = params.get('thin', 'off')                  # 'off' or 'curve'
        if thin not in ('off', 'curve'):
            raise DataError(f"Unknown thinning option '{thin}'.")
        w_size = int(params.get('window_size', 5))
        p = int(params.get('p', 4))
        if min(le, lp, b, gb) < 0:
            raise DataError("Weights must be zero or positive.")

        n_in = len(lndt)
        if thin == 'curve' and n_in >= 10:
            keep = thin_by_curve(lndt, np.log(dpd))
            lndt, dp, dpd = lndt[keep], dp[keep], dpd[keep]
        ln_dpd = np.log(dpd)

        n_pts = len(lndt)
        if w_size < 2 or w_size > n_pts // 2:
            raise DataError(f"Window size must be between 2 and {n_pts // 2} for {n_pts} points.")

        norm_stack = np.column_stack((minmax(lndt), minmax(ln_dpd)))
        real_stack = np.column_stack((lndt, ln_dpd))
        windows = create_windows(norm_stack, real_stack, w_size)
        n_w = len(windows)
        if backbone == 'kmedoids':
            assign_inverted_v_block(windows, p, early)

        dist_mat, _, _ = dissimilarity_terms(windows, le, lp, b, gb, delta, threshold)
        feats = None
        if backbone == 'kmeans':
            if le == 0 and lp == 0 and b == 0:
                raise DataError("For K-Means at least one of λE, λθ, λτ must be above zero.")
            feats = kmeans_features(windows, le, lp, b)

        elbow = None
        if method == 'semi_automated':
            k_values = list(range(1, min(10, n_w) + 1))
            scores = [float(fit_model(backbone, k, dist_mat, feats).inertia_) for k in k_values]
            # Slope ratios are reported for both rules so they can be compared.
            ratios = [None] * len(k_values)
            fallback_k = min(3, n_w)
            elbow_k, elbow_detected = fallback_k, False
            for i in range(1, len(scores) - 1):
                m1 = scores[i] - scores[i - 1]
                m2 = scores[i + 1] - scores[i]
                if m1 != 0:
                    ratios[i] = abs(m2) / abs(m1)
                    if not elbow_detected and ratios[i] < threshold_elbow:
                        elbow_k, elbow_detected = k_values[i], True
            gain_k, gains = marginal_gain_k(scores, tau)
            if k_rule == 'marginal_gain':
                k_opt, detected = gain_k, gain_k > 1
            else:
                k_opt, detected = elbow_k, elbow_detected
            k = k_opt
            elbow = {'rule': k_rule, 'k_values': k_values, 'scores': scores,
                     'ratios': ratios, 'threshold': threshold_elbow,
                     'gains': [None] + [float(gains[kk]) for kk in k_values[1:]], 'tau': tau,
                     'k_opt': int(k_opt), 'detected': detected, 'fallback_k': int(fallback_k),
                     'elbow_k': int(elbow_k), 'gain_k': int(gain_k)}
        else:
            k = int(params.get('n_clusters', 3))
            if k < 1 or k > n_w:
                raise DataError(f"Number of clusters ({k}) must be between 1 and the number of windows ({n_w}). "
                                "Try a smaller window size.")

        model = fit_model(backbone, k, dist_mat, feats)
        labels = np.asarray(model.labels_)

        # Order clusters left to right in time.
        clusters = np.unique(labels)
        xs = np.array([w['median_norm'][0] for w in windows])
        order = np.argsort([np.mean(xs[labels == c]) for c in clusters])
        mapping = {int(clusters[o]): new for new, o in enumerate(order)}
        labels = np.array([mapping[int(l)] for l in labels])
        n_clusters = len(clusters)

        # Centres in real (ln) coordinates.
        centers = []
        if hasattr(model, 'medoid_indices_'):
            center_kind = 'medoid'
            for orig, win_idx in enumerate(model.medoid_indices_):
                if orig in mapping:
                    m = windows[win_idx]['median_real']
                    centers.append({'cluster': mapping[orig], 'lndt': m[0], 'ln_dpd': m[1], 'window': int(win_idx)})
        else:
            center_kind = 'centroid'
            for c in range(n_clusters):
                meds = np.array([w['median_real'] for w, l in zip(windows, labels) if l == c])
                m = meds.mean(axis=0)
                centers.append({'cluster': c, 'lndt': m[0], 'ln_dpd': m[1], 'window': None})
        ln_range = float(np.ptp(ln_dpd)) or 1.0
        for c in centers:
            on_curve = float(np.interp(c['lndt'], lndt, ln_dpd))
            c['offset_frac'] = abs(c['ln_dpd'] - on_curve) / ln_range
        centers.sort(key=lambda c: c['cluster'])

        # Per-regime summary.
        runs = contiguous_runs(labels.tolist())
        regimes = []
        for c in range(n_clusters):
            members = [w for w, l in zip(windows, labels) if l == c]
            pts = np.vstack([w['data_real'] for w in members])
            fit = np.polyfit(pts[:, 0], pts[:, 1], 1)[0] if np.ptp(pts[:, 0]) > 0 else 0.0
            ws = [w['slope'] for w in members]
            idxs = np.concatenate([np.arange(w['start'], w['end']) for w in members])
            regimes.append({
                'cluster': c,
                'n_windows': len(members),
                'n_points': int(len(idxs)),
                't_start': float(np.exp(pts[:, 0].min())),
                't_end': float(np.exp(pts[:, 0].max())),
                'slope_fit': r(fit),
                'slope_min': r(min(ws)),
                'slope_max': r(max(ws)),
                'runs': int(runs.get(c, 0)),
                'dpd_median': float(np.median(dpd[idxs])),
            })

        out_windows = [{
            'index': w['index'], 'start': w['start'], 'end': w['end'],
            'slope': r(w['slope']), 'intercept': r(w['intercept'], 6),
            'inverted_block': bool(w['inverted_block']), 'cluster': int(l),
        } for w, l in zip(windows, labels)]

        return jsonify({
            'method': method, 'backbone': backbone, 'k': int(n_clusters),
            'windows': out_windows,
            'leftover_start': windows[-1]['end'] if windows else 0,
            'centers': [{'cluster': c['cluster'], 'lndt': r(c['lndt'], 6), 'ln_dpd': r(c['ln_dpd'], 6),
                         'window': c['window'], 'offset_frac': r(c['offset_frac'], 5)} for c in centers],
            'center_kind': center_kind,
            'regimes': regimes,
            'elbow': elbow,
            'thin': {'mode': thin, 'n_in': int(n_in), 'n_used': int(n_pts)},
            'data_used': ({'lndt': lndt.round(10).tolist(), 'dp': dp.round(10).tolist(), 'dp_dlndt': dpd.round(10).tolist()}
                          if thin != 'off' else None),
        })
    except DataError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Clustering failed: {type(e).__name__}: {e}'}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'File is larger than 10 MB.'}), 413


if __name__ == '__main__':
    app.run(debug=True)
