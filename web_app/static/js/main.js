/* Flow Regime Identification - teaching UI
 *
 * The browser keeps the uploaded data and sends it with every clustering
 * request, so each student works on their own dataset.
 */
document.addEventListener('DOMContentLoaded', function () {
    'use strict';

    // ------------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------------
    // Categorical palette validated for colour-vision deficiency (adjacent
    // regimes). Marker shapes and R# labels give a second, colour-free cue.
    const COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
                    '#008300', '#6250d6', '#e34948', '#0095a8', '#a0612a'];
    const SYMBOLS = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down',
                     'cross', 'x', 'star-triangle-up', 'hexagon', 'pentagon'];
    const INK = '#1c2733';
    const MUTED = '#7a8591';
    const GRID = '#edf0f3';
    const CENTER_TOL = 0.02;      // centre offset allowed, as a fraction of the ln Δp' range
    const HISTORY_KEY = 'fri_run_history_v1';
    const HISTORY_MAX = 300;
    const PLOT_CONFIG = { responsive: true, displaylogo: false };
    const METHOD_LABEL = { kmeans: 'K-Means', kmedoids: 'K-Medoids', semi_automated: 'Semi-automated' };

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    const state = {
        file: null,
        data: null,          // {lndt, dp, dp_dlndt, name}
        current: null,       // last run: {id, res, data, params, dataset, time}
        pinned: null,
        activeTab: 'preview',
        dirty: new Set(),
        history: loadHistory(),
        runCounter: 0,
        present: false,
        seq: 0,              // latest clustering request number
        controller: null,    // AbortController of the request in flight
        autoTimer: null,
    };
    state.runCounter = state.history.reduce((m, h) => Math.max(m, h.run || 0), 0);

    // ------------------------------------------------------------------
    // Elements
    // ------------------------------------------------------------------
    const $ = id => document.getElementById(id);
    const el = {
        uploadForm: $('uploadForm'), fileInput: $('fileInput'), fileName: $('fileName'),
        sheetRow: $('sheetRow'), sheetName: $('sheetName'), uploadBtn: $('uploadBtn'),
        uploadReport: $('uploadReport'), previewType: $('previewType'),
        method: $('clusteringMethod'), runBtn: $('updatePlot'), runHint: $('runHint'),
        plateauRegime: $('plateauRegime'), plateauTol: $('plateauTol'),
    };
    const sliders = {
        nClusters: { el: $('nClusters'), out: $('n-clusters-val'), digits: 0 },
        windowSize: { el: $('windowSize'), out: $('window-val'), digits: 0 },
        lambdaE: { el: $('lambdaE'), out: $('le-val'), digits: 1 },
        lambdaP: { el: $('lambdaP'), out: $('lp-val'), digits: 1 },
        beta: { el: $('beta'), out: $('beta-val'), digits: 1 },
        gammaBlock: { el: $('gammaBlock'), out: $('gamma-val'), digits: 1 },
        p: { el: $('p'), out: $('p-val'), digits: 0 },
        thresholdElbow: { el: $('thresholdElbow'), out: $('t-elbow-val'), digits: 2 },
        tau: { el: $('tau'), out: $('tau-val'), digits: 3 },
        earlyTimeIndex: { el: $('earlyTimeIndex'), out: $('early-val'), digits: 0 },
        delta: { el: $('delta'), out: $('delta-val'), digits: 2 },
        threshold: { el: $('threshold'), out: $('thr-val'), digits: 2 },
    };

    // ------------------------------------------------------------------
    // Small helpers
    // ------------------------------------------------------------------
    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }
    function fmt(v, d = 3) {
        if (v === null || v === undefined || !isFinite(v)) return '–';
        const a = Math.abs(v);
        if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
        return v.toFixed(d);
    }
    function fmtT(v) {
        if (!isFinite(v)) return '–';
        return v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v.toPrecision(2);
    }
    function color(c) { return COLORS[c % COLORS.length]; }
    function symbol(c) { return SYMBOLS[c % SYMBOLS.length]; }
    function rgba(hex, a) {
        const n = parseInt(hex.slice(1), 16);
        return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
    }
    function regimeName(c) { return 'R' + (c + 1); }

    // Status line in the top bar (successes) and an inline banner (errors):
    // nothing floats over the plots or the controls.
    let statusTimer = null;
    function setStatus(msg, busy) {
        const st = $('status');
        clearTimeout(statusTimer);
        st.classList.remove('fade');
        st.classList.toggle('busy', !!busy);
        st.innerHTML = '<span class="dot"></span>';
        st.appendChild(document.createTextNode(msg));
        if (!busy) statusTimer = setTimeout(() => st.classList.add('fade'), 6000);
    }
    function showError(msg) {
        $('errorText').textContent = msg;
        $('errorBanner').hidden = false;
        $('status').classList.add('fade');
    }
    function clearError() { $('errorBanner').hidden = true; }
    $('dismissError').addEventListener('click', clearError);
    function showNotification(msg, type) {
        if (type === 'error') showError(msg); else { clearError(); setStatus(msg); }
    }

    async function requestJSON(url, options) {
        let response;
        try {
            response = await fetch(url, options);
        } catch (e) {
            if (e.name === 'AbortError') throw e;
            throw new Error('Could not reach the server. Check your connection.');
        }
        let body = null;
        try { body = await response.json(); } catch (e) { /* not JSON */ }
        if (!response.ok) {
            throw new Error((body && body.error) || `Server error (HTTP ${response.status}).`);
        }
        return body;
    }

    function baseLayout(extra) {
        return Object.assign({
            font: { family: 'IBM Plex Sans, Segoe UI, sans-serif', size: 12, color: INK },
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#ffffff',
            margin: { l: 70, r: 20, t: 50, b: 60 },
            hovermode: 'closest',
            autosize: true,
            legend: { orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top', font: { size: 11 } },
        }, extra);
    }
    function axis(title, type, extra) {
        return Object.assign({
            title: { text: title }, type: type || 'linear',
            gridcolor: GRID, zeroline: false, linecolor: '#bdbdbd', ticks: 'outside', tickcolor: '#bdbdbd',
            exponentformat: 'power',
        }, type === 'log' ? { dtick: 1 } : {}, extra);
    }

    // Every plot goes through draw(): it keeps the user's zoom between runs on
    // the same data (uirevision) and enlarges text and marks in presentation mode.
    function scaleForPresent(traces, layout) {
        const k = 1.4;
        const mul = (o, key, f) => { if (o && typeof o[key] === 'number') o[key] = Math.round(o[key] * f * 10) / 10; };
        traces.forEach(t => {
            if (t.marker) { mul(t.marker, 'size', k); if (t.marker.line) mul(t.marker.line, 'width', 1.3); }
            if (t.line) mul(t.line, 'width', 1.5);
            if (t.textfont) mul(t.textfont, 'size', k);
        });
        layout.font = Object.assign({}, layout.font, { size: 17 });
        if (layout.legend) layout.legend = Object.assign({}, layout.legend, { font: { size: 15 } });
        if (layout.title && layout.title.font) mul(layout.title.font, 'size', k);
        (layout.annotations || []).forEach(a => { if (a.font) mul(a.font, 'size', k); });
        (layout.shapes || []).forEach(sh => { if (sh.line) mul(sh.line, 'width', 1.5); });
        if (layout.margin) layout.margin = Object.assign({}, layout.margin, { l: (layout.margin.l || 70) + 20, b: (layout.margin.b || 60) + 20 });
    }
    function draw(div, traces, layout) {
        // Legend beside the plot, or below it on a phone-width screen.
        const node = typeof div === 'string' ? $(div) : div;
        if (layout.showlegend !== false && node && node.clientWidth && node.clientWidth < 640) {
            layout.legend = Object.assign({}, layout.legend, { orientation: 'h', x: 0, xanchor: 'left', y: -0.3, yanchor: 'top' });
            layout.margin = Object.assign({}, layout.margin, { b: 170, r: 10, l: 55 });
        }
        if (layout.uirevision === undefined) layout.uirevision = state.data ? state.data.name : 'none';
        if (state.present) scaleForPresent(traces, layout);
        return Plotly.react(div, traces, layout, PLOT_CONFIG);
    }

    // ------------------------------------------------------------------
    // Info pop-ups (formula tooltips)
    // ------------------------------------------------------------------
    document.querySelectorAll('.info-btn').forEach(btn => {
        const pop = $('info-' + btn.dataset.info);
        if (!pop) return;
        btn.setAttribute('aria-expanded', 'false');
        btn.setAttribute('aria-controls', pop.id);
        btn.addEventListener('click', e => {
            e.preventDefault();
            e.stopPropagation();
            const open = !pop.classList.contains('open');
            pop.classList.toggle('open', open);
            btn.setAttribute('aria-expanded', String(open));
            if (open && pop.classList.contains('tex2jax_ignore')) {
                pop.classList.remove('tex2jax_ignore');
                if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([pop]).catch(() => {});
            }
        });
    });

    // ------------------------------------------------------------------
    // Sliders and method visibility
    // ------------------------------------------------------------------
    Object.values(sliders).forEach(s => {
        const sync = () => { const v = Number(s.el.value); s.out.textContent = s.format ? s.format(v) : v.toFixed(s.digits); };
        s.el.addEventListener('input', () => { sync(); scheduleRun(); });
        sync();
    });

    function currentBackbone() {
        if (el.method.value !== 'semi_automated') return el.method.value;
        const r = document.querySelector('input[name="backboneMethod"]:checked');
        return r ? r.value : 'kmeans';
    }
    function currentBlockMode() {
        const r = document.querySelector('input[name="blockMode"]:checked');
        return r ? r.value : 'off';
    }
    // Storage-block reach as sent to the server: -1 (hump only), an index, or 'auto'.
    function earlyParam() {
        const mode = currentBlockMode();
        return mode === 'auto' ? 'auto' : mode === 'index' ? Number(sliders.earlyTimeIndex.el.value) : -1;
    }
    function earlyLabel(p) {
        const e = p.early_time_index;
        if (e === 'auto') return 'auto';
        return (e !== undefined && e !== null && Number(e) >= 0) ? String(e) : null;
    }
    function currentKRule() {
        const r = document.querySelector('input[name="kRule"]:checked');
        return r ? r.value : 'elbow';
    }

    function handleMethodChange() {
        const isSemi = el.method.value === 'semi_automated';
        const medoids = currentBackbone() === 'kmedoids';
        $('nClustersContainer').hidden = isSemi;
        $('backbone-ui').hidden = !isSemi;
        const rule = currentKRule();
        $('kRuleSection').hidden = !isSemi;
        $('elbowThresholdContainer').hidden = rule !== 'elbow';
        $('tauContainer').hidden = rule !== 'marginal_gain';
        $('gammaContainer').hidden = !medoids;
        $('additionalParamsBox').hidden = !medoids;   // V-size, storage-block reach, Advanced
        $('earlyIndexRow').hidden = currentBlockMode() !== 'index';
        $('kmeansNote').hidden = medoids;
        // Close tooltips belonging to hidden controls.
        document.querySelectorAll('.info-pop.open').forEach(pop => {
            const btn = document.querySelector(`.info-btn[aria-controls="${pop.id}"]`);
            if (btn && btn.offsetParent === null) { pop.classList.remove('open'); btn.setAttribute('aria-expanded', 'false'); }
        });
    }
    el.method.addEventListener('change', () => { handleMethodChange(); scheduleRun(); });
    document.querySelectorAll('input[name="backboneMethod"], input[name="kRule"], input[name="blockMode"], #thinData').forEach(r => r.addEventListener('change', () => { handleMethodChange(); scheduleRun(); }));
    handleMethodChange();

    function collectParams() {
        const v = k => Number(sliders[k].el.value);
        const params = {
            method: el.method.value,
            n_clusters: v('nClusters'),
            window_size: v('windowSize'),
            lambda_e: v('lambdaE'),
            lambda_p: v('lambdaP'),
            beta: v('beta'),
            gamma_block: v('gammaBlock'),
            p: v('p'),
            delta: v('delta'),
            threshold: v('threshold'),
            threshold_elbow: v('thresholdElbow'),
            tau: v('tau'),
            early_time_index: earlyParam(),
            thin: $('thinData').checked ? 'curve' : 'off',
        };
        if (params.method === 'semi_automated') {
            params.backbone_method = currentBackbone();
            params.k_rule = currentKRule();
        }
        return params;
    }

    function restoreParams(p) {
        el.method.value = p.method;
        if (p.backbone_method) {
            const r = document.querySelector(`input[name="backboneMethod"][value="${p.backbone_method}"]`);
            if (r) r.checked = true;
        }
        if (p.k_rule) {
            const r = document.querySelector(`input[name="kRule"][value="${p.k_rule}"]`);
            if (r) r.checked = true;
        }
        const set = (k, val) => { if (val !== undefined && val !== null) { sliders[k].el.value = val; sliders[k].el.dispatchEvent(new Event('input')); } };
        set('nClusters', p.n_clusters); set('windowSize', p.window_size);
        set('lambdaE', p.lambda_e); set('lambdaP', p.lambda_p); set('beta', p.beta);
        set('gammaBlock', p.gamma_block); set('p', p.p); set('delta', p.delta);
        set('threshold', p.threshold); set('thresholdElbow', p.threshold_elbow);
        set('tau', p.tau);
        const e = p.early_time_index;
        const mode = e === 'auto' ? 'auto' : (e !== undefined && e !== null && Number(e) >= 0) ? 'index' : 'off';
        const mr = document.querySelector(`input[name="blockMode"][value="${mode}"]`);
        if (mr) mr.checked = true;
        if (mode === 'index') set('earlyTimeIndex', Number(e));
        $('thinData').checked = p.thin === 'curve';
        handleMethodChange();
    }

    // ------------------------------------------------------------------
    // Tabs (plots are drawn when their tab is visible so Plotly sizes them)
    // ------------------------------------------------------------------
    const RENDERERS = {
        preview: renderPreview,
        results: renderResults,
        segmentation: renderSegmentation,
        elbow: renderElbow,
        compare: renderCompare,
    };
    document.querySelectorAll('.tab').forEach(tab => {
        if (tab.getAttribute('aria-disabled') === 'true') tab.title = tab.dataset.reason;
        tab.addEventListener('click', () => activateTab(tab.dataset.tab));
    });

    function activateTab(name) {
        const tab = document.querySelector(`.tab[data-tab="${name}"]`);
        if (!tab) return;
        if (tab.getAttribute('aria-disabled') === 'true') {
            $('tabReason').textContent = `${tab.textContent}: ${tab.dataset.reason}`;
            clearTimeout(activateTab.t);
            activateTab.t = setTimeout(() => { $('tabReason').textContent = ''; }, 5000);
            return;
        }
        $('tabReason').textContent = '';
        state.activeTab = name;
        document.querySelectorAll('.tab').forEach(t => {
            t.classList.toggle('active', t === tab);
            t.setAttribute('aria-selected', String(t === tab));
        });
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + name));
        if (state.dirty.has(name)) {
            state.dirty.delete(name);
            RENDERERS[name]();
            resizeVisible();
        } else {
            document.querySelectorAll('#panel-' + name + ' .js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
        }
    }
    // Plotly does not notice container size changes made by CSS (presentation
    // mode, a hidden tab becoming visible), so resize the visible plots.
    function resizeVisible() {
        requestAnimationFrame(() => document.querySelectorAll('.tab-panel.active .js-plotly-plot').forEach(p => Plotly.Plots.resize(p)));
    }
    function invalidate(names) {
        names.forEach(n => state.dirty.add(n));
        if (names.includes(state.activeTab)) {
            state.dirty.delete(state.activeTab);
            RENDERERS[state.activeTab]();
        }
    }
    function setTabEnabled(name, enabled, reason) {
        const tab = document.querySelector(`.tab[data-tab="${name}"]`);
        tab.setAttribute('aria-disabled', String(!enabled));
        if (reason) tab.dataset.reason = reason;
        tab.title = enabled ? '' : tab.dataset.reason;
        if (!enabled && state.activeTab === name) activateTab(state.current ? 'results' : 'preview');
    }

    // ------------------------------------------------------------------
    // Upload: list sheets, then upload the chosen sheet
    // ------------------------------------------------------------------
    el.fileInput.addEventListener('change', async () => {
        const file = el.fileInput.files[0];
        state.file = file || null;
        el.uploadBtn.disabled = true;
        el.sheetRow.hidden = true;
        el.sheetName.innerHTML = '';
        hideReport();
        if (!file) { el.fileName.textContent = 'Choose a file'; return; }
        el.fileName.textContent = file.name;
        if (file.name.toLowerCase().endsWith('.csv')) {
            el.uploadBtn.disabled = false;
            return;
        }
        const fd = new FormData();
        fd.append('file', file);
        try {
            const res = await requestJSON('/sheets', { method: 'POST', body: fd });
            res.sheets.forEach(name => {
                const o = document.createElement('option');
                o.value = name; o.textContent = name;
                el.sheetName.appendChild(o);
            });
            el.sheetRow.hidden = false;
            el.uploadBtn.disabled = res.sheets.length === 0;
        } catch (e) {
            showReport(e.message);
        }
    });

    el.uploadForm.addEventListener('submit', async e => {
        e.preventDefault();
        if (!state.file) return;
        const fd = new FormData();
        fd.append('file', state.file);
        if (!el.sheetRow.hidden) fd.append('sheet_name', el.sheetName.value);
        el.uploadBtn.disabled = true;
        el.uploadBtn.textContent = 'Loading…';
        try {
            const res = await requestJSON('/upload', { method: 'POST', body: fd });
            state.data = Object.assign({ name: res.name }, res.data);
            showUploadReport(res);
            el.runBtn.disabled = false;
            el.runHint.textContent = $('autoUpdate').checked ? 'Results update after each change.' : 'Press Run clustering after changing settings.';
            sliders.windowSize.el.max = Math.max(2, Math.min(21, Math.floor(res.report.rows_used / 2)));
            invalidate(['preview']);
            activateTab('preview');
            showNotification(`Loaded ${res.name}`, 'success');
            scheduleRun(0);
        } catch (err) {
            showReport(err.message);

        } finally {
            el.uploadBtn.disabled = false;
            el.uploadBtn.textContent = 'Load data';
        }
    });

    function hideReport() { el.uploadReport.hidden = true; }
    function showReport(text) {
        el.uploadReport.textContent = text;
        el.uploadReport.hidden = false;
    }
    // After a successful load the form collapses to a one-line summary.
    function showUploadReport(res) {
        const r = res.report;
        const notes = [];
        if (r.dropped_non_numeric) notes.push(`${r.dropped_non_numeric} row(s) dropped: empty or non-numeric values.`);
        if (r.dropped_non_positive) notes.push(`${r.dropped_non_positive} row(s) dropped: Δp or Δp′ ≤ 0 (cannot take the log).`);
        if (r.sorted) notes.push('Rows were not in time order and have been sorted by lndt.');
        if (r.duplicate_times) notes.push(`${r.duplicate_times} duplicate time value(s) found.`);
        const sheet = el.sheetRow.hidden ? '' : el.sheetName.value;
        $('dataName').textContent = state.file ? state.file.name : res.name;
        $('dataMeta').textContent = (sheet ? `Sheet ${sheet}, ` : '') + `${r.rows_used} of ${r.rows_read} rows used`;
        $('dataNotes').innerHTML = notes.map(n => `<li>${escapeHtml(n)}</li>`).join('');
        hideReport();
        el.uploadForm.hidden = true;
        $('dataSummary').hidden = false;
    }
    $('changeData').addEventListener('click', () => {
        el.uploadForm.hidden = false;
        $('cancelChange').hidden = false;
        $('dataSummary').hidden = true;
        el.fileInput.focus();
    });
    $('cancelChange').addEventListener('click', () => {
        el.uploadForm.hidden = true;
        $('dataSummary').hidden = false;
        hideReport();
    });

    // ------------------------------------------------------------------
    // Preview
    // ------------------------------------------------------------------
    el.previewType.addEventListener('change', () => invalidate(['preview']));

    function renderPreview() {
        const d = state.data;
        if (!d) return;
        const t = d.lndt.map(Math.exp);
        const type = el.previewType.value;
        const traces = [];
        let layout;
        if (type === 'loglog') {
            traces.push({ x: t, y: d.dp, name: 'Δp', mode: 'markers', type: 'scatter',
                marker: { color: '#ffffff', size: 6, line: { color: '#2a78d6', width: 1.5 } },
                hovertemplate: 'Δt = %{x:.4g}<br>Δp = %{y:.4g}<extra></extra>' });
            traces.push({ x: t, y: d.dp_dlndt, name: 'Δp′ (Bourdet derivative)', mode: 'markers', type: 'scatter',
                marker: { color: '#eb6834', size: 6, symbol: 'x' },
                hovertemplate: 'Δt = %{x:.4g}<br>Δp′ = %{y:.4g}<extra></extra>' });
            layout = baseLayout({ title: { text: 'Log-log diagnostic plot' },
                xaxis: axis('Δt (hr)', 'log'), yaxis: axis('Δp, Δp′', 'log') });
        } else {
            traces.push({ x: t, y: d.dp, name: 'Δp', mode: 'lines+markers', type: 'scatter',
                line: { color: '#2a78d6', width: 2 }, marker: { size: 4 },
                hovertemplate: 'Δt = %{x:.4g}<br>Δp = %{y:.4g}<extra></extra>' });
            layout = baseLayout({ title: { text: type === 'semilog' ? 'Semi-log plot' : 'Cartesian plot' },
                showlegend: false,
                xaxis: axis('Δt (hr)', type === 'semilog' ? 'log' : 'linear'), yaxis: axis('Δp') });
        }
        const pv = $('previewPlot');
        if (pv.querySelector('.empty-state')) pv.innerHTML = '';
        draw('previewPlot', traces, layout);
    }

    // ------------------------------------------------------------------
    // Run clustering
    // ------------------------------------------------------------------
    el.runBtn.addEventListener('click', () => runClustering(false));
    $('autoUpdate').addEventListener('change', e => {
        if (state.data) el.runHint.textContent = e.target.checked ? 'Results update after each change.' : 'Press Run clustering after changing settings.';
        if (e.target.checked) scheduleRun(0);
    });

    // Automatic update: wait until the slider has been still briefly, then run.
    // The semi-automated method fits ten models, so it waits a little longer.
    function scheduleRun(delay) {
        if (!state.data || !$('autoUpdate').checked) return;
        clearTimeout(state.autoTimer);
        const wait = delay !== undefined ? delay : (el.method.value === 'semi_automated' ? 600 : 250);
        state.autoTimer = setTimeout(() => runClustering(true), wait);
    }

    async function runClustering(auto) {
        if (!state.data) return;
        clearTimeout(state.autoTimer);
        const params = collectParams();
        const seq = ++state.seq;
        if (state.controller) state.controller.abort();
        state.controller = new AbortController();
        setStatus('Updating…', true);
        if (!auto) { el.runBtn.disabled = true; el.runBtn.textContent = 'Running…'; }
        try {
            const { name, ...arrays } = state.data;
            const res = await requestJSON('/cluster', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: arrays, params: params }),
                signal: state.controller.signal,
            });
            if (seq !== state.seq) return;              // a newer request has been sent
            state.runCounter += 1;
            // With thinning on, the server returns the points it kept; plots and checks use those.
            const used = res.data_used ? Object.assign({ name }, res.data_used) : state.data;
            state.current = { id: state.runCounter, res, data: used, params, dataset: name, time: new Date() };
            populatePlateauSelect();
            ['results', 'segmentation', 'compare'].forEach(n => setTabEnabled(n, true));
            setTabEnabled('elbow', !!res.elbow, 'Available for the semi-automated method only.');
            addHistoryEntry(state.current, auto);
            invalidate(['results', 'segmentation', 'elbow', 'compare']);
            if (state.activeTab === 'preview' || (state.activeTab === 'elbow' && !res.elbow)) activateTab('results');
            clearError();
            setStatus(`Run #${state.current.id}: ${methodLabel(state.current)}, k = ${res.k}`
                + (res.thin && res.thin.mode !== 'off' ? `, ${res.thin.n_used} of ${res.thin.n_in} points after thinning` : ''));
        } catch (e) {
            if (e.name === 'AbortError' || seq !== state.seq) return;
            showError(state.current ? `${e.message} The plots still show the last successful run.` : e.message);
        } finally {
            if (seq === state.seq) {
                state.controller = null;
                el.runBtn.disabled = false;
                el.runBtn.textContent = 'Run clustering';
            }
        }
    }

    // ------------------------------------------------------------------
    // Shared geometry for a run
    // ------------------------------------------------------------------
    function runGeometry(run) {
        const d = run.data, res = run.res;
        const t = d.lndt.map(Math.exp);
        const wins = res.windows;
        // Contiguous runs of the same label -> background bands.
        const bands = [];
        let start = 0;
        for (let i = 1; i <= wins.length; i++) {
            if (i === wins.length || wins[i].cluster !== wins[start].cluster) {
                const x0 = t[wins[start].start];
                const x1 = i < wins.length ? t[wins[i].start] : t[wins[i - 1].end - 1];
                bands.push({ cluster: wins[start].cluster, x0, x1 });
                start = i;
            }
        }
        return { t, wins, bands };
    }

    function regimeTraces(run, opts) {
        const { t, wins } = runGeometry(run);
        const d = run.data, res = run.res;
        const traces = [];
        if (opts.showDp) {
            traces.push({ x: t, y: d.dp, name: 'Δp', mode: 'markers', type: 'scatter',
                marker: { color: '#ffffff', size: 5, line: { color: MUTED, width: 1 } },
                hovertemplate: 'Δt = %{x:.4g}<br>Δp = %{y:.4g}<extra>Δp</extra>' });
        }
        for (let c = 0; c < res.k; c++) {
            const xs = [], ys = [];
            wins.filter(w => w.cluster === c).forEach(w => {
                for (let i = w.start; i < w.end; i++) { xs.push(t[i]); ys.push(d.dp_dlndt[i]); }
            });
            const reg = res.regimes[c];
            traces.push({ x: xs, y: ys, type: 'scatter', mode: 'markers', name: `${regimeName(c)}  (m = ${fmt(reg.slope_fit, 2)})`,
                marker: { color: color(c), symbol: symbol(c), size: opts.markerSize || 8, line: { color: '#ffffff', width: 1 } },
                hovertemplate: `${regimeName(c)}<br>Δt = %{x:.4g}<br>Δp′ = %{y:.4g}<extra></extra>` });
        }
        const left = res.leftover_start;
        if (left < t.length) {
            traces.push({ x: t.slice(left), y: d.dp_dlndt.slice(left), type: 'scatter', mode: 'markers',
                name: 'not clustered (leftover)', marker: { color: '#c8c8c8', size: 6 },
                hovertemplate: 'not clustered<br>Δt = %{x:.4g}<br>Δp′ = %{y:.4g}<extra></extra>' });
        }
        const cname = res.center_kind === 'medoid' ? 'Medoids' : 'Centroids';
        traces.push({
            x: res.centers.map(c => Math.exp(c.lndt)), y: res.centers.map(c => Math.exp(c.ln_dpd)),
            type: 'scatter', mode: 'markers', name: cname,
            marker: { symbol: 'star', size: opts.starSize || 16, color: '#111111', line: { color: '#ffffff', width: 1.5 } },
            customdata: res.centers.map(c => [regimeName(c.cluster), (100 * c.offset_frac).toFixed(1)]),
            hovertemplate: `${cname.slice(0, -1)} of %{customdata[0]}<br>Δt = %{x:.4g}<br>Δp′ = %{y:.4g}<br>off the curve by %{customdata[1]}% of the ln Δp′ range<extra></extra>`,
        });
        return traces;
    }

    function bandShapesAndLabels(run, withLabels) {
        const { bands } = runGeometry(run);
        const shapes = bands.map(b => ({
            type: 'rect', xref: 'x', yref: 'paper', x0: b.x0, x1: b.x1, y0: 0, y1: 1,
            fillcolor: rgba(color(b.cluster), 0.13), line: { width: 0 }, layer: 'below',
        }));
        const annotations = withLabels ? bands.map(b => ({
            xref: 'x', yref: 'paper', x: (Math.log10(b.x0) + Math.log10(b.x1)) / 2, y: 1, yanchor: 'bottom',
            text: `<b>${regimeName(b.cluster)}</b>`, showarrow: false, font: { size: 11, color: INK },
        })) : [];
        return { shapes, annotations };
    }

    function ruleLabel(p) {
        return p.k_rule === 'marginal_gain' ? `marginal gain τ = ${Number(p.tau).toFixed(3)}` : `elbow m_ratio = ${Number(p.threshold_elbow).toFixed(2)}`;
    }
    function methodLabel(run) {
        const p = run.params;
        let m = METHOD_LABEL[p.method];
        if (p.method === 'semi_automated') m += ` (${METHOD_LABEL[p.backbone_method]} backbone, ${ruleLabel(p)})`;
        return m;
    }

    // ------------------------------------------------------------------
    // Results tab
    // ------------------------------------------------------------------
    function drawResultsPlot() {
        const run = state.current;
        const traces = regimeTraces(run, { showDp: true });
        const { shapes, annotations } = bandShapesAndLabels(run, true);
        const d = run.data;
        const allY = d.dp.concat(d.dp_dlndt);
        const yr = [Math.log10(Math.min(...allY)) - 0.15, Math.log10(Math.max(...allY)) + 0.15];
        const layout = baseLayout({
            title: { text: `${methodLabel(run)}, k = ${run.res.k}, w = ${run.params.window_size}` },
            xaxis: axis('Δt (hr)', 'log'),
            yaxis: axis('Δp, Δp′', 'log', { range: yr }),
            shapes, annotations,
            margin: { l: 70, r: 20, t: 70, b: 60 },
        });
        draw('resultsPlot', traces, layout);
    }

    function renderResults() {
        const run = state.current;
        if (!run) return;
        drawResultsPlot();
        renderRegimeTable(run);
        renderChecks();
    }

    function renderRegimeTable(run) {
        const head = '<tr><th>Regime</th><th>Δt from</th><th>Δt to</th><th>Segments</th><th>Points</th>'
            + '<th title="Straight-line fit of ln Δp′ against ln Δt over the whole regime">Slope m</th>'
            + '<th title="Smallest and largest segment slope in the regime">Segment m range</th>'
            + '<th title="Number of separate time intervals; 1 means continuous">Intervals</th><th>Median Δp′</th></tr>';
        const rows = run.res.regimes.map(r => `<tr>
            <td><span class="swatch" style="background:${color(r.cluster)}"></span>${regimeName(r.cluster)}</td>
            <td>${fmtT(r.t_start)}</td><td>${fmtT(r.t_end)}</td>
            <td>${r.n_windows}</td><td>${r.n_points}</td>
            <td>${fmt(r.slope_fit, 3)}</td>
            <td>${fmt(r.slope_min, 2)} … ${fmt(r.slope_max, 2)}</td>
            <td>${r.runs}</td><td>${fmt(r.dpd_median, 3)}</td></tr>`).join('');
        $('regimeTable').innerHTML = `<thead>${head}</thead><tbody>${rows}</tbody>`;
    }

    // Default plateau regime for the physics checks: the regime at the first derivative
    // minimum after the wellbore-storage hump (segment slopes: first + then -, then the
    // next - then +), which is where IARF sits. The slope of one line fitted through a
    // whole regime is not used: the WBS hump rises and falls, so that fit is near zero
    // and the hump would be taken for the plateau. If no hump-then-trough is found
    // (e.g. no storage hump), fall back to the regime of the segment with the smallest |m|.
    function plateauRegime(res) {
        const w = res.windows, s = w.map(x => x.slope), n = s.length;
        let hump = -1;
        for (let j = 0; j < n - 1; j++) { if (s[j] > 0 && s[j + 1] < 0) { hump = j; break; } }
        if (hump >= 0) {
            for (let k = hump + 1; k < n - 1; k++) {
                if (s[k] < 0 && s[k + 1] >= 0) {
                    const pick = Math.abs(s[k]) <= Math.abs(s[k + 1]) ? k : k + 1;
                    return { cluster: w[pick].cluster, how: 'derivative minimum' };
                }
            }
        }
        let best = 0;
        s.forEach((v, i) => { if (Math.abs(v) < Math.abs(s[best])) best = i; });
        return { cluster: w[best].cluster, how: 'flattest segment' };
    }
    function defaultPlateau(res) { return plateauRegime(res).cluster; }

    function populatePlateauSelect() {
        const res = state.current.res;
        const guess = plateauRegime(res);
        el.plateauRegime.innerHTML = res.regimes.map(r =>
            `<option value="${r.cluster}">${regimeName(r.cluster)}${r.cluster === guess.cluster ? ` (${guess.how})` : ''}</option>`).join('');
        el.plateauRegime.value = String(guess.cluster);
    }

    // Pure check computation, also used for history rows and comparison.
    function computeChecks(res, plateauIdx, tol) {
        const reg = res.regimes[plateauIdx];
        const wins = res.windows.filter(w => w.cluster === plateauIdx);
        const within = wins.filter(w => Math.abs(w.slope) <= tol).length;
        const plateau = {
            pass: within === wins.length,
            regime: plateauIdx, slope: reg.slope_fit, within, total: wins.length, tol,
        };
        const split = res.regimes.filter(r => r.runs > 1);
        const contiguity = { pass: split.length === 0, split: split.map(r => [r.cluster, r.runs]),
            fragments: res.regimes.reduce((s, r) => s + r.runs, 0) - res.regimes.length };
        let worst = res.centers[0];
        res.centers.forEach(c => { if (c.offset_frac > worst.offset_frac) worst = c; });
        const centers = { pass: worst.offset_frac <= CENTER_TOL, worst: worst.cluster, offset: worst.offset_frac, kind: res.center_kind };
        return { plateau, contiguity, centers };
    }

    function checkItem(pass, title, detail) {
        return `<li><span class="badge ${pass ? 'pass' : 'fail'}"><i class="fas fa-${pass ? 'check' : 'xmark'}" aria-hidden="true"></i>${pass ? 'Pass' : 'Fail'}</span>
            <span>${title}<span class="detail">${detail}</span></span></li>`;
    }

    function renderChecks() {
        const run = state.current;
        if (!run) return;
        const tol = Math.max(0, Number(el.plateauTol.value) || 0);
        const ck = computeChecks(run.res, Number(el.plateauRegime.value), tol);
        const p = ck.plateau, c = ck.contiguity, z = ck.centers;
        const items = [];
        items.push(checkItem(p.pass,
            `Plateau: every segment of ${regimeName(p.regime)} has |m| ≤ ${fmt(tol, 2)}`,
            `${p.within} of ${p.total} segments within tolerance; regime slope m = ${fmt(p.slope, 3)}.`));
        items.push(checkItem(c.pass,
            'Contiguity: each regime is one continuous time interval',
            c.pass ? 'No regime is split.' :
                c.split.map(([r, n]) => `${regimeName(r)} is split into ${n} intervals`).join('; ') + '.'));
        items.push(checkItem(z.pass,
            `Centres on the curve: every ${z.kind} within ${(CENTER_TOL * 100).toFixed(0)}% of the ln Δp′ range`,
            `Largest offset: ${regimeName(z.worst)}, ${(100 * z.offset).toFixed(1)}%.`
            + (z.kind === 'centroid' ? ' K-Means centroids are averages, so they can lie off the measured curve.' : ' Medoids are real segments, so they sit on the data.')));
        const left = run.data.lndt.length - run.res.leftover_start;
        const th = run.res.thin;
        if (th && th.mode !== 'off') items.push(`<li><span class="badge note">Note</span><span>Thinned by curve shape: ${th.n_used} of ${th.n_in} points used.</span></li>`);
        if (left > 0) items.push(`<li><span class="badge note">Note</span><span>${left} point(s) at late time were not clustered because ${run.data.lndt.length} is not a multiple of w = ${run.params.window_size}.</span></li>`);
        $('checksList').innerHTML = items.join('');
    }
    el.plateauRegime.addEventListener('change', renderChecks);
    el.plateauTol.addEventListener('input', renderChecks);

    // ------------------------------------------------------------------
    // Segmentation tab
    // ------------------------------------------------------------------
    function renderSegmentation() {
        const run = state.current;
        if (!run) return;
        const { t, wins } = runGeometry(run);
        const d = run.data, res = run.res;
        const traces = [{
            x: t, y: d.dp_dlndt, type: 'scatter', mode: 'markers', name: 'Δp′',
            marker: { color: '#ffffff', size: 6, line: { color: '#555555', width: 1 } },
            hovertemplate: 'Δt = %{x:.4g}<br>Δp′ = %{y:.4g}<extra></extra>',
        }];
        for (let c = 0; c < res.k; c++) {
            const xs = [], ys = [], txt = [];
            wins.filter(w => w.cluster === c).forEach(w => {
                const a = d.lndt[w.start], b = d.lndt[w.end - 1];
                [a, (a + b) / 2, b].forEach(lx => {
                    xs.push(Math.exp(lx)); ys.push(Math.exp(w.intercept + w.slope * lx));
                    txt.push(`Segment ${w.index + 1} (${regimeName(c)})<br>m = ${fmt(w.slope, 3)}${w.inverted_block ? '<br>in inverted-V block' : ''}`);
                });
                xs.push(null); ys.push(null); txt.push('');
            });
            traces.push({ x: xs, y: ys, text: txt, type: 'scatter', mode: 'lines', name: `${regimeName(c)} segment fits`,
                line: { color: color(c), width: 3 }, hovertemplate: '%{text}<extra></extra>' });
        }
        if (res.leftover_start < t.length) {
            traces.push({ x: t.slice(res.leftover_start), y: d.dp_dlndt.slice(res.leftover_start), type: 'scatter', mode: 'markers',
                name: 'not clustered (leftover)', marker: { color: '#c8c8c8', size: 7 } });
        }
        const shapes = [];
        wins.forEach((w, i) => {
            if (i > 0) {
                shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: t[w.start], x1: t[w.start], y0: 0, y1: 1,
                    line: { color: '#b5b5b5', width: 1, dash: 'dot' }, layer: 'below' });
            }
            if (w.inverted_block) {
                const x1 = i + 1 < wins.length ? t[wins[i + 1].start] : t[w.end - 1];
                shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: t[w.start], x1, y0: 0, y1: 1,
                    fillcolor: 'rgba(98,80,214,0.10)', line: { width: 0 }, layer: 'below' });
            }
        });
        const flagged = wins.filter(w => w.inverted_block).length;
        const title = `${wins.length} segments of w = ${run.params.window_size} points`
            + (res.backbone === 'kmedoids' ? `, ${flagged} flagged as inverted-V (p = ${run.params.p}`
                + (earlyLabel(run.params) ? `, reach ${earlyLabel(run.params) === 'auto' ? 'automatic' : 'index ' + earlyLabel(run.params)})` : ')') : '');
        draw('segPlot', traces, baseLayout({
            title: { text: title }, shapes,
            xaxis: axis('Δt (hr)', 'log'), yaxis: axis('Δp′', 'log'),
        }));
    }

    // ------------------------------------------------------------------
    // Elbow tab
    // ------------------------------------------------------------------
    function renderElbow() {
        const run = state.current;
        if (!run || !run.res.elbow) return;
        const e = run.res.elbow;
        const gainRule = e.rule === 'marginal_gain';
        const kOpt = e.k_opt;
        const iOpt = e.k_values.indexOf(kOpt);
        const kMax = e.k_values[e.k_values.length - 1];
        const traces = [
            { x: e.k_values, y: e.scores, type: 'scatter', mode: 'lines+markers', name: 'Distortion J(k)',
              line: { color: '#2a78d6', width: 2 }, marker: { size: 8, symbol: 'square' },
              hovertemplate: 'k = %{x}<br>J = %{y:.4g}<extra></extra>', xaxis: 'x', yaxis: 'y' },
            { x: [kOpt], y: [e.scores[iOpt]], type: 'scatter', mode: 'markers', name: `chosen k = ${kOpt}`,
              marker: { size: 16, color: 'rgba(0,0,0,0)', line: { color: INK, width: 2.5 } },
              hoverinfo: 'skip', xaxis: 'x', yaxis: 'y' },
        ];
        let lowerTitle, lowerRange, cut, cutLabel;
        if (gainRule) {
            // Bars: share of J(1) removed by the k-th cluster; the chosen k is dark.
            const ks = e.k_values.slice(1), gs = e.gains.slice(1);
            traces.push({ x: ks, y: gs, type: 'bar', name: 'gain gₖ', xaxis: 'x2', yaxis: 'y2',
                marker: { color: ks.map(k => (k === kOpt ? INK : (gs[ks.indexOf(k)] >= e.tau ? '#8d99a6' : '#d5dbe1'))) },
                text: gs.map(g => g.toFixed(3)), textposition: 'outside', textfont: { size: 10, color: INK }, cliponaxis: false,
                hovertemplate: 'k = %{x}<br>gₖ = %{y:.4f} of J(1)<extra></extra>' });
            lowerTitle = 'gₖ = (Jₖ₋₁ − Jₖ) / J₁';
            lowerRange = [0, 1.18 * Math.max(e.tau, ...gs, 0.01)];
            cut = e.tau; cutLabel = `τ = ${e.tau.toFixed(3)}`;
        } else {
            const ratioK = [], ratioV = [];
            e.ratios.forEach((r, i) => { if (r !== null) { ratioK.push(e.k_values[i]); ratioV.push(r); } });
            traces.push({ x: ratioK, y: ratioV, type: 'scatter', mode: 'lines+markers+text', name: 'slope ratio rₖ',
                text: ratioV.map(v => v.toFixed(2)), textposition: 'top center', textfont: { size: 10, color: INK },
                line: { color: '#eb6834', width: 2 }, marker: { size: 8 },
                hovertemplate: 'k = %{x}<br>rₖ = %{y:.3f}<extra></extra>', xaxis: 'x2', yaxis: 'y2' });
            lowerTitle = 'rₖ = |ΔJₖ₊₁| / |ΔJₖ|';
            lowerRange = [0, 1.2 * Math.max(e.threshold, ...ratioV, 0.01)];
            cut = e.threshold; cutLabel = `m_ratio = ${e.threshold.toFixed(2)}`;
        }
        const shapes = [
            { type: 'line', xref: 'x2', yref: 'y2', x0: e.k_values[0] - 0.5, x1: kMax + 0.5, y0: cut, y1: cut,
              line: { color: '#d03b3b', width: 1.5, dash: 'dash' } },
            { type: 'line', xref: 'x', yref: 'paper', x0: kOpt, x1: kOpt, y0: 0, y1: 1, line: { color: '#9a9a9a', width: 1, dash: 'dot' } },
        ];
        const annotations = [
            { xref: 'x2', yref: 'y2', x: kMax + 0.5, y: cut, xanchor: 'right', yanchor: 'bottom',
              text: cutLabel, showarrow: false, font: { size: 11, color: '#d03b3b' } },
        ];
        const ruleName = gainRule ? 'Marginal-gain rule' : 'Slope-ratio elbow';
        const layout = baseLayout({
            title: { text: `${ruleName} (${METHOD_LABEL[run.res.backbone]} backbone)` },
            xaxis: axis('', 'linear', { dtick: 1, matches: 'x2', anchor: 'y', range: [0.5, kMax + 0.5] }),
            yaxis: axis('distortion J(k)', 'linear', { domain: [0.52, 1] }),
            xaxis2: axis('number of clusters k', 'linear', { dtick: 1, anchor: 'y2', range: [0.5, kMax + 0.5] }),
            yaxis2: axis(lowerTitle, 'linear', { domain: [0, 0.4], anchor: 'x2', range: lowerRange }),
            shapes, annotations, bargap: 0.35,
        });
        draw('elbowPlot', traces, layout);

        // Explain the choice, and what the other rule would have picked.
        const elbowOther = e.elbow_k === e.fallback_k && !e.ratios.some(r => r !== null && r < e.threshold)
            ? `the slope-ratio elbow (m_ratio = ${e.threshold.toFixed(2)}) finds no elbow and would fall back to k = ${e.fallback_k}`
            : `the slope-ratio elbow (m_ratio = ${e.threshold.toFixed(2)}) would choose k = ${e.elbow_k}`;
        const gainOther = `the marginal-gain rule (τ = ${e.tau.toFixed(3)}) would choose k = ${e.gain_k}`;
        const banner = $('elbowBanner');
        if (gainRule) {
            if (e.detected) {
                banner.className = 'banner show info';
                banner.textContent = `k = ${kOpt} was chosen: it is the largest k whose added cluster still removes at least τ = ${e.tau.toFixed(3)} of J(1) `
                    + `(g${sub(kOpt)} = ${e.gains[iOpt].toFixed(3)}). For comparison, ${elbowOther}.`;
            } else {
                banner.className = 'banner show warn';
                banner.textContent = `No added cluster removes at least τ = ${e.tau.toFixed(3)} of J(1), so k = 1. Lower τ to allow more clusters.`;
            }
        } else if (e.detected) {
            banner.className = 'banner show info';
            banner.textContent = `k = ${kOpt} was chosen: it is the first k where rₖ < m_ratio (r${sub(kOpt)} = ${e.ratios[iOpt].toFixed(3)} < ${e.threshold.toFixed(2)}). `
                + `For comparison, ${gainOther}.`;
        } else {
            banner.className = 'banner show warn';
            banner.textContent = `No k had rₖ < m_ratio = ${e.threshold.toFixed(2)}, so the app fell back to the default k = ${e.fallback_k}. `
                + `This k was not detected from the data. For comparison, ${gainOther}.`;
        }
    }
    function sub(n) { return String(n).split('').map(ch => '₀₁₂₃₄₅₆₇₈₉'[Number(ch)]).join(''); }

    // ------------------------------------------------------------------
    // Compare tab
    // ------------------------------------------------------------------
    $('pinRun').addEventListener('click', () => {
        if (!state.current) return;
        state.pinned = state.current;
        invalidate(['compare']);
        showNotification(`Run #${state.current.id} pinned. Change data or settings, then open Compare.`, 'success');
    });

    function smallRunPlot(div, run) {
        const node = $(div);
        if (node.querySelector('.empty-state')) node.innerHTML = '';
        const { shapes, annotations } = bandShapesAndLabels(run, true);
        draw(div, regimeTraces(run, { showDp: false, markerSize: 6, starSize: 12 }), baseLayout({
            title: { text: `#${run.id}: ${methodLabel(run)}, k = ${run.res.k}`, font: { size: 13 } },
            xaxis: axis('Δt (hr)', 'log'), yaxis: axis('Δp′', 'log'),
            shapes, annotations, showlegend: false, height: 380, margin: { l: 60, r: 10, t: 60, b: 50 },
        }));
    }

    function runInfo(run) {
        const p = run.params;
        const ck = computeChecks(run.res, defaultPlateau(run.res), Number(el.plateauTol.value) || 0.05);
        const mark = ok => ok ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>';
        const weights = `λE ${p.lambda_e}, λθ ${p.lambda_p}, λτ ${p.beta}` + (run.res.backbone === 'kmedoids'
            ? `, λΛ ${p.gamma_block}, p ${p.p}` + (earlyLabel(p) ? `, i_early ${earlyLabel(p)}` : '') : '');
        return `<dl>
            <dt>Data</dt><dd>${escapeHtml(run.dataset)}${run.params.thin === 'curve' ? ` (thinned, ${run.data.lndt.length} points)` : ''}</dd>
            <dt>Method</dt><dd>${escapeHtml(methodLabel(run))}</dd>
            <dt>k, w</dt><dd>${run.res.k}${p.method === 'semi_automated' ? ` (${escapeHtml(ruleLabel(p))})` : ''}, ${p.window_size}</dd>
            <dt>Weights</dt><dd>${weights}</dd>
            <dt>Plateau</dt><dd>${mark(ck.plateau.pass)} ${regimeName(ck.plateau.regime)} m = ${fmt(ck.plateau.slope, 3)}</dd>
            <dt>Contiguous</dt><dd>${mark(ck.contiguity.pass)} ${ck.contiguity.fragments} extra interval(s)</dd>
            <dt>Centres</dt><dd>${mark(ck.centers.pass)} max offset ${(100 * ck.centers.offset).toFixed(1)}%</dd>
        </dl>`;
    }

    function renderCompare() {
        const note = $('compareNote');
        const cur = state.current, pin = state.pinned;
        if (cur) { smallRunPlot('compareCurrentPlot', cur); $('compareCurrentInfo').innerHTML = runInfo(cur); }
        if (pin) {
            smallRunPlot('comparePinnedPlot', pin);
            $('comparePinnedInfo').innerHTML = runInfo(pin);
            note.textContent = pin === cur
                ? 'The pinned run is the current run. Change data or parameters and run again to compare.'
                : 'Plateau check uses each run\'s default plateau regime (the derivative minimum after the hump) and the tolerance set on the Results tab.';
        } else {
            Plotly.purge('comparePinnedPlot');
            $('comparePinnedPlot').innerHTML = '<div class="empty-state" style="min-height:360px">Nothing pinned yet.</div>';
            $('comparePinnedInfo').innerHTML = '';
            note.textContent = 'Pin a run on the Results tab, then change data or parameters and run again.';
        }
    }

    // ------------------------------------------------------------------
    // Run history (kept in localStorage)
    // ------------------------------------------------------------------
    function loadHistory() {
        try {
            const raw = window.localStorage.getItem(HISTORY_KEY);
            const arr = raw ? JSON.parse(raw) : [];
            return Array.isArray(arr) ? arr : [];
        } catch (e) { return []; }
    }
    function saveHistory() {
        try { window.localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history.slice(-HISTORY_MAX))); } catch (e) { /* storage unavailable */ }
    }

    function addHistoryEntry(run, auto) {
        const res = run.res, p = run.params;
        const tol = Number(el.plateauTol.value) || 0.05;
        const ck = computeChecks(res, defaultPlateau(res), tol);
        const entry = {
            auto: !!auto,
            run: run.id,
            time: run.time.toISOString(),
            dataset: run.dataset,
            method: p.method,
            backbone: res.backbone,
            k: res.k,
            k_detected: res.elbow ? res.elbow.detected : null,
            params: p,
            plateau_regime: regimeName(ck.plateau.regime),
            plateau_slope: ck.plateau.slope,
            plateau_tol: tol,
            plateau_pass: ck.plateau.pass,
            contiguous_pass: ck.contiguity.pass,
            extra_intervals: ck.contiguity.fragments,
            centers_pass: ck.centers.pass,
            max_center_offset_pct: Number((100 * ck.centers.offset).toFixed(2)),
        };
        // Automatic updates while a student is adjusting a slider count as one run.
        // Only consecutive changes to the same single setting are merged, so a
        // change of method, rule or a different slider always starts a new row.
        const last = state.history[state.history.length - 1];
        const changed = last ? Object.keys(entry.params).filter(key => String(entry.params[key]) !== String(last.params[key])) : [];
        entry.changed = changed.length === 1 ? changed[0] : null;
        if (auto && last && last.auto && last.dataset === entry.dataset && (run.time - new Date(last.time)) < 4000
            && entry.changed && (last.changed === entry.changed || changed.length === 0)) {
            state.history[state.history.length - 1] = entry;
        } else if (auto && last && last.auto && last.dataset === entry.dataset && changed.length === 0) {
            state.history[state.history.length - 1] = entry;   // identical settings: replace
        } else {
            state.history.push(entry);
        }
        if (state.history.length > HISTORY_MAX) state.history = state.history.slice(-HISTORY_MAX);
        saveHistory();
        renderHistory();
    }

    function renderHistory() {
        const table = $('historyTable');
        const h = state.history;
        $('exportHistory').disabled = h.length === 0;
        $('clearHistory').disabled = h.length === 0;
        if (!h.length) { table.innerHTML = '<tbody><tr><td class="left muted">No runs yet.</td></tr></tbody>'; return; }
        const head = `<tr><th>#</th><th class="left">Time</th><th class="left">Data</th><th class="left">Method</th><th>k</th><th>w</th>
            <th>λE</th><th>λθ</th><th>λτ</th><th>λΛ</th><th>p</th><th>i_early</th><th class="left">k rule</th><th>δ</th><th>m₀</th>
            <th>Plateau m</th><th>Contiguous</th><th>Centres</th><th></th></tr>`;
        const tick = ok => ok ? '✓' : '✗';
        const rows = h.slice().reverse().map((r, i) => {
            const p = r.params, med = r.backbone === 'kmedoids', semi = r.method === 'semi_automated';
            const method = METHOD_LABEL[r.method] + (semi ? ` / ${METHOD_LABEL[r.backbone]}` : '');
            const k = semi ? `${r.k}${r.k_detected === false ? '*' : ''}` : r.k;
            return `<tr>
                <td>${r.run}</td><td class="left">${new Date(r.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td>
                <td class="left">${escapeHtml(r.dataset)}${p.thin === 'curve' ? ' · thinned' : ''}</td><td class="left">${escapeHtml(method)}</td>
                <td title="${r.k_detected === false ? (p.k_rule === 'marginal_gain' ? 'k = 1: no cluster reached τ' : 'fallback k: no elbow detected') : ''}">${k}</td><td>${p.window_size}</td>
                <td>${p.lambda_e}</td><td>${p.lambda_p}</td><td>${p.beta}</td>
                <td>${med ? p.gamma_block : '–'}</td><td>${med ? p.p : '–'}</td><td>${med && earlyLabel(p) ? earlyLabel(p) : '–'}</td>
                <td class="left">${semi ? escapeHtml(ruleLabel(p)) : '–'}</td>
                <td>${med ? p.delta : '–'}</td><td>${med ? p.threshold : '–'}</td>
                <td class="${r.plateau_pass ? 'ok' : 'bad'}" title="${r.plateau_regime}, tolerance ${r.plateau_tol}">${tick(r.plateau_pass)} ${fmt(r.plateau_slope, 3)}</td>
                <td class="${r.contiguous_pass ? 'ok' : 'bad'}">${tick(r.contiguous_pass)}${r.contiguous_pass ? '' : ' +' + r.extra_intervals}</td>
                <td class="${r.centers_pass ? 'ok' : 'bad'}">${tick(r.centers_pass)} ${r.max_center_offset_pct}%</td>
                <td><button class="link-btn" data-restore="${h.length - 1 - i}">Restore</button></td></tr>`;
        }).join('');
        table.innerHTML = `<thead>${head}</thead><tbody>${rows}</tbody>`;
    }

    $('historyTable').addEventListener('click', e => {
        const btn = e.target.closest('[data-restore]');
        if (!btn) return;
        const entry = state.history[Number(btn.dataset.restore)];
        if (!entry) return;
        restoreParams(entry.params);
        showNotification(`Settings of run #${entry.run} restored.`, 'success');
    });

    $('clearHistory').addEventListener('click', () => {
        if (!window.confirm('Delete all recorded runs from this browser?')) return;
        state.history = [];
        saveHistory();
        renderHistory();
    });

    $('exportHistory').addEventListener('click', () => {
        const cols = ['run', 'time', 'dataset', 'thinning', 'method', 'backbone', 'k', 'k_detected',
            'window_size', 'n_clusters_requested', 'lambda_e', 'lambda_theta', 'lambda_tau', 'lambda_Lambda', 'p_vsize',
            'early_time_index', 'k_rule', 'm_ratio', 'tau', 'delta', 'm0_near_zero_slope',
            'plateau_regime', 'plateau_slope', 'plateau_tol', 'plateau_pass',
            'contiguous_pass', 'extra_intervals', 'centers_pass', 'max_center_offset_pct'];
        const med = r => r.backbone === 'kmedoids';
        const semi = r => r.method === 'semi_automated';
        const rows = state.history.map(r => {
            const p = r.params;
            return [r.run, r.time, r.dataset, p.thin === 'curve' ? 'curve' : 'off', r.method, r.backbone, r.k, r.k_detected === null ? '' : r.k_detected,
                p.window_size, semi(r) ? '' : p.n_clusters, p.lambda_e, p.lambda_p, p.beta,
                med(r) ? p.gamma_block : '', med(r) ? p.p : '',
                med(r) && earlyLabel(p) ? earlyLabel(p) : '',
                semi(r) ? (p.k_rule || 'elbow') : '',
                semi(r) && p.k_rule !== 'marginal_gain' ? p.threshold_elbow : '',
                semi(r) && p.k_rule === 'marginal_gain' ? p.tau : '',
                med(r) ? p.delta : '', med(r) ? p.threshold : '',
                r.plateau_regime, r.plateau_slope, r.plateau_tol, r.plateau_pass,
                r.contiguous_pass, r.extra_intervals, r.centers_pass, r.max_center_offset_pct];
        });
        const csvCell = v => {
            const s = String(v === null || v === undefined ? '' : v);
            return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
        };
        const csv = [cols].concat(rows).map(r => r.map(csvCell).join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'flow_regime_runs.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });

    // ------------------------------------------------------------------
    // Presentation mode
    // ------------------------------------------------------------------
    function setPresent(on) {
        state.present = on;
        document.documentElement.classList.toggle('present', on);
        $('presentToggle').setAttribute('aria-pressed', String(on));
        try { window.localStorage.setItem('fri_present', on ? '1' : '0'); } catch (e) { /* ignore */ }
        // Redraw every plot at the new size.
        Object.keys(RENDERERS).forEach(n => state.dirty.add(n));
        if (state.data || state.current) {
            state.dirty.delete(state.activeTab);
            RENDERERS[state.activeTab]();
        }
        resizeVisible();
    }
    $('presentToggle').addEventListener('click', () => setPresent(!state.present));
    try { if (window.localStorage.getItem('fri_present') === '1') setPresent(true); } catch (e) { /* ignore */ }

    renderHistory();
});
