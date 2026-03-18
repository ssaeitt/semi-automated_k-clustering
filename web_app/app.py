from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import numpy as np
import os
import traceback
import requests
import json
from dotenv import load_dotenv
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
import auxiliary_functions as cf

load_dotenv()
app = Flask(__name__)

current_data = None

# API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def process_data(file_data, sheet_name='Sheet1'):
    try:
        if file_data.filename.endswith('.csv'):
            df = pd.read_csv(file_data)
        else:
            df = pd.read_excel(file_data, sheet_name=sheet_name)
        
        # Ensure we don't have zeros/negatives for logs
        df = df[df['dp_dlndt'] > 0].copy()
        
        lndt = df['lndt'].values
        dp = df['dp'].values
        dp_dlndt = df['dp_dlndt'].values
        lndp_dlndt = np.log(dp_dlndt)
        
        grad_2 = cf.bourdet_derivative(x=lndt, y=lndp_dlndt, L=0., transform_x=False, transform_y=False)
        
        # Scaling for clustering only
        x_norm = cf.min_max_scaler(lndt, limits=[-1,1])
        y_norm = cf.min_max_scaler(lndp_dlndt, limits=[-1,1])
        
        return {
            'time': np.exp(lndt), 
            'lndt': lndt, 
            'dp': dp, 
            'dp_dlndt': dp_dlndt,
            'lndp_dlndt': lndp_dlndt,
            'x_norm': x_norm, 
            'y_norm': y_norm
        }
    except Exception as e:
        print(f"Processing Error: {e}")
        return None

def create_windows(data_norm, data_real, window_size):
    """Creates windows storing both normalized (for ML) and real (for plotting) data."""
    windows = []
    n = data_norm.shape[0]
    for i in range(0, n - window_size + 1, window_size):
        window_norm = data_norm[i:i+window_size]
        window_real = data_real[i:i+window_size]
        if window_norm.shape[0] < 2: break
        
        median_norm = np.median(window_norm, axis=0)
        # Calculate slope on real coordinates for angular dissimilarity
        slope, _ = np.polyfit(window_real[:, 0], window_real[:, 1], 1) if np.ptp(window_real[:, 0]) != 0 else (0.0, 0)
        
        windows.append({
            'data_norm': window_norm,
            'data_real': window_real,
            'median_norm': median_norm,
            'slope': slope,
            'index': len(windows)
        })
    return windows

def assign_inverted_v_block(windows, p):
    for w in windows: w['inverted_block'] = False
    for i in range(len(windows) - p + 1):
        block = windows[i:i+p]
        if any(block[j]['slope'] > 0 and block[j+1]['slope'] < 0 for j in range(p - 1)):
            for j in range(i, i+p): windows[j]['inverted_block'] = True
    return windows

def custom_distance(w1, w2, D_max, T_max, le, lp, b, gb=1.0):
    idx_diff = abs(w1['index'] - w2['index'])
    # Euclidean in normalized space
    e_dist = np.linalg.norm(w1['data_norm'].flatten() - w2['data_norm'].flatten())
    norm_e = e_dist / D_max if D_max > 0 else e_dist
    # Angular (Slope)
    a_diff = abs(np.degrees(np.arctan(w1['slope'])) - np.degrees(np.arctan(w2['slope'])))
    if a_diff > 90: a_diff = 180 - a_diff
    norm_p = a_diff / 90.0
    # Temporal
    norm_t = (max(0, idx_diff - 1) / T_max) if T_max > 0 else max(0, idx_diff - 1)
    # Inverted-V Detection weight
    bonus = -gb if w1.get('inverted_block') and w2.get('inverted_block') else 0.0
    
    return max(le * norm_e + lp * norm_p + b * norm_t + bonus, 0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_preview', methods=['POST'])
def get_preview():
    plot_type = request.json.get('plot_type')
    if current_data is None: return jsonify({'error': 'No data'}), 400
    
    if plot_type == "Normal Plot (p vs t)":
        return jsonify({'x': current_data['time'].tolist(), 'y': current_data['dp'].tolist()})
    elif plot_type == "Semi-Log Plot (dp vs lnt)":
        # Crucial: return raw time so frontend 'log' axis works perfectly
        return jsonify({'x': current_data['time'].tolist(), 'y': current_data['dp'].tolist()})
    elif plot_type == "Log-Log Plot (Diagnostic)":
        return jsonify({
            'x': current_data['time'].tolist(), 
            'y1': current_data['dp'].tolist(), 
            'y2': current_data['dp_dlndt'].tolist()
        })
    return jsonify({'error': 'Invalid plot type'}), 400

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        params = request.json
        method = params.get('method')
        le, lp, b, gb = params.get('lambda_e', 1.0), params.get('lambda_p', 1.0), params.get('beta', 0.5), params.get('gamma_block', 1.0)
        w_size = int(params.get('window_size', 5))
        
        # Prepare windowed data (Norm for calc, Real for plotting)
        norm_stack = np.column_stack((current_data['x_norm'], current_data['y_norm']))
        real_stack = np.column_stack((current_data['lndt'], current_data['lndp_dlndt']))
        
        windows = create_windows(norm_stack, real_stack, w_size)
        
        if method in ['kmedoids', 'semi_automated']:
            assign_inverted_v_block(windows, int(params.get('p', 4)))
            
        D_max = 0
        for i in range(len(windows)):
            for j in range(i+1, len(windows)):
                d = np.linalg.norm(windows[i]['data_norm'].flatten() - windows[j]['data_norm'].flatten())
                if d > D_max: D_max = d
        T_max = windows[-1]['index'] if windows else 1
        
        n_w = len(windows)
        dist_mat = np.zeros((n_w, n_w))
        for i in range(n_w):
            for j in range(i, n_w):
                v = custom_distance(windows[i], windows[j], D_max, T_max, le, lp, b, gb)
                dist_mat[i, j] = dist_mat[j, i] = v
                
        elbow_data = None
        if method == 'kmeans':
            feats = [[le*w['median_norm'][0], le*w['median_norm'][1], lp*w['slope'], b*w['index']] for w in windows]
            X_s = StandardScaler().fit_transform(np.array(feats))
            labels = KMeans(n_clusters=int(params.get('n_clusters', 3)), random_state=42).fit_predict(X_s)
        elif method == 'kmedoids':
            labels = KMedoids(n_clusters=int(params.get('n_clusters', 3)), metric='precomputed', random_state=42).fit_predict(dist_mat)
        elif method == 'semi_automated':
            backbone = params.get('backbone_method', 'kmeans')
            if backbone == 'kmedoids':
                vis = KElbowVisualizer(KMedoids(metric='precomputed'), k=(2, 12)).fit(dist_mat)
            else:
                feats = [[le*w['median_norm'][0], le*w['median_norm'][1], lp*w['slope'], b*w['index']] for w in windows]
                X_s = StandardScaler().fit_transform(np.array(feats))
                vis = KElbowVisualizer(KMeans(), k=(2, 12)).fit(X_s)
            k_opt = vis.elbow_value_ if vis.elbow_value_ else 4
            elbow_data = {'k_values': vis.k_values_, 'k_scores': vis.k_scores_, 'elbow_value': int(k_opt)}
            labels = KMedoids(n_clusters=k_opt, metric='precomputed', random_state=42).fit_predict(dist_mat) if backbone == 'kmedoids' else KMeans(n_clusters=k_opt, random_state=42).fit_predict(X_s)

        # Chronological Re-indexing
        clusters = np.unique(labels)
        window_x_coords = np.array([w['data_real'][0,0] for w in windows])
        cluster_means = [np.mean(window_x_coords[labels == c]) for c in clusters]
        sorted_indices = np.argsort(cluster_means)
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
        labels = np.array([mapping[l] for l in labels])
        
        return jsonify({
            'plot_data': {
                'windows': [{'cluster': int(l), 'data': w['data_real'].tolist()} for l, w in zip(labels, windows)]
            }, 
            'elbow_data': elbow_data
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data
    file = request.files['file']
    sheet = request.form.get('sheet_name', 'Sheet1')
    res = process_data(file, sheet)
    if res:
        current_data = res
        return jsonify({'message': 'Upload Successful'})
    return jsonify({'error': 'Processing failed'}), 400

@app.route('/ai_search', methods=['POST'])
def ai_search():
    data = request.json
    query = data.get('query', '')
    if not query: return jsonify({'error': 'No query'}), 400
    headers = {'Authorization': f'Bearer {OPENROUTER_API_KEY}', 'Content-Type': 'application/json'}
    payload = {'model': 'deepseek/deepseek-r1:free', 'messages': [{'role': 'user', 'content': query}]}
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
