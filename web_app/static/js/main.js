document.addEventListener('DOMContentLoaded', function() {
    let isDataUploaded = false; 
    const previewSection = document.getElementById('preview-controls');
    const previewType = document.getElementById('previewType');
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const sheetNameInput = document.getElementById('sheetName');
    const clusteringMethod = document.getElementById('clusteringMethod');
    const updatePlotBtn = document.getElementById('updatePlot');
    const elbowPlotContainer = document.getElementById('elbowPlotContainer');

    const sliders = {
        nClusters: document.getElementById('nClusters'),
        windowSize: document.getElementById('windowSize'),
        lambdaE: document.getElementById('lambdaE'),
        lambdaP: document.getElementById('lambdaP'),
        beta: document.getElementById('beta'),
        gammaBlock: document.getElementById('gammaBlock'),
        p: document.getElementById('p'),
        delta: document.getElementById('delta'),
        threshold: document.getElementById('threshold'),
        thresholdElbow: document.getElementById('thresholdElbow')
    };

    uploadForm.addEventListener('submit', handleFileUpload);
    previewType.addEventListener('change', updatePreviewPlot);
    clusteringMethod.addEventListener('change', handleMethodChange);
    document.querySelectorAll('input[name="backboneMethod"]').forEach(radio => {
        radio.addEventListener('change', handleMethodChange);
    });
    updatePlotBtn.addEventListener('click', updatePlots);

    // Initial Slider Value Sync with Safety Checks
    Object.entries(sliders).forEach(([key, slider]) => {
        if (slider && slider.type !== 'hidden') {
            slider.addEventListener('input', (e) => {
                const value = e.target.value;
                if (key === 'nClusters') document.getElementById('n-clusters-val').innerText = value;
                if (key === 'windowSize') document.getElementById('window-val').innerText = value;
                if (key === 'lambdaE') document.getElementById('le-val').innerText = value;
                if (key === 'lambdaP') document.getElementById('lp-val').innerText = value;
                if (key === 'beta') document.getElementById('beta-val').innerText = value;
                if (key === 'gammaBlock') document.getElementById('gamma-val').innerText = value;
                if (key === 'p') document.getElementById('p-val').innerText = value;
                if (key === 'thresholdElbow') {
                    const el = document.getElementById('t-elbow-val');
                    if(el) el.innerText = value;
                }
            });
        }
    });

    async function handleFileUpload(e) {
        e.preventDefault();
        if (!fileInput.files.length) return;
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('sheet_name', sheetNameInput.value);

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            if (response.ok) {
                isDataUploaded = true;
                updatePlotBtn.disabled = false;

                updatePreviewPlot();
                showNotification('Upload Success!', 'success');
            }
        } catch (error) { showNotification('Connection error', 'error'); }
    }

    async function updatePreviewPlot() {
        if (!isDataUploaded) return;
        const response = await fetch('/get_preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plot_type: previewType.value })
        });
        const data = await response.json();

        const isLogX = previewType.value.includes('Semi-Log') || previewType.value.includes('Log-Log');
        const isLogY = previewType.value.includes('Log-Log');

        let traces = [];
        if (previewType.value.includes('Log-Log')) {
            traces.push({ x: data.x, y: data.y1, name: 'ΔP', mode: 'markers', marker: {color: 'blue', size: 5} });
            traces.push({ x: data.x, y: data.y2, name: 'Derivative', mode: 'markers', marker: {color: 'red', symbol: 'x', size: 5} });
        } else {
            traces.push({ x: data.x, y: data.y, name: 'ΔP', mode: 'lines+markers', line: {color: '#2196F3', width: 1}, marker: {size: 3} });
        }

        const layout = {
            title: previewType.value,
            xaxis: { type: isLogX ? 'log' : 'linear', title: 'Time (hours)' },
            yaxis: { type: isLogY ? 'log' : 'linear', title: 'Pressure / Derivative' }
        };
        Plotly.newPlot('clusterPlot', traces, layout);
    }

    async function updatePlots() {
        const method = clusteringMethod.value;
        const backboneRadio = document.querySelector('input[name="backboneMethod"]:checked');
        const params = {
            method: method,
            n_clusters: parseInt(sliders.nClusters.value),
            window_size: parseInt(sliders.windowSize.value),
            lambda_e: parseFloat(sliders.lambdaE.value),
            lambda_p: parseFloat(sliders.lambdaP.value),
            beta: parseFloat(sliders.beta.value),
            gamma_block: parseFloat(sliders.gammaBlock.value),
            p: parseInt(sliders.p.value),
            delta: parseFloat(sliders.delta.value),
            threshold: parseFloat(sliders.threshold.value),
            threshold_elbow: parseFloat(sliders.thresholdElbow.value)
        };
        if (method === 'semi_automated') params.backbone_method = backboneRadio.value;

        try {
            const response = await fetch('/cluster', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            const data = await response.json();
            if (response.ok) {
                updateClusterDisplay(data.plot_data);
                if (data.elbow_data) {
                    updateElbowPlot(data.elbow_data);
                    elbowPlotContainer.style.display = 'block';
                } else {
                    elbowPlotContainer.style.display = 'none';
                }
                showNotification('Clustering Complete', 'success');
            }
        } catch (error) { showNotification('Clustering failed', 'error'); }
    }

    function updateClusterDisplay(plotData) {
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'];
        const traces = [];
        const clusters = {};
        plotData.windows.forEach(w => {
            if (!clusters[w.cluster]) clusters[w.cluster] = { x: [], y: [] };
            w.data.forEach(pt => { clusters[w.cluster].x.push(pt[0]); clusters[w.cluster].y.push(pt[1]); });
            clusters[w.cluster].x.push(null); clusters[w.cluster].y.push(null);
        });

        Object.keys(clusters).forEach((cID, index) => {
            traces.push({
                x: clusters[cID].x, y: clusters[cID].y,
                name: `Regime ${parseInt(cID) + 1}`,
                mode: 'lines+markers',
                line: { color: colors[index % colors.length], width: 2 },
                marker: { size: 4 }
            });
        });

        if (plotData.vis_centers) {
            traces.push({
                x: plotData.vis_centers.map(c => c[0]),
                y: plotData.vis_centers.map(c => c[1]),
                name: 'Regime Centers',
                mode: 'markers',
                marker: { symbol: 'star', size: 15, color: 'black', line: { color: 'white', width: 1 } }
            });
        }
        Plotly.newPlot('clusterPlot', traces, { title: 'Results (Normalized)', xaxis: { range: [-1.1, 1.1] }, yaxis: { range: [-1.1, 1.1] } });
    }

    function updateElbowPlot(elbowData) {
        const trace = { x: elbowData.k_values, y: elbowData.k_scores, mode: 'lines+markers', marker: { symbol: 'square', size: 10 } };
        const layout = {
            title: 'Slope-Ratio Elbow Method',
            xaxis: { title: 'k' }, yaxis: { title: 'Distortion' },
            annotations: [{ xref: 'paper', yref: 'paper', x: 0.95, y: 0.95, text: `k_opt: ${elbowData.elbow_value}`, showarrow: false, bordercolor: 'black' }],
            shapes: [{ type: 'line', x0: elbowData.elbow_value, x1: elbowData.elbow_value, y0: Math.min(...elbowData.k_scores), y1: Math.max(...elbowData.k_scores), line: { dash: 'dash', color: 'silver' } }]
        };
        Plotly.newPlot('elbowPlot', [trace], layout);
    }

    function handleMethodChange() {
        const method = clusteringMethod.value;
        const isSemi = (method === 'semi_automated');
        document.getElementById('nClustersContainer').style.display = isSemi ? 'none' : 'block';
        document.getElementById('backbone-ui').style.display = isSemi ? 'block' : 'none';
        document.getElementById('elbowThresholdContainer').style.display = isSemi ? 'block' : 'none';
        
        const backbone = document.querySelector('input[name="backboneMethod"]:checked')?.value || 'kmeans';
        const showPhysics = (method === 'kmedoids' || (isSemi && backbone === 'kmedoids'));
        document.getElementById('gammaContainer').style.display = showPhysics ? 'block' : 'none';
        document.getElementById('pContainer').style.display = showPhysics ? 'block' : 'none';
    }

    function showNotification(msg, type) {
        const n = document.createElement('div');
        n.className = `notification ${type}`;
        n.innerText = msg;
        document.body.appendChild(n);
        setTimeout(() => n.remove(), 4000);
    }
    handleMethodChange();
});
