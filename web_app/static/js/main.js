document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    let isDataUploaded = false; 
    const previewSection = document.getElementById('preview-controls');
    const previewType = document.getElementById('previewType');
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const sheetNameInput = document.getElementById('sheetName');
    const clusteringMethod = document.getElementById('clusteringMethod');
    const updatePlotBtn = document.getElementById('updatePlot');
    const elbowPlotContainer = document.getElementById('elbowPlotContainer');

    // Slider elements
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

    // Event Listeners
    uploadForm.addEventListener('submit', handleFileUpload);
    previewType.addEventListener('change', updatePreviewPlot);
    clusteringMethod.addEventListener('change', handleMethodChange);
    document.querySelectorAll('input[name="backboneMethod"]').forEach(radio => {
        radio.addEventListener('change', handleMethodChange);
    });
    updatePlotBtn.addEventListener('click', updatePlots);

    // Initial Slider Value Sync
    Object.entries(sliders).forEach(([key, slider]) => {
        if (slider && slider.type !== 'hidden') {
            slider.addEventListener('input', (e) => {
                // Find the specific span to update based on the target ID in the HTML
                const value = e.target.value;
                if (key === 'nClusters') document.getElementById('n-clusters-val').innerText = value;
                if (key === 'windowSize') document.getElementById('window-val').innerText = value;
                if (key === 'lambdaE') document.getElementById('le-val').innerText = value;
                if (key === 'lambdaP') document.getElementById('lp-val').innerText = value;
                if (key === 'beta') document.getElementById('beta-val').innerText = value;
                if (key === 'gammaBlock') document.getElementById('gamma-val').innerText = value;
                if (key === 'p') document.getElementById('p-val').innerText = value;
                if (key === 'thresholdElbow') document.getElementById('t-elbow-val').innerText = value;
            });
        }
    });

    async function handleFileUpload(e) {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('sheet_name', sheetNameInput.value);

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            if (response.ok) {
                isDataUploaded = true;
                updatePlotBtn.disabled = false;
                previewSection.style.display = 'block';
                updatePreviewPlot();
                showNotification('Upload Success!', 'success');
            }
        } catch (error) { showNotification('Connection error', 'error'); }
    }

    async function updatePreviewPlot() {
        if (!isDataUploaded) return;
        const selectedType = previewType.value;
        
        const response = await fetch('/get_preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plot_type: selectedType })
        });
        const data = await response.json();

        const isLogX = selectedType.includes('Semi-Log') || selectedType.includes('Log-Log');
        const isLogY = selectedType.includes('Log-Log');

        let traces = [];
        if (selectedType.includes('Log-Log')) {
            traces.push({ x: data.x, y: data.y1, name: 'ΔP', mode: 'markers', marker: {color: 'blue', size: 5} });
            traces.push({ x: data.x, y: data.y2, name: 'Derivative', mode: 'markers', marker: {color: 'red', symbol: 'x', size: 5} });
        } else {
            traces.push({ x: data.x, y: data.y, name: 'ΔP', mode: 'lines+markers', line: {color: '#2196F3', width: 1}, marker: {size: 3} });
        }

        const layout = {
            title: selectedType,
            xaxis: { type: isLogX ? 'log' : 'linear', title: 'Time (hours)', gridcolor: '#eee' },
            yaxis: { type: isLogY ? 'log' : 'linear', title: 'Pressure / Derivative', gridcolor: '#eee' },
            paper_bgcolor: '#fdfdfd',
            plot_bgcolor: '#ffffff'
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
            threshold: parseFloat(sliders.threshold.value)
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

        // 1. Plot the regimes
        const clusters = {};
        plotData.windows.forEach(w => {
            if (!clusters[w.cluster]) clusters[w.cluster] = { x: [], y: [] };
            w.data.forEach(pt => {
                clusters[w.cluster].x.push(pt[0]);
                clusters[w.cluster].y.push(pt[1]);
            });
            clusters[w.cluster].x.push(null);
            clusters[w.cluster].y.push(null);
        });

        Object.keys(clusters).forEach((cID, index) => {
            traces.push({
                x: clusters[cID].x,
                y: clusters[cID].y,
                name: `Regime ${parseInt(cID) + 1}`,
                mode: 'lines+markers',
                line: { color: colors[index % colors.length], width: 2 },
                marker: { size: 4 }
            });    
        });

        // 2. NEW FIX: Plot Centers/Stars using vis_centers
        // These are now guaranteed to be inside the plot range
        if (plotData.vis_centers) {
            traces.push({
                x: plotData.vis_centers.map(c => c[0]),
                y: plotData.vis_centers.map(c => c[1]),
                name: 'Regime Centers',
                mode: 'markers',
                marker: {
                    symbol: 'star',
                    size: 15,
                    color: 'black',
                    line: { color: 'white', width: 1 }
                }
            });
        }
        
        const layout = {
            title: 'Flow Regime Identification (Normalized)',
            xaxis: { title: 'ln(Δt) [scaled]', range: [-1.1, 1.1] },
            yaxis: { title: 'ln(Derivative) [scaled]', range: [-1.1, 1.1] },
            showlegend: true
        };

        Plotly.newPlot('clusterPlot', traces, layout);
    }

    function updateElbowPlot(elbowData) {
        const trace = {
            x: elbowData.k_values, y: elbowData.k_scores,
            mode: 'lines+markers', name: 'Distortion',
            line: { color: '#1f77b4', width: 2 },
            marker: { color: '#1f77b4', size: 10, symbol: 'square' }
        };

        const layout = {
            title: 'Elbow Method for Optimal k',
            xaxis: { title: 'k', tickmode: 'linear' },
            yaxis: { title: 'distortion score' },
            annotations: [{
                xref: 'paper', yref: 'paper', x: 0.95, y: 0.95,
                text: `Elbow Value: ${elbowData.elbow_value}`,
                showarrow: false, bordercolor: 'black', borderwidth: 1, bgcolor: 'white', padding: 10
            }],
            shapes: [{
                type: 'line', x0: elbowData.elbow_value, x1: elbowData.elbow_value,
                y0: Math.min(...elbowData.k_scores), y1: Math.max(...elbowData.k_scores),
                line: { color: 'silver', width: 2, dash: 'dash' }
            }]
        };
        Plotly.newPlot('elbowPlot', [trace], layout);
    }

    function handleMethodChange() {
        const method = clusteringMethod.value;
        const nClustersContainer = document.getElementById('nClustersContainer');
        const backboneUI = document.getElementById('backbone-ui');
        const gammaContainer = document.getElementById('gammaContainer');
        const pContainer = document.getElementById('pContainer');

        nClustersContainer.style.display = (method === 'semi_automated') ? 'none' : 'block';
        backboneUI.style.display = (method === 'semi_automated') ? 'block' : 'none';

        const backboneRadio = document.querySelector('input[name="backboneMethod"]:checked');
        const isKmedoids = (method === 'kmedoids' || (method === 'semi_automated' && backboneRadio.value === 'kmedoids'));
        
        gammaContainer.style.display = isKmedoids ? 'block' : 'none';
        pContainer.style.display = isKmedoids ? 'block' : 'none';
    }

    function showNotification(msg, type) {
        const n = document.createElement('div');
        n.className = `notification ${type}`;
        n.innerText = msg;
        document.body.appendChild(n);
        setTimeout(() => n.remove(), 4000);
    }
    
    // Initial call to set visibility
    handleMethodChange();
});
