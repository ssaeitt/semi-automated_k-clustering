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

    // Slider elements (updated to match your new HTML IDs)
    const sliders = {
        nClusters: document.getElementById('nClusters'),
        windowSize: document.getElementById('windowSize'),
        lambdaE: document.getElementById('lambdaE'),
        lambdaP: document.getElementById('lambdaP'),
        beta: document.getElementById('beta'),
        gammaBlock: document.getElementById('gammaBlock'),
        p: document.getElementById('p'),
        delta: document.getElementById('delta'),
        threshold: document.getElementById('threshold')
    };

    // Event Listeners
    uploadForm.addEventListener('submit', handleFileUpload);
    previewType.addEventListener('change', updatePreviewPlot);
    clusteringMethod.addEventListener('change', handleMethodChange);
    document.querySelectorAll('input[name="backboneMethod"]').forEach(radio => {
        radio.addEventListener('change', handleMethodChange);
    });
    updatePlotBtn.addEventListener('click', updatePlots);

    // Initial Slider Value Display Sync
    Object.entries(sliders).forEach(([key, slider]) => {
        if (slider && slider.type !== 'hidden') {
            slider.addEventListener('input', (e) => {
                const displayId = slider.getAttribute('oninput').match(/'([^']+)'/)[1];
                document.getElementById(displayId).innerText = e.target.value;
            });
        }
    });

    // 1. Updated File Upload Handler
    async function handleFileUpload(e) {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('sheet_name', sheetNameInput.value);

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const data = await response.json();

            if (response.ok) {
                isDataUploaded = true;
                updatePlotBtn.disabled = false;
                previewSection.style.display = 'block';
                // Trigger initial preview
                updatePreviewPlot();
                showNotification('Upload Success! Data preview available.', 'success');
            } else {
                showNotification(data.error || 'Upload failed', 'error');
            }
        } catch (error) {
            showNotification('Server connection error', 'error');
        }
    }

    // 2. NEW: Async Preview Plot Function
    async function updatePreviewPlot() {
        if (!isDataUploaded) return;

        const selectedType = previewType.value;
        try {
            const response = await fetch('/get_preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ plot_type: selectedType })
            });
            const data = await response.json();

            let traces = [];
            const commonStyle = { mode: 'lines+markers', marker: { size: 4 } };

            if (selectedType === "Normal Plot (p vs t)" || selectedType === "Semi-Log Plot (dp vs lnt)") {
                traces.push({
                    x: data.x, y: data.y, name: 'Delta P', ...commonStyle,
                    line: { color: '#2196F3' }
                });
            } else {
                // Log-Log Diagnostic Plot
                traces.push({
                    x: data.x, y: data.y1, name: 'Delta P', mode: 'markers',
                    marker: { color: 'blue', size: 5, opacity: 0.6 }
                });
                traces.push({
                    x: data.x, y: data.y2, name: 'Derivative', mode: 'markers',
                    marker: { color: 'red', symbol: 'x', size: 5 }
                });
            }

            const layout = {
                title: selectedType,
                xaxis: { 
                    type: (selectedType.includes('Log')) ? 'log' : 'linear',
                    title: selectedType.includes('lnt') ? 'ln(t)' : 'Time'
                },
                yaxis: { 
                    type: (selectedType.includes('Log-Log')) ? 'log' : 'linear',
                    title: 'Pressure / Derivative'
                },
                paper_bgcolor: '#fdfdfd'
            };

            Plotly.newPlot('clusterPlot', traces, layout, { responsive: true });
        } catch (e) {
            showNotification('Error loading preview', 'error');
        }
    }

    // 3. Updated Clustering Update Handler
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
        };

        if (method === 'semi_automated') {
            params.backbone_method = backboneRadio.value;
        }

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
                }
            }
        } catch (error) {
            showNotification('Clustering failed', 'error');
        }
    }

    function handleMethodChange() {
        // UI visibility is handled by the inline script in index.html
        elbowPlotContainer.style.display = 'none';
    }

    // Standard Plotly utility for results
    function updateClusterDisplay(plotData) {
        // (Use your existing Cluster plotting logic here, it works with the new labels)
        // Ensure you use the colors array as before to visualize the regimes.
    }

    function showNotification(msg, type) {
        const n = document.createElement('div');
        n.className = `notification ${type}`;
        n.innerText = msg;
        document.body.appendChild(n);
        setTimeout(() => n.remove(), 4000);
    }
});
