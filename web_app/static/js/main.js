document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    let rawData = null; // To store data for preprocessing plots
    const previewSection = document.getElementById('preview-controls');
    const previewType = document.getElementById('previewType');
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const sheetNameInput = document.getElementById('sheetName');
    const clusteringMethod = document.getElementById('clusteringMethod');
    const backboneSelector = document.querySelector('.backbone-selector');
    const backboneRadios = document.querySelectorAll('input[name="backboneMethod"]');
    const kmedoidsParams = document.querySelector('.kmedoids-params');
    const extraMetricsToggle = document.getElementById('extraMetricsToggle');
    const extraMetricsParams = document.querySelector('.extra-metrics-params');
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
        // Extra metrics sliders
        delta: document.getElementById('delta'),
        threshold: document.getElementById('threshold')
    };

    // Plot elements
    const clusterPlot = document.getElementById('clusterPlot');
    const elbowPlot = document.getElementById('elbowPlot');

    // Initialize plots
    let clusterPlotInstance = null;
    let elbowPlotInstance = null;

    // Event Listeners
    uploadForm.addEventListener('submit', handleFileUpload);
    clusteringMethod.addEventListener('change', handleMethodChange);
    backboneRadios.forEach(radio => {
        radio.addEventListener('change', handleBackboneChange);
    });
    updatePlotBtn.addEventListener('click', updatePlots);
    extraMetricsToggle.addEventListener('change', toggleExtraMetrics);
    
    // Add event listeners to all sliders
// --- Optimized Slider Listener ---
    Object.entries(sliders).forEach(([key, slider]) => {
        if (slider && slider.type !== 'hidden') { 
            slider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                // Look for the span in the same group as the slider
                const display = e.target.parentElement.querySelector('.slider-value');
                if (display) {
                    display.textContent = (key === 'p') ? value : value.toFixed(1);
                }
            });
        }
    });

    // --- Clean Visibility Logic ---
    function updateParametersVisibility(method, backbone) {
        const kmedoidsUI = document.getElementById('kmedoids-extra-ui');
        
        // The "Logic Gate": Show if main method is kmedoids
        // OR if we are in semi-automated mode AND the backbone is kmedoids
        const shouldShow = (method === 'kmedoids') ||
            (method === 'semi_automated' && backbone === 'kmedoids');

        if (kmedoidsUI) {
            kmedoidsUI.style.display = shouldShow ? 'block' : 'none';
        }
        // Handle the backbone selector visibility (only for semi-automated)
        backboneSelector.style.display = (method === 'semi_automated') ? 'block' : 'none';
    }

    function handleMethodChange(e) {
        const method = e.target.value;
        console.log("Method changed to:", method);
        
        updateParametersVisibility(method, getSelectedBackboneMethod());
        elbowPlotContainer.style.display = 'none';
    }

    // Helper to find which backbone radio button is checked
    function getSelectedBackboneMethod() {
        const selectedRadio = document.querySelector('input[name="backboneMethod"]:checked');
        return selectedRadio ? selectedRadio.value : 'kmeans'; 
    }
    
    function handleBackboneChange(e) {
        const backboneMethod = e.target.value;
        console.log("Backbone method changed to:", backboneMethod);
        updateParametersVisibility(clusteringMethod.value, backboneMethod);
    }
    // File Upload Handler
    async function handleFileUpload(e) {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('sheet_name', sheetNameInput.value);

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const data = await response.json();
        
            if (response.ok) {
                // Store the raw data for previewing
                rawData = data; 
                updatePlotBtn.disabled = false;
            
                // SHOW SECTION 2 and generate the first plot
                if (previewSection){
                    previewSection.style.display = 'block';
                }
                
                updatePreviewPlot(); 
            
                showNotification('Data loaded! You can now preview data in Section 2.', 'success');
            } else {
                showNotification(data.error || 'Error uploading file', 'error');
            }
        } catch (error) {
            showNotification('Error: ' + error.message, 'error');
        }
    }

    function updatePreviewPlot() {
        if (!rawData) return;

        const type = previewType.value;
        let traces = [];
        let layout = {
            title: '',
            xaxis: { title: 'Time', type: 'linear' },
            yaxis: { title: 'Pressure', type: 'linear' },
            margin: { t: 50, b: 50, l: 60, r: 30 }
        };

        if (type === 'normal') {
            traces.push({
                x: rawData.raw_t,
                y: rawData.raw_dp,
                mode: 'lines+markers',
                name: 'p vs t',
                line: { color: '#2196F3' }
            });
            layout.title = 'Normal Plot (Cartesian)';
        } 
        else if (type === 'semilog') {
            traces.push({
                x: rawData.raw_t,
                y: rawData.raw_dp,
                mode: 'lines+markers',
                name: 'p vs log t',
                line: { color: '#4CAF50' }
            });
            layout.title = 'Semi-Log Plot';
            layout.xaxis.type = 'log';
        } 
        else if (type === 'loglog') {
            traces.push({
                x: rawData.raw_t,
                y: rawData.raw_dp,
                mode: 'markers',
                name: 'Delta P',
                marker: { color: 'blue' }
            });
            traces.push({
                x: rawData.raw_t,
                y: rawData.raw_der,
                mode: 'markers',
                name: 'Derivative',
                marker: { color: 'red', symbol: 'x' }
            });
            layout.title = 'Log-Log Diagnostic Plot';
            layout.xaxis.type = 'log';
            layout.yaxis.type = 'log';
            layout.xaxis.title = 'dt';
            layout.yaxis.title = 'dp & dp\'';
        }

        Plotly.newPlot('clusterPlot', traces, layout, {responsive: true});
    }
        
        const trace = { x: x, y: y, mode: 'lines+markers', type: 'scatter', marker: {color: '#2196F3'} };
        const layout = {
            title: type.charAt(0).toUpperCase() + type.slice(1) + ' Plot',
            xaxis: { title: xTitle, type: xType },
            yaxis: { title: yTitle, type: yType }
        };

        Plotly.newPlot('clusterPlot', [trace], layout);
    }

    // Listen for when the student changes the Preview Dropdown
    previewType.addEventListener('change', updatePreviewPlot);
    
    // Update Plots Handler
    async function updatePlots() {
        const method = clusteringMethod.value;
        const backbone = getSelectedBackboneMethod(); // Define this clearly at the start
    
        const params = {
            method: method,
            n_clusters: parseInt(sliders.nClusters.value),
            window_size: parseInt(sliders.windowSize.value),
            lambda_e: parseFloat(sliders.lambdaE.value),
            lambda_p: parseFloat(sliders.lambdaP.value),
            beta: parseFloat(sliders.beta.value)
        };

        if (method === 'semi_automated') {
            params.backbone_method = backbone;
        }

        // FIX: Ensure K-Medoids logic triggers correctly for both standalone and backbone
        if (method === 'kmedoids' || (method === 'semi_automated' && backbone === 'kmedoids')) {
            params.gamma_block = parseFloat(sliders.gammaBlock.value);
            params.p = parseInt(sliders.p.value);
        
            // Also send extra metrics if the checkbox exists
            if (extraMetricsToggle && extraMetricsToggle.checked) {
                params.delta = parseFloat(sliders.delta.value);
                params.threshold = parseFloat(sliders.threshold.value);
            } else {
                params.delta = 0.1; // Default fallbacks
                params.threshold = 0.1;
            }
        }

        // Log parameters for debugging
        console.log("Sending parameters to server:", params);

        try {
            const response = await fetch('/cluster', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });

            const data = await response.json();
            
            if (response.ok) {
                updateClusterPlot(data.plot_data);
                
                // Update elbow plot if available (for semi-automated method)
                if (method === 'semi_automated' && data.elbow_data) {
                    console.log("Received elbow data:", data.elbow_data);
                    updateElbowPlot(data.elbow_data);
                    elbowPlotContainer.style.display = 'block';
                } else {
                    elbowPlotContainer.style.display = 'none';
                }
                
                showNotification("Clustering completed successfully!", "success");
            } else {
                showNotification(data.error || "Error performing clustering", "error");
            }
        } catch (error) {
            console.error("Error during plot update:", error);
            showNotification('Error updating plots: ' + error.message, 'error');
        }
    }

    // Plot Update Functions
    function updateClusterPlot(plotData) {
        console.log("Updating cluster plot with data:", plotData);
        
        // Clear the plot container
        const clusterPlotElement = document.getElementById('clusterPlot');
        clusterPlotElement.innerHTML = '';
        
        // Define a color palette
        const colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ];
        
        // Prepare the traces for plotting
        const traces = [];
        
        // Group windows by cluster
        const clusterGroups = {};
        plotData.windows.forEach(window => {
            const clusterLabel = window.cluster;
            if (!clusterGroups[clusterLabel]) {
                clusterGroups[clusterLabel] = [];
            }
            clusterGroups[clusterLabel].push(window);
        });
        
        // Sort clusters by their labels to ensure consistent coloring
        const sortedClusters = Object.keys(clusterGroups).sort((a, b) => parseInt(a) - parseInt(b));
        
        // For each cluster, create traces for its segments
        sortedClusters.forEach((clusterLabel, clusterIndex) => {
            const color = colors[clusterIndex % colors.length];
            const clusterName = `Cluster ${parseInt(clusterLabel) + 1}`;
            
            // Create a dummy point for the legend
            traces.push({
                x: [null],
                y: [null],
                mode: 'lines+markers',
                name: clusterName,
                line: { color: color, width: 2 },
                marker: { color: color },
                legendgroup: clusterName,
                showlegend: true
            });
            
            // Add traces for each window segment in this cluster
            clusterGroups[clusterLabel].forEach(window => {
                traces.push({
                    x: window.data.map(d => d[0]),
                    y: window.data.map(d => d[1]),
                    mode: 'lines+markers',
                    name: clusterName,
                    line: { color: color, width: 2 },
                    marker: {
                        color: color,
                        size: 4,
                        symbol: 'circle'
                    },
                    legendgroup: clusterName,
                    showlegend: false
                });
            });
        });
        
        // Add medoid points if available
        if (plotData.medoid_indices && plotData.medoid_indices.length > 0) {
            const medoidPoints = plotData.medoid_indices.map(i => {
                if (i < plotData.windows.length) {
                    return plotData.windows[i].median;
                }
                return null;
            }).filter(p => p !== null);
            
            if (medoidPoints.length > 0) {
                traces.push({
                    x: medoidPoints.map(m => m[0]),
                    y: medoidPoints.map(m => m[1]),
                    mode: 'markers',
                    name: 'Medoids',
                    marker: { 
                        size: 12, 
                        symbol: 'star',
                        color: '#000000',
                        line: { color: '#ffffff', width: 1 }
                    },
                    type: 'scatter'
                });
            }
        }
        
        // Add cluster centers if available
        if (plotData.centers && plotData.centers.length > 0) {
            traces.push({
                x: plotData.centers.map(c => c[0]),
                y: plotData.centers.map(c => c[1]),
                mode: 'markers',
                name: 'Centers',
                marker: { 
                    size: 12, 
                    symbol: 'star',
                    color: '#000000',
                    line: { color: '#ffffff', width: 1 }
                },
                type: 'scatter'
            });
        }
        
        const layout = {
            title: {
                text: 'Clustering Results',
                font: { size: 20, family: 'Roboto' }
            },
            xaxis: { 
                title: { text: 'ln(Δt)', font: { size: 14 } },
                showgrid: true,
                gridcolor: '#e0e0e0'
            },
            yaxis: { 
                title: { text: 'ln(dΔp/dlnΔt)', font: { size: 14 } },
                showgrid: true,
                gridcolor: '#e0e0e0'
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: -0.2,
                xanchor: 'center',
                x: 0.5
            },
            margin: { t: 50, b: 100, l: 80, r: 50 },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };
        
        console.log("Plotting with traces:", traces);
        Plotly.newPlot(clusterPlotElement, traces, layout, config);
    }

    // Function to update the elbow plot
    function updateElbowPlot(elbowData) {
        console.log("Updating elbow plot with data:", elbowData);
        
        // Clear the plot container
        const elbowPlotElement = document.getElementById('elbowPlot');
        elbowPlotElement.innerHTML = '';
        
        // Check if we have valid elbow data
        if (!elbowData || !elbowData.k_values || !elbowData.k_scores) {
            console.error("No valid elbow data provided");
            elbowPlotContainer.style.display = 'none';
            return;
        }
        
        // Ensure the container is visible
        elbowPlotContainer.style.display = 'block';
        
        const kValues = elbowData.k_values;
        const kScores = elbowData.k_scores;
        const elbowValue = elbowData.elbow_value;
        const elbowScore = elbowData.elbow_score;
        
        // Create the main score line trace
        const scoreLine = {
            x: kValues,
            y: kScores,
            mode: 'lines+markers',
            name: 'Distortion Score',
            line: {
                color: 'blue',
                width: 2
            },
            marker: {
                color: 'blue',
                size: 8,
                symbol: 'square'
            }
        };
        
        const traces = [scoreLine];
        
        // Add vertical line at elbow point if available
        if (elbowValue !== null) {
            // Add vertical dashed line at elbow point
            traces.push({
                x: [elbowValue, elbowValue],
                y: [Math.min(...kScores) * 0.9, Math.max(...kScores) * 1.1],
                mode: 'lines',
                name: `Elbow Point (k=${elbowValue})`,
                line: {
                    color: 'black',
                    width: 2,
                    dash: 'dash'
                }
            });
            
            // Create tangent line for elbow point
            if (elbowValue < kValues.length - 1) {
                // Find the index of the elbow value in kValues
                const elbowIndex = kValues.indexOf(elbowValue);
                
                if (elbowIndex !== -1 && elbowIndex < kValues.length - 1) {
                    const x1 = elbowValue;
                    const y1 = kScores[elbowIndex];
                    const x2 = kValues[elbowIndex + 1];
                    const y2 = kScores[elbowIndex + 1];
                    
                    // Calculate slope of the tangent line
                    const slope = (y2 - y1) / (x2 - x1);
                    
                    // Extrapolate the line backward and forward
                    const extraStartX = Math.max(1, x1 - 1);
                    const extraEndX = Math.min(Math.max(...kValues) + 1, x2 + 2);
                    
                    // Calculate y values using the line equation: y = slope * (x - x1) + y1
                    const extraStartY = slope * (extraStartX - x1) + y1;
                    const extraEndY = slope * (extraEndX - x1) + y1;
                    
                    // Add tangent line
                    traces.push({
                        x: [extraStartX, x1, x2, extraEndX],
                        y: [extraStartY, y1, y2, extraEndY],
                        mode: 'lines',
                        name: 'Tangent Line',
                        line: {
                            color: 'red',
                            width: 2,
                            dash: 'dash'
                        }
                    });
                }
            }
        }
        
        // Create the layout
        const layout = {
            title: {
                text: 'Elbow Method for Optimal k',
                font: { size: 20, family: 'Roboto' }
            },
            xaxis: { 
                title: { text: 'k', font: { size: 14 } },
                tickmode: 'array',
                tickvals: kValues,
                showgrid: true,
                gridcolor: '#e0e0e0'
            },
            yaxis: { 
                title: { text: 'Distortion score (WCSS)', font: { size: 14 } },
                showgrid: true,
                gridcolor: '#e0e0e0',
                tickformat: '.3f'
            },
            showlegend: true,
            legend: {
                x: 1.05,
                y: 1,
                xanchor: 'left',
                font: {
                    family: 'Courier New, monospace',
                    size: 12
                }
            },
            margin: { t: 50, b: 80, l: 80, r: 150 },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff',
            annotations: [
                {
                    x: 1.05,
                    y: 0.98,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Estimator: ${insertLineBreaks(elbowData.estimator, 30)}<br>` +
                          `Locate Elbow: ${elbowData.locate_elbow}<br>` +
                          `Elbow Value: ${elbowData.elbow_value}<br>` +
                          `Elbow Score: ${elbowData.elbow_score?.toFixed(3)}`,
                    showarrow: false,
                    font: {
                        family: 'Courier New, monospace',
                        size: 12
                    },
                    align: 'right',
                    bordercolor: '#c7c7c7',
                    borderwidth: 1,
                    bgcolor: '#ffffff',
                    opacity: 0.8,
                }
            ]
        };
        
        // Configuration for the plot
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };
        
        // Create the plot
        Plotly.newPlot(elbowPlotElement, traces, layout, config);
    }

    // Notification function
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Remove notification after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    // Line break function
    function insertLineBreaks(text, maxLineLength) {
        let result = '';
        let lineLength = 0;
        const words = text.split(' ');
    
        words.forEach((word, index) => {
            if (lineLength + word.length + 1 > maxLineLength) {
                result += '<br>';
                lineLength = 0;
            } else if (index > 0) {
                result += ' ';
                lineLength += 1;
            }
            result += word;
            lineLength += word.length;
        });
    
        return result;
    }

    handleMethodChange({ target: clusteringMethod });
}); 
