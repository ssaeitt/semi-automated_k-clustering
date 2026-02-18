document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const searchSidebar = document.querySelector('.search-sidebar');
    const searchToggle = document.querySelector('.search-toggle');
    const closeSidebar = document.querySelector('.close-sidebar');
    const searchTabs = document.querySelectorAll('.search-tab');
    const searchInput = document.querySelector('.search-input'); // Using the class selector
    const searchButton = document.getElementById('searchButton');
    const searchResults = document.querySelector('.search-results');
    const container = document.querySelector('.container');
    const stopButton = document.querySelector('.stop-button');
    
    // Model dropdown elements
    const modelDropdownContainer = document.getElementById('modelDropdownContainer');
    const modelDropdownButton = document.getElementById('modelDropdownButton');
    const modelDropdownMenu = document.getElementById('modelDropdownMenu');
    const selectedModelText = document.getElementById('selectedModel');
    const modelOptions = document.querySelectorAll('.model-option');

    // State
    let currentTab = 'ai'; // Default to AI
    let selectedModel = 'deepseek/deepseek-r1:free';
    let isSearching = false;
    let currentController = null; 

    // Event Listeners
    searchToggle.addEventListener('click', toggleSidebar);
    closeSidebar.addEventListener('click', toggleSidebar);
    searchTabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
    
    searchButton.addEventListener('click', performSearch);
    stopButton.addEventListener('click', stopSearch);
    
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !isSearching) {
            performSearch();
        }
    });
    
    modelDropdownButton.addEventListener('click', toggleModelDropdown);
    
    modelOptions.forEach(option => {
        option.addEventListener('click', function() {
            selectModel(this.dataset.model, this.textContent);
            toggleModelDropdown();
        });
    });

    // Functions
    function toggleSidebar() {
        searchSidebar.classList.toggle('active');
        container.classList.toggle('sidebar-active');
        if (searchSidebar.classList.contains('active')) {
            searchInput.focus();
        }
        
        setTimeout(() => {
            if (window.Plotly) {
                const clusterPlot = document.getElementById('clusterPlot');
                const elbowPlot = document.getElementById('elbowPlot');
                if (clusterPlot) Plotly.Plots.resize(clusterPlot);
                if (elbowPlot) Plotly.Plots.resize(elbowPlot);
            }
        }, 300);
    }

    function switchTab(tab) {
        currentTab = tab;
        searchTabs.forEach(t => t.classList.remove('active'));
        const activeTabEl = document.querySelector(`[data-tab="${tab}"]`);
        if (activeTabEl) activeTabEl.classList.add('active');
        
        searchResults.innerHTML = ''; 
        modelDropdownContainer.style.display = tab === 'ai' ? 'block' : 'none';
        
        if (tab === 'web') {
            searchInput.placeholder = "Web Search is disabled...";
            searchInput.disabled = true; // Visual cue that we are using AI only
        } else {
            searchInput.placeholder = "Ask AI about Well Test Analysis...";
            searchInput.disabled = false;
        }
    }
    
    function toggleModelDropdown() {
        modelDropdownMenu.classList.toggle('show');
    }
    
    function selectModel(modelId, modelName) {
        selectedModel = modelId;
        selectedModelText.textContent = modelName;
        modelOptions.forEach(option => {
            option.dataset.model === modelId ? option.classList.add('selected') : option.classList.remove('selected');
        });
    }

    function stopSearch() {
        if (currentController) {
            currentController.abort();
            currentController = null;
        }
        isSearching = false;
        stopButton.style.display = 'none';
        searchButton.innerHTML = '<i class="fas fa-search"></i>';
        searchButton.disabled = false;
    }

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query || isSearching) return;

        isSearching = true;
        stopButton.style.display = 'block';
        searchButton.innerHTML = '<div class="loading-spinner"></div>';
        searchButton.disabled = true;
        
        try {
            currentController = new AbortController();
            const signal = currentController.signal;
            
            if (currentTab === 'ai') {
                initializeAIResultContainer();
                
                const response = await fetch('/ai_search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        query: query,
                        model: selectedModel,
                        stream: true 
                    }),
                    signal
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'AI Search Error');
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    let lines = buffer.split('\n');
                    buffer = lines.pop(); 
                    
                    for (const line of lines) {
                        if (line.trim()) {
                            try {
                                const data = JSON.parse(line);
                                processStreamedData(data);
                            } catch (e) { console.error("JSON parse error", e); }
                        }
                    }
                }
                finalizeAIResult();
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                searchResults.innerHTML = '<div class="search-result-item">Search cancelled</div>';
            } else {
                searchResults.innerHTML = `<div class="search-result-item error">Error: ${error.message}</div>`;
            }
        } finally {
            searchButton.innerHTML = '<i class="fas fa-search"></i>';
            searchButton.disabled = false;
            isSearching = false;
            stopButton.style.display = 'none';
            currentController = null;
        }
    }

    function initializeAIResultContainer() {
        searchResults.innerHTML = `
            <div class="search-result-item ai-result">
                <div class="model-info"><i class="fas fa-robot"></i> ${selectedModelText.textContent}</div>
                <div class="thinking-section">
                    <div class="thinking-header"><i class="fas fa-brain"></i> Thinking Process</div>
                    <div class="thinking-content"></div>
                </div>
                <div class="answer-section" style="display: none;">
                    <div class="answer-header"><i class="fas fa-comment-dots"></i> Answer</div>
                    <div class="answer-content"></div>
                </div>
            </div>`;
    }

    function processStreamedData(data) {
        const thinkingContent = document.querySelector('.thinking-content');
        const answerSection = document.querySelector('.answer-section');
        const answerContent = document.querySelector('.answer-content');
        
        if (data.token_type === 'thinking' && data.token) {
            thinkingContent.innerText += data.token; // Using innerText for streaming safety
            thinkingContent.scrollTop = thinkingContent.scrollHeight;
        } else if (data.token_type === 'answer' && data.token) {
            answerSection.style.display = 'block';
            answerContent.innerText += data.token;
            answerContent.scrollTop = answerContent.scrollHeight;
        }
    }

    function finalizeAIResult() {
        // Final UI cleanup if needed
    }

    // Initialize the UI on AI Tab
    switchTab('ai');
});
