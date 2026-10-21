document.addEventListener('DOMContentLoaded', function () {
    const searchSidebar = document.querySelector('.search-sidebar');
    const searchToggle = document.querySelector('.search-toggle');
    const closeSidebar = document.querySelector('.close-sidebar');
    const searchInput = document.querySelector('.search-input');
    const searchButton = document.getElementById('searchButton');
    const stopButton = document.getElementById('stopButton');
    const searchResults = document.querySelector('.search-results');
    const modelDropdownButton = document.getElementById('modelDropdownButton');
    const modelDropdownMenu = document.getElementById('modelDropdownMenu');
    const selectedModelText = document.getElementById('selectedModel');
    const modelOptions = document.querySelectorAll('.model-option');

    if (!searchSidebar || !searchInput || !searchButton) return;

    const firstOption = document.querySelector('.model-option');
    let selectedModel = firstOption ? firstOption.dataset.model : '';
    let isSearching = false;
    let currentController = null;

    searchToggle.addEventListener('click', toggleSidebar);
    closeSidebar.addEventListener('click', toggleSidebar);
    searchButton.addEventListener('click', performSearch);
    stopButton.addEventListener('click', stopSearch);
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && searchSidebar.classList.contains('active')) toggleSidebar();
    });
    searchInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !isSearching) performSearch();
    });
    modelDropdownButton.addEventListener('click', () => modelDropdownMenu.classList.toggle('show'));
    modelOptions.forEach(option => option.addEventListener('click', function () {
        selectedModel = this.dataset.model;
        selectedModelText.textContent = this.textContent;
        modelOptions.forEach(o => o.classList.toggle('selected', o === this));
        modelDropdownMenu.classList.remove('show');
    }));

    function toggleSidebar() {
        const open = searchSidebar.classList.toggle('active');
        searchToggle.setAttribute('aria-expanded', String(open));
        if (open) searchInput.focus();
        setTimeout(() => {
            if (window.Plotly) document.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.Plots.resize(el));
        }, 300);
    }

    function stopSearch() {
        if (currentController) currentController.abort();
    }

    function setBusy(busy) {
        isSearching = busy;
        stopButton.style.display = busy ? 'block' : 'none';
        searchButton.disabled = busy;
        searchButton.innerHTML = busy ? '<div class="loading-spinner"></div>' : '<i class="fas fa-search"></i>';
    }

    function showMessage(text, isError) {
        searchResults.innerHTML = '';
        const div = document.createElement('div');
        div.className = 'search-result-item' + (isError ? ' error' : '');
        div.textContent = text;
        searchResults.appendChild(div);
    }

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query || isSearching) return;
        setBusy(true);
        currentController = new AbortController();
        try {
            const response = await fetch('/ai_search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, model: selectedModel }),
                signal: currentController.signal
            });
            if (!response.ok) {
                let msg = `AI search failed (HTTP ${response.status}).`;
                try { msg = (await response.json()).error || msg; } catch (e) { /* not JSON */ }
                throw new Error(msg);
            }
            initializeAIResultContainer();
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let gotAnswer = false;
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                for (const line of lines) {
                    if (!line.trim()) continue;
                    let data;
                    try { data = JSON.parse(line); } catch (e) { continue; }
                    if (data.token_type === 'error') throw new Error(data.token);
                    if (data.token_type === 'answer') gotAnswer = true;
                    processStreamedData(data);
                }
            }
            if (!gotAnswer) {
                document.querySelector('.answer-section').style.display = 'block';
                document.querySelector('.answer-content').textContent = '(The model returned no answer. Try again or pick another model.)';
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                const note = document.createElement('div');
                note.className = 'search-result-item';
                note.textContent = 'Search stopped.';
                searchResults.appendChild(note);
            } else {
                showMessage('Error: ' + error.message, true);
            }
        } finally {
            setBusy(false);
            currentController = null;
        }
    }

    function initializeAIResultContainer() {
        searchResults.innerHTML = `
            <div class="search-result-item ai-result">
                <div class="model-info"><i class="fas fa-robot"></i> <span class="model-name"></span></div>
                <div class="thinking-section" style="display: none;">
                    <div class="thinking-header"><i class="fas fa-brain"></i> Thinking process</div>
                    <div class="thinking-content"></div>
                </div>
                <div class="answer-section" style="display: none;">
                    <div class="answer-header"><i class="fas fa-comment-dots"></i> Answer</div>
                    <div class="answer-content"></div>
                </div>
            </div>`;
        searchResults.querySelector('.model-name').textContent = selectedModelText.textContent;
    }

    function processStreamedData(data) {
        if (!data.token) return;
        if (data.token_type === 'thinking') {
            const sec = document.querySelector('.thinking-section');
            const el = document.querySelector('.thinking-content');
            sec.style.display = 'block';
            el.textContent += data.token;
            el.scrollTop = el.scrollHeight;
        } else if (data.token_type === 'answer') {
            document.querySelector('.answer-section').style.display = 'block';
            document.querySelector('.answer-content').textContent += data.token;
        }
    }
});
