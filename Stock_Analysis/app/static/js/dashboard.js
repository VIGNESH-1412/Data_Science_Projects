// Dashboard JS - Handles simulations, sliders, model predictions, history tracking

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const companySelect = document.getElementById('company-select');
    const simulationForm = document.getElementById('simulation-form');
    const forecastButton = document.getElementById('forecast-btn');
    
    // Sliders & Inputs
    const fields = ['open', 'high', 'low', 'adj_close', 'volume'];
    const elements = {};
    fields.forEach(f => {
        elements[f] = {
            input: document.getElementById(`input-${f}`),
            slider: document.getElementById(`slider-${f}`),
            valDisplay: document.getElementById(`val-${f}`)
        };
    });

    // Result Card Displays
    const resultCard = document.getElementById('result-card');
    const emptyResult = document.getElementById('empty-result');
    const resultPrice = document.getElementById('result-price');
    const resultTrend = document.getElementById('result-trend');
    const resultTrendIcon = document.getElementById('result-trend-icon');
    const resultTrendBadge = document.getElementById('result-trend-badge');
    const resultConfidence = document.getElementById('result-confidence');
    const resultTime = document.getElementById('result-time');
    const resultSummary = document.getElementById('result-summary');

    // History elements
    const historyTableBody = document.getElementById('history-table-body');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    // Store latest fetched values globally to calculate slider bounds
    let latestValues = null;

    // Load initial prediction history
    loadHistory();

    // Set up slider-input sync and values display
    fields.forEach(f => {
        const el = elements[f];
        if (el.input && el.slider) {
            // Slider to Input
            el.slider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                el.input.value = f === 'volume' ? Math.round(val) : val.toFixed(2);
                if (el.valDisplay) el.valDisplay.textContent = formatDisplayVal(val, f);
                validateInputs();
            });

            // Input to Slider
            el.input.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value) || 0;
                el.slider.value = val;
                if (el.valDisplay) el.valDisplay.textContent = formatDisplayVal(val, f);
                validateInputs();
            });
        }
    });

    // Handle stock selection change
    if (companySelect) {
        companySelect.addEventListener('change', async (e) => {
            const company = e.target.value;
            if (!company) {
                resetForm();
                return;
            }
            
            // Clear previous simulation results card
            if (emptyResult) emptyResult.classList.remove('hidden');
            if (resultCard) resultCard.classList.add('hidden');
            
            // Show loading indicators
            forecastButton.disabled = true;
            forecastButton.innerHTML = `<div class="loader-spinner !w-5 !h-5 mr-2"></div> Synchronizing Data...`;
            
            try {
                const response = await fetch(`/api/stocks/${company}`);
                const data = await response.json();
                
                if (response.ok && data.latest) {
                    latestValues = data.latest;
                    populateInputs(latestValues);
                    forecastButton.disabled = false;
                    forecastButton.innerHTML = `<i class="fas fa-microchip mr-2 text-cyan-400"></i> Run Forecast Engine`;
                    
                    // Show a toast message
                    showToast(`Synced latest trading data for ${company}`, 'success');
                } else {
                    showToast(data.error || 'Failed to fetch stock parameters.', 'error');
                }
            } catch (err) {
                console.error(err);
                showToast('Failed to connect to stock database API.', 'error');
            }
        });
    }

    // Populate input fields and dynamically size slider bounds around stock price
    function populateInputs(latest) {
        fields.forEach(f => {
            const el = elements[f];
            const val = f === 'adj_close' ? latest.Adj_Close : latest[capitalizeFirst(f)];
            
            // Set dynamic limits based on latest price (+/- 40% for price, +/- 60% for volume)
            let min, max, step;
            if (f === 'volume') {
                min = Math.max(100, Math.round(val * 0.2));
                max = Math.round(val * 3.0);
                step = 100;
            } else {
                min = Math.max(0.1, val * 0.6);
                max = val * 1.4;
                step = 0.05;
            }
            
            el.slider.min = min.toFixed(2);
            el.slider.max = max.toFixed(2);
            el.slider.step = step;
            el.slider.value = val;
            
            el.input.value = f === 'volume' ? Math.round(val) : val.toFixed(2);
            if (el.valDisplay) el.valDisplay.textContent = formatDisplayVal(val, f);
        });
        
        validateInputs();
    }

    function capitalizeFirst(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }

    function formatDisplayVal(val, field) {
        if (field === 'volume') {
            if (val >= 1e6) return (val / 1e6).toFixed(2) + 'M';
            if (val >= 1e3) return (val / 1e3).toFixed(2) + 'K';
            return Math.round(val).toString();
        }
        return '₹' + val.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function resetForm() {
        fields.forEach(f => {
            const el = elements[f];
            el.input.value = '';
            el.slider.value = el.slider.min || 0;
            if (el.valDisplay) el.valDisplay.textContent = '₹0.00';
        });
        latestValues = null;
        forecastButton.disabled = true;
    }

    // Dynamic input validation
    function validateInputs() {
        let isValid = true;
        
        const open = parseFloat(elements.open.input.value) || 0;
        const high = parseFloat(elements.high.input.value) || 0;
        const low = parseFloat(elements.low.input.value) || 0;
        const adj = parseFloat(elements.adj_close.input.value) || 0;
        const vol = parseFloat(elements.volume.input.value) || 0;

        // Visual validation feedback
        if (high < low || high < open || high < adj) {
            elements.high.input.classList.add('border-red-500', 'focus:ring-red-500');
            isValid = false;
        } else {
            elements.high.input.classList.remove('border-red-500', 'focus:ring-red-500');
        }

        if (low > high || low > open || low > adj) {
            elements.low.input.classList.add('border-red-500', 'focus:ring-red-500');
            isValid = false;
        } else {
            elements.low.input.classList.remove('border-red-500', 'focus:ring-red-500');
        }

        forecastButton.disabled = !isValid || !companySelect.value;
    }

    // Submit Simulation to Model
    if (simulationForm) {
        simulationForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const company = companySelect.value;
            if (!company) return;

            forecastButton.disabled = true;
            forecastButton.innerHTML = `<div class="loader-spinner !w-5 !h-5 mr-2"></div> Running AI Inference...`;

            const payload = {
                company: company,
                open: parseFloat(elements.open.input.value),
                high: parseFloat(elements.high.input.value),
                low: parseFloat(elements.low.input.value),
                adj_close: parseFloat(elements.adj_close.input.value),
                volume: parseFloat(elements.volume.input.value)
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                if (response.ok && data.success) {
                    displayResult(data);
                    loadHistory(); // Refresh history log
                    showToast('Simulation complete. AI forecast generated!', 'success');
                } else {
                    showToast(data.error || 'Forecasting failed.', 'error');
                }
            } catch (err) {
                console.error(err);
                showToast('Failed to connect to forecast model pipeline.', 'error');
            } finally {
                forecastButton.disabled = false;
                forecastButton.innerHTML = `<i class="fas fa-microchip mr-2 text-purple-500 dark:text-purple-400"></i> Run Forecast Engine`;
            }
        });
    }

    // Render results card
    function displayResult(res) {
        emptyResult.classList.add('hidden');
        resultCard.classList.remove('hidden');

        // Apply a glowing border trigger
        resultCard.className = 'glass-panel p-6 rounded-2xl relative border-l-4 ' + 
            (res.trend === 'BULLISH' ? 'border-emerald-500 shadow-[0_0_30px_rgba(16,185,129,0.15)]' : 'border-rose-500 shadow-[0_0_30px_rgba(244,63,94,0.15)]');

        // Set values
        resultPrice.textContent = '₹' + res.predicted_close.toLocaleString('en-IN', { minimumFractionDigits: 2 });
        resultConfidence.textContent = `${res.confidence.toFixed(2)}%`;
        resultTime.textContent = res.timestamp;

        // Trend indicators
        if (res.trend === 'BULLISH') {
            resultTrend.textContent = 'BULLISH';
            resultTrend.className = 'text-emerald-400 font-extrabold tracking-wide';
            resultTrendIcon.className = 'fas fa-arrow-trend-up text-emerald-400 text-3xl animate-bounce';
            resultTrendBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
            resultTrendBadge.textContent = 'Strong Buy Signal';
        } else {
            resultTrend.textContent = 'BEARISH';
            resultTrend.className = 'text-rose-400 font-extrabold tracking-wide';
            resultTrendIcon.className = 'fas fa-arrow-trend-down text-rose-400 text-3xl animate-bounce';
            resultTrendBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30';
            resultTrendBadge.textContent = 'Sell / Caution Signal';
        }

        // Percentage movement
        const diff = res.predicted_close - res.open;
        const pct = (diff / res.open) * 100;
        const dir = diff >= 0 ? 'above' : 'below';
        
        resultSummary.innerHTML = `
            The forecasting algorithm predicts the close price will settle 
            <span class="font-semibold text-slate-100">${Math.abs(pct).toFixed(2)}%</span> 
            ${dir} the simulated opening trade price of 
            <span class="font-semibold text-slate-100">₹${res.open.toFixed(2)}</span>.
        `;
    }

    // Retrieve SQLite simulation logs
    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();
            
            if (response.ok && data.history) {
                renderHistoryTable(data.history);
            }
        } catch (err) {
            console.error('History load error:', err);
        }
    }

    function renderHistoryTable(items) {
        if (!historyTableBody) return;
        
        if (items.length === 0) {
            historyTableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="px-6 py-10 text-center text-slate-500">
                        <i class="fas fa-database text-3xl mb-3 block opacity-30"></i>
                        No prediction runs logged yet. Execute a simulation above.
                    </td>
                </tr>
            `;
            return;
        }

        historyTableBody.innerHTML = items.map(item => {
            const trendClass = item.trend === 'BULLISH' ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20' : 'text-rose-400 bg-rose-500/10 border-rose-500/20';
            const trendIcon = item.trend === 'BULLISH' ? 'fa-arrow-up' : 'fa-arrow-down';
            
            return `
                <tr class="border-b border-slate-200/50 dark:border-white/[0.04] hover:bg-slate-50 dark:hover:bg-white/[0.02] transition-colors">
                    <td class="px-6 py-4 font-bold text-slate-700 dark:text-slate-200">${item.company_name}</td>
                    <td class="px-6 py-4 text-slate-600 dark:text-slate-400">₹${item.open_price.toFixed(2)}</td>
                    <td class="px-6 py-4 text-slate-600 dark:text-slate-400">₹${item.high_price.toFixed(2)} / ₹${item.low_price.toFixed(2)}</td>
                    <td class="px-6 py-4 font-semibold text-purple-600 dark:text-purple-400">₹${item.predicted_close.toFixed(2)}</td>
                    <td class="px-6 py-4">
                        <span class="px-2.5 py-1 rounded-full text-xs font-bold border ${trendClass} inline-flex items-center gap-1">
                            <i class="fas ${trendIcon}"></i> ${item.trend}
                        </span>
                    </td>
                    <td class="px-6 py-4 text-slate-500 dark:text-slate-400 text-sm">${item.timestamp}</td>
                    <td class="px-6 py-4 text-center">
                        <button onclick="deleteHistoryRow(${item.id})" class="text-slate-400 hover:text-rose-500 dark:text-slate-500 dark:hover:text-rose-400 transition-colors p-1" title="Delete record">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
    }

    // Global handles for actions within row templates
    window.deleteHistoryRow = async (id) => {
        if (!confirm('Are you sure you want to delete this prediction log?')) return;
        
        try {
            const response = await fetch(`/api/history/${id}`, { method: 'DELETE' });
            const data = await response.json();
            
            if (response.ok && data.success) {
                loadHistory();
                showToast('Log record deleted.', 'success');
            } else {
                showToast(data.error || 'Failed to delete record.', 'error');
            }
        } catch (err) {
            console.error(err);
            showToast('Connection error.', 'error');
        }
    };

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', async () => {
            if (!confirm('Warning! This will permanently delete your entire prediction history database. Proceed?')) return;
            
            try {
                const response = await fetch('/api/history/clear', { method: 'POST' });
                const data = await response.json();
                
                if (response.ok && data.success) {
                    loadHistory();
                    showToast('Full prediction history log wiped.', 'success');
                } else {
                    showToast(data.error || 'Failed to clear database.', 'error');
                }
            } catch (err) {
                console.error(err);
                showToast('Connection error.', 'error');
            }
        });
    }
});
