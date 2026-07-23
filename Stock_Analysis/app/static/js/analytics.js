// Analytics JS - Renders premium financial charts using ApexCharts

document.addEventListener('DOMContentLoaded', () => {
    const companySelect = document.getElementById('analytics-company-select');
    const loadingOverlay = document.getElementById('analytics-loading');
    const analyticsContent = document.getElementById('analytics-content');
    
    // Stats Displays
    const statCompanyName = document.getElementById('stat-company-name');
    const statPriceRange = document.getElementById('stat-price-range');
    const statAvgVolume = document.getElementById('stat-avg-volume');
    const statRecentClose = document.getElementById('stat-recent-close');
    const statPriceTrend = document.getElementById('stat-price-trend');

    let priceChart = null;
    let volumeChart = null;
    let cachedData = null; // Store active data set

    // Initial Trigger if pre-selected
    if (companySelect && companySelect.value) {
        loadAnalytics(companySelect.value);
    }

    if (companySelect) {
        companySelect.addEventListener('change', (e) => {
            const company = e.target.value;
            if (company) {
                loadAnalytics(company);
            } else {
                analyticsContent.classList.add('hidden');
            }
        });
    }

    // Capture global theme switches to repaint charts
    window.addEventListener('theme-changed', (e) => {
        if (cachedData) {
            renderCharts(cachedData, e.detail.theme);
        }
    });

    async function loadAnalytics(company) {
        if (!loadingOverlay) return;
        
        loadingOverlay.classList.remove('hidden');
        analyticsContent.classList.add('hidden');

        try {
            const response = await fetch(`/api/stocks/${company}`);
            const data = await response.json();
            
            if (response.ok && data.history && data.history.length > 0) {
                cachedData = data;
                
                // Populate Stats Panel
                populateStats(company, data.history);
                
                // Render Charts
                const currentTheme = localStorage.getItem('theme') || 'dark';
                renderCharts(data, currentTheme);
                
                loadingOverlay.classList.add('hidden');
                analyticsContent.classList.remove('hidden');
            } else {
                showToast(data.error || 'Failed to fetch analytics datasets.', 'error');
                loadingOverlay.classList.add('hidden');
            }
        } catch (err) {
            console.error(err);
            showToast('Connection to financial API timed out.', 'error');
            loadingOverlay.classList.add('hidden');
        }
    }

    function populateStats(company, history) {
        statCompanyName.textContent = company;
        
        const closes = history.map(h => h.Close);
        const volumes = history.map(h => h.Volume);
        
        const maxPrice = Math.max(...closes);
        const minPrice = Math.min(...closes);
        const avgVol = volumes.reduce((a, b) => a + b, 0) / volumes.length;
        const recentClose = closes[closes.length - 1];
        const previousClose = closes[closes.length - 2] || recentClose;
        
        const change = recentClose - previousClose;
        const changePct = (change / previousClose) * 100;

        statPriceRange.textContent = `₹${minPrice.toFixed(2)} - ₹${maxPrice.toFixed(2)}`;
        
        if (avgVol >= 1e6) {
            statAvgVolume.textContent = (avgVol / 1e6).toFixed(2) + 'M shares';
        } else {
            statAvgVolume.textContent = Math.round(avgVol).toLocaleString() + ' shares';
        }
        
        statRecentClose.textContent = `₹${recentClose.toFixed(2)}`;

        // Trend badge styling
        if (change >= 0) {
            statPriceTrend.className = 'px-3 py-1 text-xs font-bold rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1';
            statPriceTrend.innerHTML = `<i class="fas fa-arrow-trend-up"></i> +${changePct.toFixed(2)}%`;
        } else {
            statPriceTrend.className = 'px-3 py-1 text-xs font-bold rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30 flex items-center gap-1';
            statPriceTrend.innerHTML = `<i class="fas fa-arrow-trend-down"></i> ${changePct.toFixed(2)}%`;
        }
    }

    function renderCharts(data, theme) {
        // Destroy old charts to prevent duplicate allocations
        if (priceChart) priceChart.destroy();
        if (volumeChart) volumeChart.destroy();

        const isDark = theme === 'dark';
        const textColor = isDark ? '#9CA3AF' : '#475569';
        const borderColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.06)';

        const priceSeriesData = data.history.map(item => ({
            x: new Date(item.Date),
            y: [item.Open, item.High, item.Low, item.Close]
        }));

        const volumeSeriesData = data.history.map(item => ({
            x: new Date(item.Date),
            y: item.Volume
        }));

        // 1. Candlestick Price Chart Config
        const priceChartOptions = {
            series: [{
                name: 'Price Candlestick',
                data: priceSeriesData
            }],
            chart: {
                type: 'candlestick',
                height: 380,
                id: 'candles',
                toolbar: {
                    show: true,
                    autoSelected: 'pan'
                },
                background: 'transparent',
                foreColor: textColor
            },
            grid: {
                borderColor: borderColor,
                xaxis: { lines: { show: true } }
            },
            theme: {
                mode: theme
            },
            title: {
                text: 'Market Share Price (OHLC)',
                align: 'left',
                style: {
                    fontFamily: 'Outfit, sans-serif',
                    fontSize: '16px',
                    fontWeight: 600
                }
            },
            xaxis: {
                type: 'datetime',
                labels: {
                    datetimeFormatter: {
                        year: 'yyyy',
                        month: 'MMM \'yy',
                        day: 'dd MMM'
                    }
                }
            },
            yaxis: {
                tooltip: {
                    enabled: true
                },
                labels: {
                    formatter: function (val) {
                        return '₹' + val.toFixed(0);
                    }
                }
            },
            plotOptions: {
                candlestick: {
                    colors: {
                        upward: '#10B981',
                        downward: '#F43F5E'
                    },
                    wick: {
                        useFillColor: true
                    }
                }
            }
        };

        // 2. Bar Volume Chart Config
        const volumeChartOptions = {
            series: [{
                name: 'Volume',
                data: volumeSeriesData
            }],
            chart: {
                type: 'bar',
                height: 180,
                background: 'transparent',
                foreColor: textColor,
                toolbar: {
                    show: false
                }
            },
            grid: {
                borderColor: borderColor
            },
            theme: {
                mode: theme
            },
            title: {
                text: 'Trading Volume',
                align: 'left',
                style: {
                    fontFamily: 'Outfit, sans-serif',
                    fontSize: '14px',
                    fontWeight: 600
                }
            },
            colors: ['#A855F7'],
            dataLabels: {
                enabled: false
            },
            xaxis: {
                type: 'datetime'
            },
            yaxis: {
                labels: {
                    formatter: function (val) {
                        if (val >= 1e6) return (val / 1e6).toFixed(1) + 'M';
                        if (val >= 1e3) return (val / 1e3).toFixed(0) + 'K';
                        return val;
                    }
                }
            }
        };

        // Render instances
        priceChart = new ApexCharts(document.querySelector("#price-chart-container"), priceChartOptions);
        priceChart.render();

        volumeChart = new ApexCharts(document.querySelector("#volume-chart-container"), volumeChartOptions);
        volumeChart.render();
    }
});
