// Global JavaScript - Aether Stock Intelligence Platform

document.addEventListener('DOMContentLoaded', () => {
    // Theme Manager
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    
    // Check local storage for theme preference, default to dark
    const storedTheme = localStorage.getItem('theme') || 'dark';
    
    if (storedTheme === 'light') {
        document.body.classList.add('light-mode');
        document.documentElement.classList.remove('dark');
        if (themeIcon) {
            themeIcon.className = 'fas fa-moon text-slate-500 hover:text-cyan-500';
        }
    } else {
        document.body.classList.remove('light-mode');
        document.documentElement.classList.add('dark');
        if (themeIcon) {
            themeIcon.className = 'fas fa-sun text-yellow-400 hover:text-yellow-300';
        }
    }
    
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const isLight = document.body.classList.toggle('light-mode');
            if (isLight) {
                document.documentElement.classList.remove('dark');
            } else {
                document.documentElement.classList.add('dark');
            }
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
            
            if (themeIcon) {
                if (isLight) {
                    themeIcon.className = 'fas fa-moon text-slate-500 hover:text-cyan-500';
                } else {
                    themeIcon.className = 'fas fa-sun text-yellow-400 hover:text-yellow-300';
                }
            }
            
            // Dispatch a custom event to re-render charts under the new theme if needed
            window.dispatchEvent(new CustomEvent('theme-changed', { detail: { theme: isLight ? 'light' : 'dark' } }));
        });
    }

    // Mobile Navbar Toggle
    const mobileMenuButton = document.getElementById('mobile-menu-btn');
    const mobileNav = document.getElementById('mobile-nav');

    if (mobileMenuButton && mobileNav) {
        mobileMenuButton.addEventListener('click', () => {
            mobileNav.classList.toggle('hidden');
        });
    }
});

// Utility: Show Alert Banner (Toast)
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) {
        // Create container if not exists
        const div = document.createElement('div');
        div.id = 'toast-container';
        div.className = 'fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none';
        document.body.appendChild(div);
    }
    
    const toast = document.createElement('div');
    toast.className = `glass-panel px-6 py-4 rounded-xl shadow-2xl flex items-center gap-3 transform translate-y-10 opacity-0 transition-all duration-300 pointer-events-auto border-l-4 ${
        type === 'success' ? 'border-emerald-500' : 'border-red-500'
    }`;
    
    const icon = type === 'success' ? 'fa-check-circle text-emerald-400' : 'fa-exclamation-circle text-red-400';
    
    toast.innerHTML = `
        <i class="fas ${icon} text-lg"></i>
        <div class="text-sm font-medium text-slate-100">${message}</div>
        <button class="ml-auto text-slate-400 hover:text-slate-100 focus:outline-none" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.getElementById('toast-container').appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.classList.remove('translate-y-10', 'opacity-0');
    }, 10);
    
    // Auto dismiss after 4 seconds
    setTimeout(() => {
        toast.classList.add('opacity-0', 'translate-y-2');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
