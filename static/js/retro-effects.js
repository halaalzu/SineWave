// Retro Effects JavaScript for FlowState

function selectMode(mode) {
    const container = document.querySelector('.cassette-container');
    container.style.transition = 'transform 0.4s ease, opacity 0.4s ease';
    container.style.transform = 'scale(0.95)';
    container.style.opacity = '0.7';

    setTimeout(() => {
        if (mode === 'freestyle') {
            // Redirect to existing freestyle route
            window.location.href = '/freestyle';
        } else if (mode === 'levels') {
            // Redirect to song selection route  
            window.location.href = '/songs';
        }
        
        container.style.transform = 'scale(1)';
        container.style.opacity = '1';
    }, 400);
}

function goToAnalytics() {
    const container = document.querySelector('.cassette-container');
    container.style.transition = 'transform 0.4s ease';
    container.style.transform = 'rotateY(180deg)';

    setTimeout(() => {
        // Redirect to dashboard route
        window.location.href = '/dashboard';
        
        container.style.transform = 'rotateY(0deg)';
    }, 400);
}

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === '1') selectMode('freestyle');
    if (e.key === '2') selectMode('levels'); 
    if (e.key === 'a' || e.key === 'A') goToAnalytics();
});

// Add sound effect on hover (visual feedback)
document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.menu-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transition = 'all 0.2s ease';
        });
    });
});

// Animate barcode scanning effect
function animateBarcodeScanning() {
    const barcodes = document.querySelectorAll('.barcode span, .barcode-bottom span');
    let delay = 0;
    
    barcodes.forEach(span => {
        setTimeout(() => {
            span.style.opacity = '0.3';
            setTimeout(() => {
                span.style.opacity = '1';
            }, 100);
        }, delay);
        delay += 50;
    });
}

// Run barcode animation every 10 seconds
setInterval(animateBarcodeScanning, 10000);

// Cassette tape reel animation enhancement
function enhanceTapeReelAnimation() {
    const reel = document.querySelector('.tape-reel');
    if (reel) {
        let speed = 8; // seconds per rotation
        
        // Speed up on hover
        reel.addEventListener('mouseenter', () => {
            reel.style.animationDuration = `${speed * 0.3}s`;
        });
        
        reel.addEventListener('mouseleave', () => {
            reel.style.animationDuration = `${speed}s`;
        });
    }
}

// Initialize effects
document.addEventListener('DOMContentLoaded', () => {
    enhanceTapeReelAnimation();
    
    // Add subtle entrance animation
    const container = document.querySelector('.cassette-container');
    if (container) {
        container.style.opacity = '0';
        container.style.transform = 'scale(0.9)';
        
        setTimeout(() => {
            container.style.transition = 'all 0.6s ease';
            container.style.opacity = '1';
            container.style.transform = 'scale(1)';
        }, 200);
    }
});