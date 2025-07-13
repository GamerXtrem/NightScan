/**
 * NightScan Main JavaScript
 * Handles file uploads, WebSocket connections, and UI interactions
 */

let socket;
let currentUserId = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    // Get current user ID from data attribute
    const userElement = document.getElementById('currentUserData');
    if (userElement) {
        currentUserId = parseInt(userElement.dataset.userId);
        initializeWebSocket();
    }
    
    // Setup file upload handlers
    setupFileUpload();
    
    // Auto-hide messages after 5 seconds
    setTimeout(() => {
        const messages = document.querySelectorAll('.message');
        messages.forEach(msg => {
            msg.style.animation = 'slideOut 0.5s ease forwards';
            setTimeout(() => msg.remove(), 500);
        });
    }, 5000);
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
});

/**
 * Initialize WebSocket connection
 */
function initializeWebSocket() {
    if (typeof io === 'undefined') {
        console.error('Socket.IO not loaded');
        return;
    }
    
    socket = io();
    
    socket.on('connect', function() {
        updateConnectionStatus(true);
        
        // Authenticate user
        socket.emit('authenticate', {
            userId: currentUserId,
            token: 'user-session-token'
        });
        
        // Subscribe to prediction updates
        socket.emit('subscribe', {
            event_types: ['prediction_complete']
        });
    });

    socket.on('disconnect', function() {
        updateConnectionStatus(false);
    });

    socket.on('notification', function(data) {
        if (data.eventType === 'prediction_complete') {
            updatePredictionStatus(data.data);
            showNotification('Prediction Complete', 
                `${data.data.filename} has been processed`);
        }
    });
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(connected) {
    const indicator = document.getElementById('statusDot');
    const text = document.getElementById('connectionText');
    
    if (indicator && text) {
        if (connected) {
            indicator.style.background = '#4CAF50';
            text.textContent = 'Connected';
        } else {
            indicator.style.background = '#f44336';
            text.textContent = 'Disconnected';
        }
    }
}

/**
 * Update prediction status in the UI
 */
function updatePredictionStatus(data) {
    // Update prediction in the list
    const predictionsList = document.getElementById('predictionsList');
    // In a real implementation, you would update the specific prediction item
    console.log('Prediction updated:', data);
    
    // Optionally reload the predictions list
    // location.reload();
}

/**
 * Show browser notification
 */
function showNotification(title, message) {
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(title, {
            body: message,
            icon: '/static/nightscan-icon.png'
        });
    }
}

/**
 * Setup file upload handlers
 */
function setupFileUpload() {
    const fileInput = document.getElementById('file');
    const uploadArea = document.querySelector('.file-upload-area');
    
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('drop', handleDrop);
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        updateFileInfo(file);
    }
}

/**
 * Handle file drop
 */
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    
    const uploadArea = event.target.closest('.file-upload-area');
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type === 'audio/wav' || file.name.toLowerCase().endsWith('.wav')) {
            document.getElementById('file').files = files;
            updateFileInfo(file);
        } else {
            showMessage('Please select a WAV file', 'error');
        }
    }
}

/**
 * Handle drag over
 */
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    event.target.closest('.file-upload-area').classList.add('dragover');
}

/**
 * Handle drag leave
 */
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    event.target.closest('.file-upload-area').classList.remove('dragover');
}

/**
 * Update file info display
 */
function updateFileInfo(file) {
    const uploadText = document.querySelector('.upload-text');
    if (uploadText) {
        uploadText.innerHTML = `<strong>Selected:</strong> ${file.name}<br><small>${(file.size / (1024*1024)).toFixed(1)} MB</small>`;
    }
}

/**
 * Show message to user
 */
function showMessage(text, type = 'success') {
    const messagesContainer = document.querySelector('.messages') || createMessagesContainer();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = text;
    messagesContainer.appendChild(messageDiv);
    
    setTimeout(() => {
        messageDiv.style.animation = 'slideOut 0.5s ease forwards';
        setTimeout(() => messageDiv.remove(), 500);
    }, 5000);
}

/**
 * Create messages container if it doesn't exist
 */
function createMessagesContainer() {
    const container = document.createElement('div');
    container.className = 'messages';
    document.body.appendChild(container);
    return container;
}

// Export functions for global access if needed
window.NightScan = {
    showMessage,
    updateConnectionStatus,
    handleFileSelect,
    handleDrop,
    handleDragOver,
    handleDragLeave
};