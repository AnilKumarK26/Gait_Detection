// Main JavaScript for Gait Detection Application

let currentMode = 'webcam';
let isRunning = true;

// Switch to webcam mode
function switchToWebcam() {
    currentMode = 'webcam';
    
    // Update UI
    document.getElementById('webcamBtn').classList.add('active');
    document.getElementById('uploadBtn').classList.remove('active');
    document.getElementById('uploadSection').style.display = 'none';
    
    // Switch to webcam feed
    fetch('/switch_mode/webcam')
        .then(response => response.json())
        .then(data => {
            console.log('Switched to webcam mode');
            updateVideoFeed();
            showStatus('Webcam mode activated', 'success');
        })
        .catch(error => {
            console.error('Error switching to webcam:', error);
            showStatus('Error switching to webcam', 'error');
        });
}

// Show upload section
function showUpload() {
    document.getElementById('webcamBtn').classList.remove('active');
    document.getElementById('uploadBtn').classList.add('active');
    document.getElementById('uploadSection').style.display = 'block';
}

// Handle file upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }
    
    // Update file name display
    document.getElementById('fileName').textContent = file.name;
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading status
    showStatus('Uploading file...', 'loading');
    
    // Upload file
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentMode = data.mode;
            showStatus(`File uploaded successfully! Mode: ${data.mode}`, 'success');
            updateVideoFeed();
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        showStatus('Error uploading file', 'error');
    });
}

// Update video feed
function updateVideoFeed() {
    const videoFeed = document.getElementById('videoFeed');
    const timestamp = new Date().getTime();
    videoFeed.src = `/video_feed?t=${timestamp}`;
}

// Show status message
function showStatus(message, type) {
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.textContent = message;
    statusDiv.style.display = 'block';
    
    // Style based on type
    if (type === 'success') {
        statusDiv.style.background = 'rgba(34, 197, 94, 0.2)';
        statusDiv.style.color = '#22c55e';
        statusDiv.style.border = '1px solid #22c55e';
    } else if (type === 'error') {
        statusDiv.style.background = 'rgba(239, 68, 68, 0.2)';
        statusDiv.style.color = '#ef4444';
        statusDiv.style.border = '1px solid #ef4444';
    } else if (type === 'loading') {
        statusDiv.style.background = 'rgba(100, 200, 255, 0.2)';
        statusDiv.style.color = '#64C8FF';
        statusDiv.style.border = '1px solid #64C8FF';
    }
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        statusDiv.style.display = 'none';
    }, 5000);
}

// Start detection
function startDetection() {
    const videoFeed = document.getElementById('videoFeed');
    const statusIndicator = document.getElementById('statusIndicator');
    
    videoFeed.style.opacity = '1';
    statusIndicator.classList.add('active');
    statusIndicator.classList.remove('inactive');
    statusIndicator.textContent = '● LIVE';
    
    isRunning = true;
    updateVideoFeed();
}

// Stop detection
function stopDetection() {
    const videoFeed = document.getElementById('videoFeed');
    const statusIndicator = document.getElementById('statusIndicator');
    
    fetch('/stop')
        .then(response => response.json())
        .then(data => {
            videoFeed.style.opacity = '0.5';
            statusIndicator.classList.remove('active');
            statusIndicator.classList.add('inactive');
            statusIndicator.textContent = '● PAUSED';
            isRunning = false;
        })
        .catch(error => {
            console.error('Error stopping detection:', error);
        });
}

// Toggle fullscreen
function toggleFullscreen() {
    const videoWrapper = document.querySelector('.video-wrapper');
    
    if (!document.fullscreenElement) {
        videoWrapper.requestFullscreen().catch(err => {
            console.error('Error attempting to enable fullscreen:', err);
        });
    } else {
        document.exitFullscreen();
    }
}

// Handle fullscreen changes
document.addEventListener('fullscreenchange', () => {
    const videoWrapper = document.querySelector('.video-wrapper');
    if (document.fullscreenElement) {
        videoWrapper.style.height = '100vh';
    } else {
        videoWrapper.style.height = 'auto';
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (event) => {
    switch(event.key) {
        case ' ':
            event.preventDefault();
            if (isRunning) {
                stopDetection();
            } else {
                startDetection();
            }
            break;
        case 'f':
        case 'F':
            toggleFullscreen();
            break;
        case 'w':
        case 'W':
            switchToWebcam();
            break;
    }
});

// Initialize on page load
window.addEventListener('load', () => {
    console.log('Gait Detection Application Loaded');
    switchToWebcam();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, optionally pause detection
        console.log('Page hidden');
    } else {
        // Page is visible, resume if needed
        console.log('Page visible');
        if (isRunning) {
            updateVideoFeed();
        }
    }
});

// Error handling for video feed
document.getElementById('videoFeed').addEventListener('error', () => {
    console.error('Video feed error');
    showStatus('Video feed disconnected. Reconnecting...', 'error');
    
    // Attempt to reconnect after 2 seconds
    setTimeout(() => {
        if (isRunning) {
            updateVideoFeed();
        }
    }, 2000);
});