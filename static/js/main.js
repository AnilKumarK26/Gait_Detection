/* Real-Time Multi-Person Right Leg Detection — Frontend */

let isRunning   = false;
let currentMode = 'webcam';
let gaitPollId  = null;
let reconnectTimer = null;

function connectFeed() {
    const img = document.getElementById('videoFeed');
    img.src = '/video_feed?' + Date.now();
}

function disconnectFeed() {
    const img = document.getElementById('videoFeed');
    img.src = '';
}

function switchToWebcam() {
    currentMode = 'webcam';
    document.getElementById('webcamBtn').classList.add('active');
    document.getElementById('uploadBtn').classList.remove('active');
    document.getElementById('uploadSection').style.display = 'none';

    disconnectFeed();

    fetch('/switch_mode/webcam')
        .then(() => {
            if (isRunning) connectFeed();
        })
        .catch(err => {
            console.error('switch_mode/webcam failed:', err);
            showStatus('Error switching to webcam', 'error');
        });
}

function showUpload() {
    currentMode = 'upload';
    document.getElementById('webcamBtn').classList.remove('active');
    document.getElementById('uploadBtn').classList.add('active');
    document.getElementById('uploadSection').style.display = 'block';

    disconnectFeed();
    fetch('/stop').catch(() => {});
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    document.getElementById('fileName').textContent = file.name;
    showStatus('Uploading…', 'loading');

    const fd = new FormData();
    fd.append('file', file);

    fetch('/upload', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                currentMode = data.mode;
                showStatus(`Uploaded — mode: ${data.mode}`, 'success');
                isRunning = true;
                connectFeed();
                _setLiveUI(true);
            } else {
                showStatus(data.error || 'Upload failed', 'error');
            }
        })
        .catch(() => showStatus('Network error during upload', 'error'));
}

function startDetection() {
    isRunning = true;
    _setLiveUI(true);

    if (currentMode === 'webcam') {
        fetch('/switch_mode/webcam')
            .then(() => connectFeed())
            .catch(() => connectFeed());
    } else {
        connectFeed();
    }
}

function stopDetection() {
    isRunning = false;
    disconnectFeed();
    _setLiveUI(false);

    fetch('/stop').catch(() => {});
}

function _setLiveUI(live) {
    const ind = document.getElementById('statusIndicator');
    if (live) {
        ind.classList.add('active');
        ind.classList.remove('inactive');
        ind.textContent = '● LIVE';
    } else {
        ind.classList.remove('active');
        ind.classList.add('inactive');
        ind.textContent = '● PAUSED';
    }
}

function showStatus(message, type) {
    const el = document.getElementById('uploadStatus');
    el.textContent = message;
    el.style.display = 'block';

    const styles = {
        success: { bg: 'rgba(34,197,94,.2)',  color: '#22c55e', border: '1px solid #22c55e' },
        error:   { bg: 'rgba(239,68,68,.2)',  color: '#ef4444', border: '1px solid #ef4444' },
        loading: { bg: 'rgba(100,200,255,.2)', color: '#64C8FF', border: '1px solid #64C8FF' },
    };
    const s = styles[type] || styles.loading;
    el.style.background = s.bg;
    el.style.color      = s.color;
    el.style.border     = s.border;

    clearTimeout(el._hideTimer);
    el._hideTimer = setTimeout(() => { el.style.display = 'none'; }, 5000);
}

function toggleFullscreen() {
    const wrapper = document.querySelector('.video-wrapper');
    if (!document.fullscreenElement) {
        wrapper.requestFullscreen().catch(() => {});
    } else {
        document.exitFullscreen();
    }
}

document.addEventListener('fullscreenchange', () => {
    const wrapper = document.querySelector('.video-wrapper');
    wrapper.style.height = document.fullscreenElement ? '100vh' : 'auto';
});

function startGaitPolling() {
    if (gaitPollId) return;
    gaitPollId = setInterval(() => {
        if (!isRunning) return;
        fetch('/gait_data')
            .then(r => r.json())
            .then(updateGaitPanel)
            .catch(() => {});
    }, 200);
}

function stopGaitPolling() {
    if (gaitPollId) {
        clearInterval(gaitPollId);
        gaitPollId = null;
    }
}

function updateGaitPanel(data) {
    if (!data) return;
    
    // Handle multi-person data
    if (data.num_people !== undefined && data.people && data.people.length > 0) {
        const firstPerson = data.people[0];
        _setText('gaitConfidence', firstPerson.confidence != null ? firstPerson.confidence.toFixed(2) : '—');
        _setText('gaitKnee', firstPerson.knee_angle_deg != null ? firstPerson.knee_angle_deg + '°' : '—');
        _setText('gaitScale', firstPerson.scale_factor != null ? firstPerson.scale_factor.toFixed(3) : '—');
        _setText('gaitStride', firstPerson.stride_count != null ? String(firstPerson.stride_count) : '—');
        _setText('gaitTime', firstPerson.timestamp_s != null ? firstPerson.timestamp_s.toFixed(1) + ' s' : '—');
        _setText('gaitStatus', `${data.num_people} ${data.num_people === 1 ? 'person' : 'people'} detected`);
    } else {
        _setText('gaitConfidence', '—');
        _setText('gaitKnee', '—');
        _setText('gaitScale', '—');
        _setText('gaitStride', '—');
        _setText('gaitTime', '—');
        _setText('gaitStatus', data.status || 'waiting…');
    }
}

function _setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

document.addEventListener('DOMContentLoaded', () => {
    const img = document.getElementById('videoFeed');

    img.addEventListener('error', () => {
        if (!isRunning) return;
        console.warn('Video feed error — scheduling reconnect');
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => {
            if (isRunning) connectFeed();
        }, 2000);
    });
});

document.addEventListener('keydown', e => {
    switch (e.key) {
        case ' ':
            e.preventDefault();
            isRunning ? stopDetection() : startDetection();
            break;
        case 'f': case 'F':
            toggleFullscreen();
            break;
        case 'w': case 'W':
            switchToWebcam();
            break;
    }
});

window.addEventListener('load', () => {
    console.log('Multi-Person Gait Detection App loaded');
    isRunning = true;
    _setLiveUI(true);
    connectFeed();
    startGaitPolling();
});

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopGaitPolling();
    } else {
        if (isRunning) {
            connectFeed();
            startGaitPolling();
        }
    }
});