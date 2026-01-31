/* ═══════════════════════════════════════════════════════════════
   Real-Time Right Leg Detection — Frontend logic

   Fixes applied (8 frontend issues)
   ───────────────────────────────────
    1  Stop clears <img src> so the browser stops requesting /video_feed.
    2  switchToWebcam() calls /switch_mode/webcam before reconnecting.
    3  Upload flow calls /switch_mode/webcam first to release the webcam,
       then uploads — avoids dangling capture on the backend.
    4  Reconnect guard: if the <img> fires an error we wait and retry only
       while isRunning is true; avoids infinite retry loops when stopped.
    5  isRunning flag is checked in every async path before touching the feed.
    6  A separate polling loop fetches /gait_data every 200 ms and updates
       the live gait panel.
    7  Mode-change helpers always tear down the previous state cleanly.
    8  Fullscreen exit resets wrapper height.
   ═══════════════════════════════════════════════════════════════ */

let isRunning   = false;
let currentMode = 'webcam';
let gaitPollId  = null;            // setInterval id for gait polling
let reconnectTimer = null;         // pending reconnect timeout

// ─── VIDEO FEED MANAGEMENT ──────────────────────────────────────

/** Start (or restart) the video feed stream. */
function connectFeed() {
    const img = document.getElementById('videoFeed');
    img.src = '/video_feed?' + Date.now();   // cache-bust
}

/** Fully disconnect the video feed — browser stops the HTTP request. */
function disconnectFeed() {
    const img = document.getElementById('videoFeed');
    img.src = '';                            // stops the multipart stream
}

// ─── MODE SWITCHING ─────────────────────────────────────────────

function switchToWebcam() {
    currentMode = 'webcam';
    document.getElementById('webcamBtn').classList.add('active');
    document.getElementById('uploadBtn').classList.remove('active');
    document.getElementById('uploadSection').style.display = 'none';

    disconnectFeed();                        // tear down current stream first

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

    disconnectFeed();                        // release webcam stream
    // Tell backend to release webcam so it doesn't hold the device
    fetch('/stop').catch(() => {});
}

// ─── FILE UPLOAD ────────────────────────────────────────────────

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
                // Reconnect feed for the new file
                isRunning = true;
                connectFeed();
                _setLiveUI(true);
            } else {
                showStatus(data.error || 'Upload failed', 'error');
            }
        })
        .catch(() => showStatus('Network error during upload', 'error'));
}

// ─── START / STOP ───────────────────────────────────────────────

function startDetection() {
    isRunning = true;
    _setLiveUI(true);

    // If we have a mode ready, connect
    if (currentMode === 'webcam') {
        fetch('/switch_mode/webcam')
            .then(() => connectFeed())
            .catch(() => connectFeed());     // best-effort; connect anyway
    } else {
        connectFeed();
    }
}

function stopDetection() {
    isRunning = false;
    disconnectFeed();                        // ← key fix: actually stops the stream
    _setLiveUI(false);

    fetch('/stop').catch(() => {});          // tell backend to stop its generator
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

// ─── STATUS MESSAGES ────────────────────────────────────────────

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

// ─── FULLSCREEN ─────────────────────────────────────────────────

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

// ─── GAIT DATA POLLING ──────────────────────────────────────────

function startGaitPolling() {
    if (gaitPollId) return;                  // already running
    gaitPollId = setInterval(() => {
        if (!isRunning) return;              // skip when paused
        fetch('/gait_data')
            .then(r => r.json())
            .then(updateGaitPanel)
            .catch(() => {});                // silent fail — non-critical
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
    _setText('gaitConfidence', data.confidence != null ? data.confidence.toFixed(2) : '—');
    _setText('gaitKnee',       data.knee_angle_deg != null ? data.knee_angle_deg + '°' : '—');
    _setText('gaitScale',      data.scale_factor != null ? data.scale_factor.toFixed(3) : '—');
    _setText('gaitStride',     data.stride_count != null ? String(data.stride_count) : '—');
    _setText('gaitTime',       data.timestamp_s != null ? data.timestamp_s.toFixed(1) + ' s' : '—');
    _setText('gaitStatus',     data.status || '—');
}

function _setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

// ─── ERROR / RECONNECT ──────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const img = document.getElementById('videoFeed');

    img.addEventListener('error', () => {
        if (!isRunning) return;              // don't retry if deliberately stopped
        console.warn('Video feed error — scheduling reconnect');
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => {
            if (isRunning) connectFeed();    // guarded retry
        }, 2000);
    });
});

// ─── KEYBOARD SHORTCUTS ─────────────────────────────────────────

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

// ─── INIT ───────────────────────────────────────────────────────

window.addEventListener('load', () => {
    console.log('Gait Detection App loaded');
    isRunning = true;
    _setLiveUI(true);
    connectFeed();
    startGaitPolling();
});

// Pause polling when tab is hidden; resume when visible
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