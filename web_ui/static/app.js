// Stoned web UI - WebSocket client + DOM updates.
// The server pushes alternating messages:
//   binary -> latest JPEG frame
//   text   -> JSON metrics

(() => {
  const $ = (id) => document.getElementById(id);

  const camImg          = $('cam');
  const placeholderEl   = $('placeholder');
  const statusPill      = $('status');
  const calBanner       = $('calibration');
  const calFill         = calBanner.querySelector('.cal-fill');
  const panelHR         = $('panel-hr');
  const panelExpr       = $('panel-expr');
  const exprList        = $('expr-list');
  const hrBpmEl         = $('hr-bpm');
  const hrvEl           = $('hrv');
  const stressCatEl     = $('stress-cat');
  const stressPctEl     = $('stress-pct');
  const stressBar       = $('stress-bar');
  const hrPath          = $('hr-path');
  const hrSvg           = $('hr-svg');
  const hrGraphBpm      = $('hr-graph-bpm');

  const EXPR_ORDER = [
    ['AU12', 'Smile'],
    ['AU4',  'Frown'],
    ['AU1',  'Brow Raise'],
    ['AU5',  'Eye Wide'],
    ['AU6',  'Cheek Raise'],
    ['AU15', 'Lip Down'],
    ['AU23', 'Lip Tight'],
    ['AU26', 'Jaw Open'],
    ['AU45', 'Blink'],
    ['AU2',  'Outer Brow'],
  ];

  // Build expression rows once
  const exprRowEls = {};
  EXPR_ORDER.forEach(([code, label]) => {
    const row = document.createElement('div');
    row.className = 'expr-row';
    row.innerHTML = `
      <span class="expr-name">${label}</span>
      <span class="expr-bar"><span class="expr-bar-fill"></span></span>
      <span class="expr-pct">0%</span>
    `;
    exprList.appendChild(row);
    exprRowEls[code] = {
      fill: row.querySelector('.expr-bar-fill'),
      pct:  row.querySelector('.expr-pct'),
    };
  });

  // Persisted state
  const HR_MAX_SAMPLES = 180;
  const hrHistory = [];
  let lastFrameW = 1280;
  let lastFrameH = 720;
  let lastBbox = null;
  let lastFrameUrl = null;
  let ws = null;
  let reconnectTimer = null;

  function setStatus(text, cls) {
    statusPill.textContent = text;
    statusPill.classList.remove('connected', 'disconnected');
    if (cls) statusPill.classList.add(cls);
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  // The frame is displayed with object-fit: cover. Map frame-normalised coords
  // to the actual on-screen pixel position so panels can sit beside the face.
  function frameToScreen(bx, by, bw, bh, frameW, frameH) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const frameAR  = frameW / frameH;
    const screenAR = vw / vh;
    let dispW, dispH, offX, offY;
    if (screenAR > frameAR) {
      dispW = vw; dispH = vw / frameAR;
      offX = 0;   offY  = (vh - dispH) / 2;
    } else {
      dispH = vh; dispW = vh * frameAR;
      offX = (vw - dispW) / 2; offY = 0;
    }
    return {
      x: offX + bx * dispW,
      y: offY + by * dispH,
      w: bw * dispW,
      h: bh * dispH,
    };
  }

  function anchorToFace(bbox) {
    const r = frameToScreen(bbox.x, bbox.y, bbox.w, bbox.h, lastFrameW, lastFrameH);
    const gap = 20;

    const hrRect   = panelHR.getBoundingClientRect();
    const exprRect = panelExpr.getBoundingClientRect();

    const hrLeft = clamp(r.x + r.w + gap, 8, window.innerWidth  - hrRect.width  - 8);
    const hrTop  = clamp(r.y + (r.h - hrRect.height) / 2, 8, window.innerHeight - hrRect.height - 8);
    panelHR.style.left = `${hrLeft}px`;
    panelHR.style.top  = `${hrTop}px`;

    const exLeft = clamp(r.x - exprRect.width - gap, 8, window.innerWidth  - exprRect.width  - 8);
    const exTop  = clamp(r.y + (r.h - exprRect.height) / 2, 8, window.innerHeight - exprRect.height - 8);
    panelExpr.style.left = `${exLeft}px`;
    panelExpr.style.top  = `${exTop}px`;
  }

  function applyMetrics(m) {
    if (m.frame_w && m.frame_h) {
      lastFrameW = m.frame_w;
      lastFrameH = m.frame_h;
    }
    if (m.face_bbox) lastBbox = m.face_bbox;

    // Calibration banner
    if (m.is_calibrating) {
      calBanner.classList.remove('hidden');
      calFill.style.width = `${m.calibration_pct || 0}%`;
    } else {
      calBanner.classList.add('hidden');
    }

    // Heart rate
    if (m.hr_bpm > 0) {
      hrBpmEl.textContent = m.hr_bpm;
      hrGraphBpm.textContent = `${m.hr_bpm} BPM`;
      hrHistory.push(m.hr_bpm);
      if (hrHistory.length > HR_MAX_SAMPLES) hrHistory.shift();
      drawHrGraph();
    } else {
      hrBpmEl.textContent = '--';
      hrGraphBpm.textContent = '-- BPM';
    }

    // HRV
    hrvEl.textContent = (m.hrv_rmssd != null) ? Math.round(m.hrv_rmssd) : '--';

    // Stress
    const score = Math.max(0, Math.min(1, m.stress_score || 0));
    stressCatEl.textContent = m.stress_category || '--';
    stressPctEl.textContent = `${Math.round(score * 100)}%`;
    stressBar.style.width = `${score * 100}%`;

    // Action units
    const aus = m.action_units || {};
    EXPR_ORDER.forEach(([code]) => {
      const v = Math.max(0, Math.min(1, aus[code] || 0));
      const els = exprRowEls[code];
      els.fill.style.width = `${v * 100}%`;
      els.fill.classList.toggle('hot', v > 0.5);
      els.pct.textContent = `${Math.round(v * 100)}%`;
    });

    // Anchor panels to the face bbox
    if (m.face_detected && m.face_bbox) {
      anchorToFace(m.face_bbox);
      panelHR.classList.remove('hidden');
      panelExpr.classList.remove('hidden');
    } else {
      panelHR.classList.add('hidden');
      panelExpr.classList.add('hidden');
    }
  }

  function drawHrGraph() {
    if (hrHistory.length < 2) return;
    const w = hrSvg.clientWidth || 1;
    const h = hrSvg.clientHeight || 1;
    hrSvg.setAttribute('viewBox', `0 0 ${w} ${h}`);

    const min = Math.min(...hrHistory);
    const max = Math.max(...hrHistory);
    const span = Math.max(20, max - min + 10);
    const lo = Math.max(30, (min + max) / 2 - span / 2);
    const hi = lo + span;

    let d = '';
    hrHistory.forEach((v, i) => {
      const x = (i / (hrHistory.length - 1)) * w;
      const y = h - ((v - lo) / (hi - lo)) * h;
      d += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
    });
    hrPath.setAttribute('d', d);
  }

  // Reposition panels on resize / orientation change
  window.addEventListener('resize', () => {
    if (lastBbox) anchorToFace(lastBbox);
  });

  // ---- WebSocket connection ----
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws`;
    setStatus('Connecting', '');
    ws = new WebSocket(url);
    ws.binaryType = 'blob';

    ws.onopen = () => {
      setStatus('Live', 'connected');
      // Keep-alive
      const keep = setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          clearInterval(keep);
          return;
        }
        try { ws.send('ping'); } catch (_) {}
      }, 15000);
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        try { applyMetrics(JSON.parse(ev.data)); } catch (e) {}
      } else {
        const url = URL.createObjectURL(ev.data);
        camImg.onload = checkFrameContent;
        camImg.src = url;
        if (lastFrameUrl) URL.revokeObjectURL(lastFrameUrl);
        lastFrameUrl = url;
      }
    };

    ws.onclose = () => {
      setStatus('Reconnecting', 'disconnected');
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 1500);
    };
    ws.onerror = () => { try { ws.close(); } catch (_) {} };
  }

  // Sample a few pixels from the rendered frame; if the frame is essentially
  // a single colour (e.g. DroidCam's green disconnected screen) keep the
  // placeholder visible. Otherwise hide it.
  const _sampleCanvas = document.createElement('canvas');
  _sampleCanvas.width  = 24;
  _sampleCanvas.height = 24;
  const _sampleCtx = _sampleCanvas.getContext('2d', { willReadFrequently: true });
  let _sampleSkip = 0;
  function checkFrameContent() {
    // throttle - once every 6 frames is plenty
    if (_sampleSkip++ % 6 !== 0) return;
    if (!camImg.naturalWidth) return;
    try {
      _sampleCtx.drawImage(camImg, 0, 0, 24, 24);
      const data = _sampleCtx.getImageData(0, 0, 24, 24).data;
      let rSum = 0, gSum = 0, bSum = 0;
      let rMin = 255, rMax = 0, gMin = 255, gMax = 0, bMin = 255, bMax = 0;
      const n = data.length / 4;
      for (let i = 0; i < data.length; i += 4) {
        rSum += data[i]; gSum += data[i + 1]; bSum += data[i + 2];
        rMin = Math.min(rMin, data[i]);     rMax = Math.max(rMax, data[i]);
        gMin = Math.min(gMin, data[i + 1]); gMax = Math.max(gMax, data[i + 1]);
        bMin = Math.min(bMin, data[i + 2]); bMax = Math.max(bMax, data[i + 2]);
      }
      const rAvg = rSum / n, gAvg = gSum / n, bAvg = bSum / n;
      const range = Math.max(rMax - rMin, gMax - gMin, bMax - bMin);
      const greenDominant = (gAvg - rAvg > 30) && (gAvg - bAvg > 30) && gAvg > 100;
      const flat = range < 12;
      if (flat || greenDominant) {
        placeholderEl.classList.remove('hidden');
      } else {
        placeholderEl.classList.add('hidden');
      }
    } catch (_) {
      placeholderEl.classList.add('hidden');
    }
  }

  connect();
})();
