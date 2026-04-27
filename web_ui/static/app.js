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

  // -------- Pinch drag state --------
  // For each panel we remember an OFFSET from its face-anchored position.
  // Saved positions follow the face automatically and persist across sessions.
  const OFFSET_STORE_KEY = 'stoned.panelOffsets.v1';
  let panelOffsets = loadOffsets();          // { hr: {dx,dy}, expr: {dx,dy} }
  let dragPanelName = null;                  // which panel is currently grabbed
  let dragGrabOffset = { x: 0, y: 0 };       // hand-mid offset within the panel at grab time
  let dragHandPrev = null;                   // 'left'|'right'|null - which hand is grabbing
  let pinchPrev = { left: false, right: false };

  // Grace periods stop a one- or two-frame loss of pinch detection from
  // accidentally releasing the gesture. If the pinch comes back within the
  // window, the gesture resumes seamlessly.
  const ZOOM_RELEASE_GRACE_MS = 400;
  const DRAG_RELEASE_GRACE_MS = 280;
  let zoomReleasePending = null;   // timestamp the LEFT-hand pinch first dropped
  let dragReleasePending = null;   // timestamp the RIGHT-hand pinch first dropped

  // Pinch CONFIRMATION hold: a pinch must be held continuously for this long
  // before any drag/zoom action triggers. This filters out incidental brief
  // pinches that happen during normal hand movement (thumb and index briefly
  // close together by chance).
  const PINCH_HOLD_MS = 2000;
  let leftPinchStart  = null;   // when the current LEFT raw pinch streak started
  let rightPinchStart = null;   // when the current RIGHT raw pinch streak started

  function loadOffsets() {
    try {
      const raw = localStorage.getItem(OFFSET_STORE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (_) {
      return {};
    }
  }
  function saveOffsets() {
    try {
      localStorage.setItem(OFFSET_STORE_KEY, JSON.stringify(panelOffsets));
    } catch (_) {}
  }

  // -------- Per-panel left-hand vertical zoom --------
  // Pinch with the LEFT hand OVER a specific panel. While the pinch is held,
  // moving the hand UP grows that panel, DOWN shrinks it. Each panel has its
  // own saved size in localStorage.
  const SCALE_STORE_KEY = 'stoned.panelScales.v2';
  const MIN_SCALE = 0.65, MAX_SCALE = 2.0;
  // Sensitivity: a hand sweep of 50% of the frame height roughly doubles size.
  const ZOOM_VERT_SENSITIVITY = 1.6;
  let panelScales = loadScales();    // { hr: 1.0, expr: 1.0 }
  const zoom = {
    active: false,
    target: null,                    // 'hr' | 'expr' - which panel is being zoomed
    initialY: 0,
    initialScale: 1,
  };
  function loadScales() {
    try {
      const raw = localStorage.getItem(SCALE_STORE_KEY);
      const obj = raw ? JSON.parse(raw) : {};
      return {
        hr:   clamp(parseFloat(obj.hr)   || 1, MIN_SCALE, MAX_SCALE),
        expr: clamp(parseFloat(obj.expr) || 1, MIN_SCALE, MAX_SCALE),
      };
    } catch (_) {
      return { hr: 1, expr: 1 };
    }
  }
  function saveScales() {
    try { localStorage.setItem(SCALE_STORE_KEY, JSON.stringify(panelScales)); } catch (_) {}
  }
  function applyPanelScale(name) {
    const el = (name === 'hr') ? panelHR : panelExpr;
    el.style.setProperty('--panel-scale', (panelScales[name] || 1).toFixed(3));
  }
  applyPanelScale('hr');
  applyPanelScale('expr');

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

    // Default anchor positions (right of face / left of face)
    let hrLeft = r.x + r.w + gap;
    let hrTop  = r.y + (r.h - hrRect.height) / 2;
    let exLeft = r.x - exprRect.width - gap;
    let exTop  = r.y + (r.h - exprRect.height) / 2;

    // Apply user-saved offsets so panels stay where the user dropped them,
    // measured relative to the face anchor.
    const off = panelOffsets || {};
    if (off.hr)   { hrLeft += off.hr.dx;   hrTop += off.hr.dy; }
    if (off.expr) { exLeft += off.expr.dx; exTop += off.expr.dy; }

    // Don't move the panel that's actively being dragged - the drag handler
    // is authoritative until release.
    if (dragPanelName !== 'hr') {
      panelHR.style.left = `${clamp(hrLeft, 8, window.innerWidth  - hrRect.width  - 8)}px`;
      panelHR.style.top  = `${clamp(hrTop,  8, window.innerHeight - hrRect.height - 8)}px`;
    }
    if (dragPanelName !== 'expr') {
      panelExpr.style.left = `${clamp(exLeft, 8, window.innerWidth  - exprRect.width  - 8)}px`;
      panelExpr.style.top  = `${clamp(exTop,  8, window.innerHeight - exprRect.height - 8)}px`;
    }

    // After both panels have been positioned, push them apart if they collide.
    resolveOverlaps();
  }

  // ---- Overlap resolution ----
  // Reads each panel's TARGET position (from inline style), not its currently
  // animating rectangle. This avoids feedback loops where mid-transition rects
  // produced different "best directions" each frame.
  function targetRect(el) {
    const left = parseFloat(el.style.left) || 0;
    const top  = parseFloat(el.style.top)  || 0;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    return { left, top, right: left + w, bottom: top + h, width: w, height: h };
  }

  function _overlap(a, b) {
    return a.left < b.right && b.left < a.right &&
           a.top  < b.bottom && b.top  < a.bottom;
  }

  function resolveOverlaps() {
    const a = panelHR, b = panelExpr;
    if (a.classList.contains('hidden') || b.classList.contains('hidden')) return;

    const rA = targetRect(a);
    const rB = targetRect(b);
    if (!_overlap(rA, rB)) return;

    // Pick the panel to move. Never move the one being actively dragged.
    // Otherwise, prefer to move the Expressions panel since HR is the
    // primary data block. Result: push direction is the same every frame
    // for a given input, so no oscillation.
    const movable    = (dragPanelName === 'hr')   ? b
                     : (dragPanelName === 'expr') ? a
                     : b;
    const stationary = (movable === a) ? b : a;
    const sRect      = (stationary === a) ? rA : rB;
    const mRect      = (movable === a) ? rA : rB;
    const gap = 12;
    const margin = 8;

    // Try each direction in fixed priority: down, up, right, left.
    // First one that fits wins. Deterministic.
    const tryMoves = [
      { axis: 'top',  value: sRect.bottom + gap,                check: v => v + mRect.height <= window.innerHeight - margin },
      { axis: 'top',  value: sRect.top - mRect.height - gap,    check: v => v >= margin },
      { axis: 'left', value: sRect.right + gap,                 check: v => v + mRect.width <= window.innerWidth - margin },
      { axis: 'left', value: sRect.left - mRect.width - gap,    check: v => v >= margin },
    ];

    let placed = false;
    for (const m of tryMoves) {
      if (m.check(m.value)) {
        if (m.axis === 'top') {
          movable.style.top = `${m.value}px`;
        } else {
          movable.style.left = `${m.value}px`;
        }
        placed = true;
        break;
      }
    }
    if (!placed) return;   // can't fit anywhere - leave as-is

    // CRITICAL: persist the new position as a saved offset so the next
    // face-anchor pass uses it directly (no second push). This is what
    // breaks the loop you were seeing.
    if (lastBbox) {
      const name = (movable === panelHR) ? 'hr' : 'expr';
      const r = frameToScreen(lastBbox.x, lastBbox.y, lastBbox.w, lastBbox.h, lastFrameW, lastFrameH);
      const baseLeft = (name === 'hr')
        ? r.x + r.w + 20
        : r.x - movable.offsetWidth - 20;
      const baseTop = r.y + (r.h - movable.offsetHeight) / 2;
      const cur = targetRect(movable);
      panelOffsets[name] = {
        dx: cur.left - baseLeft,
        dy: cur.top  - baseTop,
      };
      saveOffsets();
    }
  }

  // ---- Pinch-drag ----
  function pinchToScreen(mid) {
    // mid is normalised 0-1 in frame coords. Convert through the same
    // object-fit: cover mapping the camera image uses.
    const r = frameToScreen(mid.x, mid.y, 0, 0, lastFrameW, lastFrameH);
    return { x: r.x, y: r.y };
  }

  function panelAtPoint(px, py) {
    // Returns the panel name that contains (px, py), or null.
    // We test the on-screen rect of each draggable panel.
    for (const [name, el] of [['hr', panelHR], ['expr', panelExpr]]) {
      if (el.classList.contains('hidden')) continue;
      const r = el.getBoundingClientRect();
      if (px >= r.left && px <= r.right && py >= r.top && py <= r.bottom) {
        return { name, el, rect: r };
      }
    }
    return null;
  }

  function applyHands(hands) {
    if (!hands) return;
    const left  = hands.left;     // -> ZOOM hand
    const right = hands.right;    // -> DRAG hand
    const now = performance.now();

    // ---- Raw pinch state from server ----
    const rawLeft  = !!(left  && left.pinch  && left.mid);
    const rawRight = !!(right && right.pinch && right.mid);

    // ---- Track when each raw pinch streak started ----
    if (rawLeft  && leftPinchStart  === null) leftPinchStart  = now;
    else if (!rawLeft)  leftPinchStart = null;
    if (rawRight && rightPinchStart === null) rightPinchStart = now;
    else if (!rawRight) rightPinchStart = null;

    // ---- "Confirmed" pinch: held long enough OR already in an active gesture ----
    // This is the key filter that ignores incidental brief pinches caused by
    // normal hand movement.
    const leftPinching = rawLeft && (
      (leftPinchStart !== null && now - leftPinchStart >= PINCH_HOLD_MS) ||
      zoom.active                                        // already zooming
    );
    const rightPinching = rawRight && (
      (rightPinchStart !== null && now - rightPinchStart >= PINCH_HOLD_MS) ||
      (dragPanelName !== null && dragHandPrev === 'right')  // already dragging
    );

    // ============ LEFT HAND -> per-panel ZOOM ============
    if (zoom.active) {
      if (leftPinching) {
        zoomReleasePending = null;
        const dy = zoom.initialY - left.mid.y;
        const newScale = clamp(
          zoom.initialScale + dy * ZOOM_VERT_SENSITIVITY,
          MIN_SCALE, MAX_SCALE
        );
        panelScales[zoom.target] = newScale;
        applyPanelScale(zoom.target);
      } else {
        // Possibly releasing - grace period
        if (zoomReleasePending === null) zoomReleasePending = now;
        if (now - zoomReleasePending >= ZOOM_RELEASE_GRACE_MS) {
          const t = zoom.target;
          if (t) {
            const el = (t === 'hr') ? panelHR : panelExpr;
            el.classList.remove('zooming');
          }
          zoom.active = false;
          zoom.target = null;
          zoomReleasePending = null;
          saveScales();
        }
      }
    } else if (leftPinching) {
      // Start zoom IF the left pinch is hovering over a panel
      const screen = pinchToScreen(left.mid);
      const hit = panelAtPoint(screen.x, screen.y);
      if (hit) {
        zoom.active = true;
        zoom.target = hit.name;
        zoom.initialY = left.mid.y;
        zoom.initialScale = panelScales[hit.name] || 1;
        zoomReleasePending = null;
        hit.el.classList.add('zooming');
      }
    }

    // ============ RIGHT HAND -> DRAG ============
    const wasPinching = pinchPrev.right;

    if (rightPinching && wasPinching && dragHandPrev === 'right' && right.mid) {
      // Continuing a drag
      dragReleasePending = null;
      moveDrag(pinchToScreen(right.mid));
    } else if (rightPinching && !wasPinching && right && right.mid) {
      // Confirmed pinch newly started
      if (dragPanelName !== null && dragHandPrev === 'right') {
        dragReleasePending = null;
        moveDrag(pinchToScreen(right.mid));
      } else if (dragPanelName === null) {
        const screen = pinchToScreen(right.mid);
        const hit = panelAtPoint(screen.x, screen.y);
        if (hit) {
          startDrag(hit.name, hit.el, hit.rect, screen, 'right');
          dragReleasePending = null;
        }
      }
    } else if (!rightPinching && wasPinching && dragHandPrev === 'right') {
      if (dragReleasePending === null) dragReleasePending = now;
    }

    pinchPrev.right = rightPinching;
    pinchPrev.left  = leftPinching;

    // Confirm drag release if grace expired
    if (dragPanelName && dragReleasePending !== null &&
        (now - dragReleasePending >= DRAG_RELEASE_GRACE_MS)) {
      endDrag();
      dragReleasePending = null;
    }
  }

  function startDrag(name, el, rect, screen, side) {
    dragPanelName = name;
    dragHandPrev = side;
    // Where inside the panel the user "grabbed" - so the panel doesn't snap
    // its top-left corner to the hand on grab.
    dragGrabOffset = {
      x: screen.x - rect.left,
      y: screen.y - rect.top,
    };
    // Disable transition during drag so the panel tracks the hand snappily
    el.style.transition = 'none';
    el.classList.add('dragging');
  }

  function moveDrag(screen) {
    if (!dragPanelName) return;
    const el = dragPanelName === 'hr' ? panelHR : panelExpr;
    const newLeft = clamp(screen.x - dragGrabOffset.x, 8, window.innerWidth  - el.offsetWidth  - 8);
    const newTop  = clamp(screen.y - dragGrabOffset.y, 8, window.innerHeight - el.offsetHeight - 8);
    el.style.left = `${newLeft}px`;
    el.style.top  = `${newTop}px`;
  }

  function endDrag() {
    if (!dragPanelName) {
      dragHandPrev = null;
      return;
    }
    const name = dragPanelName;
    const el = name === 'hr' ? panelHR : panelExpr;

    // Compute offset relative to current face anchor and persist it.
    if (lastBbox) {
      const r = frameToScreen(lastBbox.x, lastBbox.y, lastBbox.w, lastBbox.h, lastFrameW, lastFrameH);
      const gap = 20;
      const elRect = el.getBoundingClientRect();
      const baseLeft = (name === 'hr')
        ? r.x + r.w + gap
        : r.x - elRect.width - gap;
      const baseTop = r.y + (r.h - elRect.height) / 2;
      panelOffsets[name] = {
        dx: elRect.left - baseLeft,
        dy: elRect.top  - baseTop,
      };
      saveOffsets();
    }

    el.style.transition = '';
    el.classList.remove('dragging');
    dragPanelName = null;
    dragHandPrev = null;

    // After dropping, if the dropped panel ended up overlapping the other
    // one, shift the OTHER one out of the way so both stay visible.
    resolveOverlaps();
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

    // Heart rate (number only, no bar)
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

    // HRV (RMSSD): number only
    hrvEl.textContent = (m.hrv_rmssd != null) ? Math.round(m.hrv_rmssd) : '--';

    // Stress (0-1)
    const score = Math.max(0, Math.min(1, m.stress_score || 0));
    stressCatEl.textContent = m.stress_category || '--';
    stressPctEl.textContent = `${Math.round(score * 100)}%`;
    stressBar.style.width = `${score * 100}%`;
    stressBar.style.setProperty('--intensity', score.toFixed(2));

    // Action units - same gradient + intensity-driven glow as the stress bar
    const aus = m.action_units || {};
    EXPR_ORDER.forEach(([code]) => {
      const v = Math.max(0, Math.min(1, aus[code] || 0));
      const els = exprRowEls[code];
      els.fill.style.width = `${v * 100}%`;
      els.fill.style.setProperty('--intensity', v.toFixed(2));
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

    // Hand gestures (pinch-drag)
    applyHands(m.hands);
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
