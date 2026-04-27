// Stoned web UI - WebSocket client + DOM updates.
// The server pushes alternating messages:
//   binary -> latest JPEG frame
//   text   -> JSON metrics

(() => {
  const $ = (id) => document.getElementById(id);

  const camImg          = $('cam');     // now a <video> element
  const placeholderEl   = $('placeholder');
  const statusPill      = $('status');
  const camSwitchBtn    = $('cam-switch');
  const cardsOverlayEl  = $('cards-overlay');
  const calBanner       = $('calibration');
  const calFill         = calBanner.querySelector('.cal-fill');
  const panelHR         = $('panel-hr');
  const panelExpr       = $('panel-expr');
  const panelGraph      = $('panel-graph');
  const exprList        = $('expr-list');
  const hrBpmEl         = $('hr-bpm');
  const hrvEl           = $('hrv');
  const stressCatEl     = $('stress-cat');
  const stressPctEl     = $('stress-pct');
  const stressBar       = $('stress-bar');
  const hrPath          = $('hr-path');
  const hrSvg           = $('hr-svg');
  const hrGraphBpm      = $('hr-graph-bpm');

  // Poker DOM
  const cardOverlay     = $('card-overlay');
  const panelWin        = $('panel-win');
  const winPctEl        = $('win-pct');
  const winOutsEl       = $('win-outs');
  const winBar          = $('win-bar');
  const panelBoard      = $('panel-board');
  const myHandCardsEl   = $('my-hand-cards');
  const myHandHintEl    = $('my-hand-hint');
  const boardCardsEl    = $('board-cards');
  const boardEyebrowEl  = $('board-eyebrow');
  const boardHintEl     = $('board-hint');
  const lhLoader        = $('lh-loader');
  const rhLoader        = $('rh-loader');
  const panelMyHands    = $('panel-my-hands');
  const myHandsListEl   = $('my-hands-list');
  const panelOppHands   = $('panel-opp-hands');
  const oppHandsListEl  = $('opp-hands-list');

  let currentContext = 'none';   // tracked so app.js can gate gestures (no panel-drag in poker)

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
  let ws = null;
  let reconnectTimer = null;
  let cameraStream = null;     // MediaStream from getUserMedia
  let captureTimer = null;     // setTimeout handle for the capture loop

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

  // The video is rendered with object-fit: cover; if the front camera is in
  // use we ALSO mirror it on the X axis (selfie convention). The server
  // processes the un-mirrored frame, so when mapping face-bbox / hand
  // positions to screen coords we mirror only when the display is mirrored.
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
    let leftFrame = bx;
    if (currentFacing === 'user') {
      // Mirror X to match the mirrored display
      leftFrame = 1 - (bx + bw);
    }
    return {
      x: offX + leftFrame * dispW,
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

  // -------- Poker rendering --------
  const SUIT_GLYPH = { h: '♥', d: '♦', s: '♠', c: '♣' };

  // Stroke-dasharray of the .pinch-loader-fill circle (2π·44, kept in sync
  // with the CSS so the offset math lands on whole-circle = 0% progress).
  const PINCH_LOADER_CIRC = 276.46;
  function pinchLoaderColor(progress) {
    if (progress < 0.5)  return 'var(--accent-green)';
    if (progress < 0.85) return 'var(--accent-yellow)';
    return 'var(--accent-red)';
  }
  function updatePinchLoader(loaderEl, hand, progress, label, frameW, frameH) {
    // Bail if the pinch isn't being counted (grip-rejected, or simply not pinching).
    if (!hand || !hand.mid || progress < 0.02) {
      loaderEl.classList.add('hidden');
      return;
    }
    const pos = frameToScreen(hand.mid.x, hand.mid.y, 0, 0, frameW, frameH);
    loaderEl.style.left = `${pos.x}px`;
    loaderEl.style.top  = `${pos.y}px`;
    const fillEl = loaderEl.querySelector('.pinch-loader-fill');
    const clamped = Math.max(0, Math.min(1, progress));
    fillEl.style.strokeDashoffset = (PINCH_LOADER_CIRC * (1 - clamped)).toFixed(1);
    fillEl.style.stroke = pinchLoaderColor(clamped);
    loaderEl.querySelector('.pinch-loader-pct').textContent   = `${Math.round(clamped * 100)}%`;
    loaderEl.querySelector('.pinch-loader-label').textContent = label;
    loaderEl.classList.remove('hidden');
  }
  function renderMiniCard(treysStr, dim) {
    if (!treysStr || treysStr.length < 2) return '';
    const rank = treysStr[0] === 'T' ? '10' : treysStr[0];
    const suit = treysStr[1].toLowerCase();
    const isRed = (suit === 'h' || suit === 'd');
    const cls = ['mini-card'];
    if (isRed) cls.push('red');
    if (dim)   cls.push('dim');
    return `<div class="${cls.join(' ')}">
      <span class="mini-card-rank">${rank}</span>
      <span class="mini-card-suit">${SUIT_GLYPH[suit] || ''}</span>
    </div>`;
  }

  function renderHandList(el, hands) {
    if (!hands || !hands.length) {
      el.innerHTML = '<div class="hand-row"><span class="hand-row-name muted">--</span><span class="hand-row-pct">--</span></div>';
      updateHandListFade(el);
      return;
    }
    // Top N — keep DOM small even when treys returns many classes
    const top = hands.slice(0, 8);
    const rows = top.map(h => {
      const pct = Math.round((h.prob || 0) * 100);
      const intensity = Math.min(1, h.prob || 0).toFixed(2);
      const cardsHtml = (h.cards || []).slice(0, 5).map(c => renderMiniCard(c, false)).join('');
      return `
        <div class="hand-row">
          <span class="hand-row-name">${h.name || '--'}</span>
          <span class="hand-row-pct">${pct}%</span>
          <span class="hand-row-bar">
            <span class="hand-row-bar-fill" style="width:${pct}%; --intensity:${intensity};"></span>
          </span>
          <div class="card-row" style="grid-column:1/-1;margin:2px 0 0;min-height:0;">${cardsHtml}</div>
        </div>`;
    }).join('');
    el.innerHTML = rows;
    updateHandListFade(el);
  }

  // Toggle the bottom fade hint based on whether the list can scroll further.
  function updateHandListFade(listEl) {
    const wrap = listEl.parentElement;
    if (!wrap || !wrap.classList.contains('hand-panel-wrap')) return;
    const canScroll = listEl.scrollHeight - listEl.clientHeight - listEl.scrollTop > 4;
    wrap.classList.toggle('has-more', canScroll);
  }
  // Update fade hint when the user scrolls so it disappears at the bottom.
  ['my-hands-list', 'opp-hands-list'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('scroll', () => updateHandListFade(el), { passive: true });
  });

  // -------- Hand-gesture scroll --------
  // "Peace sign" (index + middle up, ring + pinky down) on EITHER hand scrolls
  // the hand-rank lists. Top half of the frame -> My Winning Hands; bottom
  // half -> Opponent Winning Hands. Matches the OLD UI's split. We accept
  // either hand because the user is usually holding cards with one of them.
  const SCROLL_GAIN = 1800;       // pixels of scroll per unit of normalised dy
  let lastScrollY = null;         // previous y_normalized while gesture active
  let lastScrollHand = null;      // 'left' | 'right' — locked once the gesture starts

  function applyHandScroll(hands) {
    if (!hands) { lastScrollY = null; lastScrollHand = null; return; }
    // Find an active scroll gesture on either hand. Prefer the hand that was
    // already scrolling so a brief detection wobble doesn't reset the delta.
    const candidates = [];
    if (hands.right && hands.right.scroll && hands.right.scroll.active) candidates.push(['right', hands.right.scroll]);
    if (hands.left  && hands.left.scroll  && hands.left.scroll.active)  candidates.push(['left',  hands.left.scroll]);
    if (candidates.length === 0) {
      lastScrollY = null;
      lastScrollHand = null;
      return;
    }
    let pick = candidates.find(([name]) => name === lastScrollHand) || candidates[0];
    const [handName, sh] = pick;
    const y = sh.y;                            // 0..1, top of frame = 0
    if (lastScrollHand !== handName) {
      // Switched hands or just started — establish a baseline this frame.
      lastScrollHand = handName;
      lastScrollY = y;
      return;
    }
    if (lastScrollY === null) {
      lastScrollY = y;
      return;
    }
    const dy = y - lastScrollY;
    lastScrollY = y;
    // Pick the list based on which half of the frame the gesture is in.
    const target = (y < 0.5) ? myHandsListEl : oppHandsListEl;
    target.scrollTop += dy * SCROLL_GAIN;
    updateHandListFade(target);
  }

  function renderCardOverlay(cards, frameW, frameH) {
    if (!cards || !cards.length) {
      cardOverlay.innerHTML = '';
      return;
    }
    // Use the same object-fit:cover mapping as anchorToFace().
    cardOverlay.innerHTML = cards.map(c => {
      const r = frameToScreen(c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h, frameW, frameH);
      const cls = `card-box ${c.status}`;
      return `<div class="${cls}" style="left:${r.x}px;top:${r.y}px;width:${r.w}px;height:${r.h}px;">
        <span class="card-box-label">${c.label} ${c.status}</span>
      </div>`;
    }).join('');
  }

  function applyPoker(m) {
    const ctx = m.context || 'none';
    // Cards UI is the default. The only context that swaps to the face UI is
    // pure 'face' (face detected, no cards, no saved hand). Everything else —
    // 'poker', 'hybrid_poker', 'hybrid' (face + saved hand), and 'none' —
    // shows the cards dashboard, with empty-state hints when there's no data.
    const cardsMode = (ctx !== 'face');

    // Card bounding boxes — drawn whenever the server has any stable card
    // entries (not just current-frame YOLO hits). This way a card that's
    // already finalized keeps its box during a brief YOLO miss.
    const allCards = m.cards || [];
    if (allCards.length > 0) {
      cardOverlay.classList.remove('hidden');
      renderCardOverlay(allCards, m.frame_w, m.frame_h);
    } else {
      cardOverlay.classList.add('hidden');
      cardOverlay.innerHTML = '';
    }

    if (!cardsMode) {
      panelWin.classList.add('hidden');
      panelBoard.classList.add('hidden');
      panelMyHands.classList.add('hidden');
      panelOppHands.classList.add('hidden');
      lhLoader.classList.add('hidden');
      rhLoader.classList.add('hidden');
      return;
    }

    // Win panel — only when a hand is registered (otherwise equity is meaningless)
    const hasHand = (m.registered_hand || []).length === 2;
    if (hasHand) {
      panelWin.classList.remove('hidden');
      const eq = Math.max(0, Math.min(100, m.equity || 0));
      winPctEl.textContent = `${eq.toFixed(1)}%`;
      winOutsEl.textContent = `OUTS ${m.outs || 0}`;
      winBar.style.width = `${eq}%`;
      winBar.style.setProperty('--intensity', (eq / 100).toFixed(2));
    } else {
      panelWin.classList.add('hidden');
    }

    // Board / saved-hand panel: eyebrows always show so the user can read
    // what each gesture does; the cards row OR the hint line shows under
    // each eyebrow depending on whether that section has data.
    panelBoard.classList.remove('hidden');
    if (hasHand) {
      const visibleCurrent = new Set(allCards.map(c => c.treys));
      myHandCardsEl.innerHTML = m.registered_hand
        .map(c => renderMiniCard(c, !visibleCurrent.has(c)))
        .join('');
      myHandHintEl.classList.add('hidden');
    } else {
      myHandCardsEl.innerHTML = '';
      myHandHintEl.classList.remove('hidden');
    }
    const board = m.board_cards || [];
    if (board.length) {
      const streetLabels = { 3: 'Board · Flop', 4: 'Board · Turn', 5: 'Board · River' };
      boardEyebrowEl.textContent = streetLabels[board.length] || `Board · ${board.length}C`;
      boardCardsEl.innerHTML = board.map(c => renderMiniCard(c, false)).join('');
      boardHintEl.classList.add('hidden');
    } else {
      boardEyebrowEl.textContent = 'Board';
      boardCardsEl.innerHTML = '';
      boardHintEl.classList.remove('hidden');
    }

    // Floating circular pinch loaders. They appear at each hand's mid-point
    // when a pinch starts being counted and fill green -> yellow -> red as
    // the hold progresses. Hidden when not pinching or when the gesture is
    // grip-rejected (server stops sending hold > 0).
    updatePinchLoader(lhLoader, m.hands && m.hands.left,  m.lh_hold || 0,
                      hasHand ? 'Save Hand' : 'Save Hand', m.frame_w, m.frame_h);
    updatePinchLoader(rhLoader, m.hands && m.hands.right, m.rh_hold || 0,
                      'Lock Board', m.frame_w, m.frame_h);

    // Hand-rank lists — only meaningful once a hand is saved
    if (hasHand) {
      panelMyHands.classList.remove('hidden');
      panelOppHands.classList.remove('hidden');
      renderHandList(myHandsListEl, m.my_hands || []);
      renderHandList(oppHandsListEl, m.opp_hands || []);
    } else {
      panelMyHands.classList.add('hidden');
      panelOppHands.classList.add('hidden');
    }
  }

  function applyContextVisibility(ctx, faceDetected) {
    // Face UI shows whenever a face is detected, regardless of whether
    // cards are also visible (hybrid_poker). Cards have their own overlays
    // and don't fight the face panels for screen space.
    const showFace = faceDetected && (ctx === 'face' || ctx === 'hybrid_poker');
    if (showFace) {
      panelHR.classList.remove('hidden');
      panelExpr.classList.remove('hidden');
      panelGraph.classList.remove('hidden');
    } else {
      panelHR.classList.add('hidden');
      panelExpr.classList.add('hidden');
      panelGraph.classList.add('hidden');
    }
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

    currentContext = m.context || 'none';

    // Poker overlay (cards, win-equity, board, hand lists) — owns the screen
    // when context is 'poker' or 'hybrid_poker'. Otherwise everything below
    // collapses to the existing face UI.
    applyPoker(m);

    // Anchor face panels whenever a face is in view, regardless of whether
    // cards are also present (hybrid_poker mode shows both).
    const faceIsActive = !!m.face_detected
                      && m.face_bbox
                      && (currentContext === 'face' || currentContext === 'hybrid_poker');
    if (faceIsActive) {
      anchorToFace(m.face_bbox);
    }
    applyContextVisibility(currentContext, !!m.face_detected);

    // Gesture routing: panel-drag while face is active, otherwise cards-list
    // scroll. Right-pinch can drive the board lock without fighting the drag
    // handler because the drag handler doesn't run in pure-cards mode.
    if (faceIsActive) {
      applyHands(m.hands);
    } else {
      if (dragPanelName) {
        endDrag();
      }
      applyHandScroll(m.hands);
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
      // Server only sends text (metrics JSON) now; no more binary frames.
      if (typeof ev.data === 'string') {
        try { applyMetrics(JSON.parse(ev.data)); } catch (e) {}
      }
    };

    ws.onclose = () => {
      setStatus('Reconnecting', 'disconnected');
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 1500);
    };
    ws.onerror = () => { try { ws.close(); } catch (_) {} };
  }

  // ----- Camera (browser-side getUserMedia) -----
  const FACING_KEY = 'stoned.facingMode.v1';
  let currentFacing = (() => {
    try { return localStorage.getItem(FACING_KEY) || 'user'; } catch (_) { return 'user'; }
  })();

  async function startCamera(facing) {
    facing = facing || currentFacing || 'user';

    // Camera APIs are gated by "secure context": HTTPS or localhost only.
    const isSecure = window.isSecureContext === true
                  || location.hostname === 'localhost'
                  || location.hostname === '127.0.0.1';
    const hasMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

    if (!isSecure || !hasMedia) {
      setPlaceholder(
        'Camera blocked: insecure connection',
        `Browsers only allow camera access on https:// or localhost. ` +
        `Restart the laptop server with HTTPS:  python web_ui/server.py --https  ` +
        `then open https://${location.host} on the phone (accept the security warning once). ` +
        `Or open http://localhost:8000 on the laptop itself.`
      );
      return false;
    }

    // If a stream is already running, stop it first so the new facing mode
    // can take over.
    if (cameraStream) {
      try { cameraStream.getTracks().forEach(t => t.stop()); } catch (_) {}
      cameraStream = null;
    }

    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facing },
          width:  { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      camImg.srcObject = cameraStream;
      await camImg.play();
      placeholderEl.classList.add('hidden');
      camSwitchBtn.classList.remove('hidden');
      currentFacing = facing;
      try { localStorage.setItem(FACING_KEY, facing); } catch (_) {}
      // Mirror only when using the front camera (selfie convention).
      camImg.style.transform = (facing === 'user') ? 'scaleX(-1)' : '';
      startCaptureLoop();
      return true;
    } catch (e) {
      console.error('[camera] getUserMedia failed:', e);
      const denied = e && (e.name === 'NotAllowedError' || e.name === 'SecurityError');
      setPlaceholder(
        denied ? 'Camera blocked' : 'Camera failed',
        denied
          ? 'Allow camera access in your browser settings, then reload this page.'
          : (e && e.message) || 'Could not access the camera. Try reloading or use a different browser.'
      );
      return false;
    }
  }

  async function switchCamera() {
    const next = (currentFacing === 'user') ? 'environment' : 'user';
    await startCamera(next);
  }
  camSwitchBtn.addEventListener('click', switchCamera);

  function setPlaceholder(title, sub) {
    placeholderEl.classList.remove('hidden');
    const t = document.getElementById('placeholder-title');
    const s = document.getElementById('placeholder-sub');
    if (t) t.textContent = title;
    if (s) s.textContent = sub;
  }

  // ----- Frame capture + upload -----
  // Periodically grab the current <video> frame, encode JPEG, send to server.
  // The server processes it and returns metrics over the same socket.
  const TARGET_UPLOAD_FPS = 15;
  const UPLOAD_PERIOD_MS  = 1000 / TARGET_UPLOAD_FPS;
  const MAX_FRAME_W = 960;        // cap upload resolution for bandwidth
  const JPEG_QUALITY = 0.7;
  const _capCanvas = document.createElement('canvas');
  const _capCtx    = _capCanvas.getContext('2d');
  let _busy = false;              // single in-flight upload at a time

  function startCaptureLoop() {
    if (captureTimer) return;
    const loop = () => {
      captureFrame();
      captureTimer = setTimeout(loop, UPLOAD_PERIOD_MS);
    };
    loop();
  }

  function captureFrame() {
    if (_busy) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (camImg.readyState < 2 || !camImg.videoWidth) return;

    const vw = camImg.videoWidth;
    const vh = camImg.videoHeight;
    const scale = Math.min(1, MAX_FRAME_W / vw);
    const w = Math.round(vw * scale);
    const h = Math.round(vh * scale);
    if (_capCanvas.width !== w)  _capCanvas.width  = w;
    if (_capCanvas.height !== h) _capCanvas.height = h;
    // Draw the un-mirrored video into the canvas so the server sees the
    // raw camera. (CSS mirrors the on-screen <video>, the canvas does not.)
    _capCtx.drawImage(camImg, 0, 0, w, h);

    _busy = true;
    _capCanvas.toBlob(async (blob) => {
      _busy = false;
      if (!blob) return;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      try {
        const buf = await blob.arrayBuffer();
        ws.send(buf);
      } catch (_) {}
    }, 'image/jpeg', JPEG_QUALITY);
  }

  // ----- Touch gestures (mobile) -----
  // Hand-pinch detection requires the hands to be visible in the camera.
  // When the front camera is pointed at a face the hands are below the
  // phone, so MediaPipe never sees them. Touch is the natural mobile
  // equivalent: tap-and-drag to move a panel, two-finger pinch to resize.
  // Both reuse the same offset / scale persistence as the hand gestures.
  function attachTouchGestures(el, name) {
    let touchDrag = null;     // { id, dx, dy }   - one-finger drag state
    let touchPinch = null;    // { dist0, scale0 } - two-finger pinch state

    function getPointerById(touches, id) {
      for (let i = 0; i < touches.length; i++) {
        if (touches[i].identifier === id) return touches[i];
      }
      return null;
    }

    el.addEventListener('touchstart', (ev) => {
      ev.preventDefault();
      const ts = ev.touches;
      if (ts.length === 1 && !touchPinch) {
        // Begin drag
        const r = el.getBoundingClientRect();
        touchDrag = {
          id: ts[0].identifier,
          dx: ts[0].clientX - r.left,
          dy: ts[0].clientY - r.top,
        };
        el.style.transition = 'none';
        el.classList.add('dragging');
      } else if (ts.length >= 2) {
        // Begin pinch zoom (cancel any drag)
        touchDrag = null;
        el.classList.remove('dragging');
        el.style.transition = '';
        const ax = ts[0].clientX, ay = ts[0].clientY;
        const bx = ts[1].clientX, by = ts[1].clientY;
        touchPinch = {
          dist0: Math.hypot(ax - bx, ay - by) || 1,
          scale0: panelScales[name] || 1,
        };
        el.classList.add('zooming');
      }
    }, { passive: false });

    el.addEventListener('touchmove', (ev) => {
      ev.preventDefault();
      if (touchPinch && ev.touches.length >= 2) {
        const ax = ev.touches[0].clientX, ay = ev.touches[0].clientY;
        const bx = ev.touches[1].clientX, by = ev.touches[1].clientY;
        const dist = Math.hypot(ax - bx, ay - by) || 1;
        const newScale = clamp(touchPinch.scale0 * (dist / touchPinch.dist0),
                               MIN_SCALE, MAX_SCALE);
        panelScales[name] = newScale;
        applyPanelScale(name);
      } else if (touchDrag) {
        const t = getPointerById(ev.touches, touchDrag.id);
        if (!t) return;
        const newLeft = clamp(t.clientX - touchDrag.dx, 8,
                              window.innerWidth - el.offsetWidth - 8);
        const newTop  = clamp(t.clientY - touchDrag.dy, 8,
                              window.innerHeight - el.offsetHeight - 8);
        el.style.left = `${newLeft}px`;
        el.style.top  = `${newTop}px`;
      }
    }, { passive: false });

    function endTouch(ev) {
      // If pinch ends -> save scale
      if (touchPinch && ev.touches.length < 2) {
        el.classList.remove('zooming');
        saveScales();
        touchPinch = null;
      }
      // If all touches gone and we were dragging -> save offset
      if (touchDrag && ev.touches.length === 0) {
        el.style.transition = '';
        el.classList.remove('dragging');
        if (lastBbox) {
          const r = frameToScreen(lastBbox.x, lastBbox.y, lastBbox.w, lastBbox.h,
                                  lastFrameW, lastFrameH);
          const baseLeft = (name === 'hr')
            ? r.x + r.w + 20
            : r.x - el.offsetWidth - 20;
          const baseTop = r.y + (r.h - el.offsetHeight) / 2;
          const cur = el.getBoundingClientRect();
          panelOffsets[name] = {
            dx: cur.left - baseLeft,
            dy: cur.top  - baseTop,
          };
          saveOffsets();
        }
        touchDrag = null;
        resolveOverlaps();
      }
    }
    el.addEventListener('touchend',    endTouch, { passive: false });
    el.addEventListener('touchcancel', endTouch, { passive: false });
  }

  attachTouchGestures(panelHR,   'hr');
  attachTouchGestures(panelExpr, 'expr');

  // ----- Boot -----
  function boot() {
    connect();
    // The camera permission prompt only fires after a user gesture on iOS,
    // so we wait for the first tap. On Android Chrome the first call works
    // without a tap, but waiting also works.
    const onFirstTap = async () => {
      document.removeEventListener('click',    onFirstTap);
      document.removeEventListener('touchend', onFirstTap);
      // Try fullscreen + camera together
      try {
        const el = document.documentElement;
        const req = el.requestFullscreen || el.webkitRequestFullscreen;
        if (req && !document.fullscreenElement) req.call(el).catch(() => {});
      } catch (_) {}
      await startCamera();
    };
    document.addEventListener('click',    onFirstTap, { once: true });
    document.addEventListener('touchend', onFirstTap, { once: true });
  }

  boot();
})();
