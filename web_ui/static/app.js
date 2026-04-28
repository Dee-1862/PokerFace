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
  const panelVerdict    = $('panel-verdict');
  const verdictLabel    = $('verdict-label');
  const verdictBar      = $('verdict-bar');
  const verdictConf     = $('verdict-conf');
  const verdictMode     = $('verdict-mode');
  const btnStrong       = $('btn-strong');
  const btnBluff        = $('btn-bluff');
  const verdictToggle   = $('verdict-toggle');
  const verdictClose    = $('verdict-close');
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
  const myHandsToggle   = $('my-hands-toggle');
  const myHandsClose    = $('my-hands-close');
  const oppHandsToggle  = $('opp-hands-toggle');
  const oppHandsClose   = $('opp-hands-close');

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
  let ws = null;
  let reconnectTimer = null;
  let cameraStream = null;     // MediaStream from getUserMedia
  let captureTimer = null;     // setTimeout handle for the capture loop

  // -------- Touch drag/zoom limits --------
  // Shared by every screen-pinned panel via attachScreenPanelGestures.
  const MIN_SCALE = 0.65, MAX_SCALE = 2.0;

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

  // Face-anchor logic was removed. All panels are now screen-pinned: their
  // default position comes from CSS, the user can drag them anywhere, and
  // their positions persist across reloads via attachScreenPanelGestures.

  // Hand-gesture pinch-drag and pinch-zoom for the face panels were removed.
  // On a phone the hands holding the device aren't visible to MediaPipe, so
  // these gestures rarely fire reliably. The unified touch handler below
  // (attachScreenPanelGestures) now owns drag + pinch zoom for every panel.

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
    // Reuse the object-fit:cover mapping that the card overlay already does.
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
      const computing = !!m.equity_computing;
      winPctEl.textContent = computing ? '…' : `${eq.toFixed(1)}%`;
      winOutsEl.textContent = computing ? 'COMPUTING' : `OUTS ${m.outs || 0}`;
      winBar.style.width = `${computing ? 0 : eq}%`;
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

    // Hand-rank lists — only meaningful once a hand is saved. Each panel
    // collapses to a small launcher icon by default; the user expands it
    // explicitly when they want to see the full distribution.
    if (hasHand) {
      const expandMine = !!handPanelExpanded['my-hands'];
      const expandOpp  = !!handPanelExpanded['opp-hands'];
      myHandsToggle.classList.toggle('hidden',  expandMine);
      oppHandsToggle.classList.toggle('hidden', expandOpp);
      panelMyHands.classList.toggle('hidden',  !expandMine);
      panelOppHands.classList.toggle('hidden', !expandOpp);
      if (expandMine) renderHandList(myHandsListEl,  m.my_hands  || []);
      if (expandOpp)  renderHandList(oppHandsListEl, m.opp_hands || []);
    } else {
      panelMyHands.classList.add('hidden');
      panelOppHands.classList.add('hidden');
      myHandsToggle.classList.add('hidden');
      oppHandsToggle.classList.add('hidden');
    }
  }

  // ---- Bluff verdict rendering ----
  // The verdict has two presentations:
  //   1. A floating eye icon in the bottom-right (always shown when face +
  //      verdict are live). The icon's color reflects the current call.
  //   2. The full panel with confidence bar + showdown buttons. Hidden by
  //      default; expanded when the user clicks the icon, dismissed via × .
  // The user's preference (expanded / collapsed) is persisted so it sticks
  // across reloads.
  const VERDICT_EXPANDED_KEY = 'stoned.verdictExpanded.v1';
  let verdictExpanded = (function () {
    try { return localStorage.getItem(VERDICT_EXPANDED_KEY) === '1'; }
    catch (_) { return false; }
  })();
  function setVerdictExpanded(open) {
    verdictExpanded = !!open;
    try { localStorage.setItem(VERDICT_EXPANDED_KEY, verdictExpanded ? '1' : '0'); }
    catch (_) {}
  }
  if (verdictToggle) {
    verdictToggle.addEventListener('click', () => setVerdictExpanded(true));
  }
  if (verdictClose) {
    verdictClose.addEventListener('click', (ev) => {
      ev.stopPropagation();
      setVerdictExpanded(false);
    });
  }

  function renderVerdict(verdict, faceIsActive) {
    // No face / no verdict -> hide both icon and panel.
    if (!faceIsActive || !verdict) {
      panelVerdict.classList.add('hidden');
      verdictToggle.classList.add('hidden');
      verdictToggle.classList.remove('is-bluff', 'is-strong');
      return;
    }

    const isBluff = verdict.prediction === 'BLUFFING';

    // ---- Always-on launcher icon ----
    verdictToggle.classList.toggle('hidden', verdictExpanded);
    verdictToggle.classList.toggle('is-bluff',  isBluff);
    verdictToggle.classList.toggle('is-strong', !isBluff);

    // ---- Expanded panel (only when the user has opened it) ----
    panelVerdict.classList.toggle('hidden', !verdictExpanded);
    if (!verdictExpanded) return;

    verdictLabel.textContent = verdict.prediction;
    verdictLabel.classList.toggle('verdict-bluffing', isBluff);
    verdictLabel.classList.toggle('verdict-strong',  !isBluff);

    const conf = Math.max(0, Math.min(1, verdict.confidence || 0));
    verdictBar.style.width = `${(verdict.p_bluff || 0) * 100}%`;
    verdictBar.style.setProperty('--intensity', conf.toFixed(2));

    verdictConf.textContent = `${Math.round(conf * 100)}% confidence`;
    verdictMode.textContent = verdict.mode_tag || '';
  }

  // ---- Hand-rank launcher state (My / Opponent winning hands) ----
  // Both panels start collapsed as small icons; clicking the icon opens the
  // full panel and × in the panel collapses it back.
  const HAND_PANEL_KEYS = {
    'my-hands':  'stoned.myHandsExpanded.v1',
    'opp-hands': 'stoned.oppHandsExpanded.v1',
  };
  const handPanelExpanded = {
    'my-hands':  (function () {
      try { return localStorage.getItem(HAND_PANEL_KEYS['my-hands']) === '1'; }
      catch (_) { return false; }
    })(),
    'opp-hands': (function () {
      try { return localStorage.getItem(HAND_PANEL_KEYS['opp-hands']) === '1'; }
      catch (_) { return false; }
    })(),
  };
  function setHandPanelExpanded(name, open) {
    handPanelExpanded[name] = !!open;
    try { localStorage.setItem(HAND_PANEL_KEYS[name], open ? '1' : '0'); }
    catch (_) {}
  }
  if (myHandsToggle)  myHandsToggle.addEventListener('click',  () => setHandPanelExpanded('my-hands',  true));
  if (myHandsClose)   myHandsClose.addEventListener('click',   (ev) => { ev.stopPropagation(); setHandPanelExpanded('my-hands',  false); });
  if (oppHandsToggle) oppHandsToggle.addEventListener('click', () => setHandPanelExpanded('opp-hands', true));
  if (oppHandsClose)  oppHandsClose.addEventListener('click',  (ev) => { ev.stopPropagation(); setHandPanelExpanded('opp-hands', false); });

  // Send a manual showdown to the server (button presses)
  function sendShowdown(wasBluffing) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ type: 'showdown', was_bluffing: !!wasBluffing }));
    } catch (_) {}
    // Tiny visual ack on the pressed button
    const btn = wasBluffing ? btnBluff : btnStrong;
    btn.style.borderColor = wasBluffing ? 'var(--accent-red)' : 'var(--accent-green)';
    setTimeout(() => { btn.style.borderColor = ''; }, 350);
  }
  if (btnStrong) btnStrong.addEventListener('click', () => sendShowdown(false));
  if (btnBluff)  btnBluff .addEventListener('click', () => sendShowdown(true));

  // ---- Board / hand buttons (touch alternative to pinch-and-hold) ----
  function sendCardAction(action, btn) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ type: 'card_action', action }));
    } catch (_) {}
    if (btn) {
      btn.style.borderColor = 'var(--accent-yellow)';
      setTimeout(() => { btn.style.borderColor = ''; }, 280);
    }
  }
  [
    ['btn-save-hand',   'save_hand'],
    ['btn-reset-hand',  'reset_hand'],
    ['btn-lock-board',  'lock_board'],
    ['btn-reset-board', 'reset_board'],
    ['btn-reset-all',   'reset_all'],
  ].forEach(([id, action]) => {
    const b = document.getElementById(id);
    if (b) b.addEventListener('click', () => sendCardAction(action, b));
  });

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
      // Verdict panel + its launcher icon both follow face visibility.
      panelVerdict.classList.add('hidden');
      verdictToggle.classList.add('hidden');
    }
  }

  function applyMetrics(m) {
    if (m.frame_w && m.frame_h) {
      lastFrameW = m.frame_w;
      lastFrameH = m.frame_h;
    }

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

    // All panels are screen-pinned now. The previous face-anchoring code
    // moved them every frame, which fought the touch-drag handlers and the
    // user wanted simple "drop it where I put it" behaviour.
    const faceIsActive = !!m.face_detected
                      && m.face_bbox
                      && (currentContext === 'face' || currentContext === 'hybrid_poker');
    applyContextVisibility(currentContext, !!m.face_detected);

    // Bluff verdict panel
    renderVerdict(m.verdict, faceIsActive);

    // Hand-gesture scroll only - panel drag/zoom now lives entirely in the
    // touch handlers (attachScreenPanelGestures) so MediaPipe noise can't
    // bump panels around.
    applyHandScroll(m.hands);
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

  // Panels are screen-pinned, so a resize doesn't need to reposition them
  // (CSS handles edge clamping during drag). Hook left intentionally empty.

  // ---- WebSocket connection ----
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws`;
    setStatus('Connecting', '');
    console.log('[stoned] opening WebSocket to', url);
    ws = new WebSocket(url);
    ws.binaryType = 'blob';

    ws.onopen = () => {
      console.log('[stoned] WebSocket open');
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

    ws.onclose = (ev) => {
      console.log('[stoned] WebSocket closed', ev && ev.code, ev && ev.reason);
      setStatus('Reconnecting', 'disconnected');
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 1500);
    };
    ws.onerror = (ev) => {
      console.warn('[stoned] WebSocket error', ev);
      try { ws.close(); } catch (_) {}
    };
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
      // Front camera: mirror like a real mirror so what you see matches your
      // perspective (raise right hand -> appears on right of screen).
      // Back camera: no mirror (raw view).
      camImg.style.transform = (facing === 'user') ? 'scaleX(-1)' : 'scaleX(1)';
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
  // ----- Touch drag + pinch-zoom for screen-anchored panels -----
  // Verdict + poker panels aren't face-anchored, so each panel just persists
  // its own left/top + scale to localStorage. One-finger drags, two-finger
  // pinches resize (minimize/maximize). Buttons inside the panel keep
  // working because we ignore touchstart that lands on a <button>.
  const POKER_OFFSET_KEY = 'stoned.pokerPanelOffsets.v1';
  const POKER_SCALE_KEY  = 'stoned.pokerPanelScales.v1';
  let pokerPanelOffsets = (function () {
    try {
      const raw = localStorage.getItem(POKER_OFFSET_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (_) { return {}; }
  })();
  let pokerPanelScales = (function () {
    try {
      const raw = localStorage.getItem(POKER_SCALE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (_) { return {}; }
  })();
  function savePokerOffsets() {
    try { localStorage.setItem(POKER_OFFSET_KEY, JSON.stringify(pokerPanelOffsets)); } catch (_) {}
  }
  function savePokerScales() {
    try { localStorage.setItem(POKER_SCALE_KEY, JSON.stringify(pokerPanelScales)); } catch (_) {}
  }

  function attachPokerTouchDrag(el, name) {
    if (!el) return;
    // Restore persisted position + scale.
    const saved = pokerPanelOffsets[name];
    if (saved) {
      el.style.left   = `${saved.left}px`;
      el.style.top    = `${saved.top}px`;
      el.style.right  = 'auto';
      el.style.bottom = 'auto';
    }
    const savedScale = clamp(parseFloat(pokerPanelScales[name]) || 1,
                             MIN_SCALE, MAX_SCALE);
    pokerPanelScales[name] = savedScale;
    el.style.setProperty('--panel-scale', savedScale.toFixed(3));

    let drag  = null;     // { id, dx, dy }
    let pinch = null;     // { dist0, scale0 }

    el.addEventListener('touchstart', (ev) => {
      // Don't hijack button taps or touches that should scroll the inner
      // hand-list. Drag is still grabbable from the eyebrow / panel chrome.
      const tgt = ev.target;
      if (tgt && tgt.closest && (tgt.closest('button') || tgt.closest('.hand-list'))) return;
      const ts = ev.touches;
      if (ts.length === 1 && !pinch) {
        ev.preventDefault();
        const r = el.getBoundingClientRect();
        drag = {
          id: ts[0].identifier,
          dx: ts[0].clientX - r.left,
          dy: ts[0].clientY - r.top,
        };
        el.style.transition = 'none';
        el.classList.add('dragging');
      } else if (ts.length >= 2) {
        ev.preventDefault();
        // Two-finger pinch: cancel any drag, start zoom.
        drag = null;
        el.classList.remove('dragging');
        el.style.transition = '';
        // Pin the panel via explicit left/top before scaling so right/bottom-
        // anchored panels (e.g. My Hands) don't grow off-screen as scale rises.
        const r = el.getBoundingClientRect();
        if (!el.style.left || el.style.right !== 'auto') {
          el.style.left   = `${r.left}px`;
          el.style.top    = `${r.top}px`;
          el.style.right  = 'auto';
          el.style.bottom = 'auto';
        }
        const ax = ts[0].clientX, ay = ts[0].clientY;
        const bx = ts[1].clientX, by = ts[1].clientY;
        pinch = {
          dist0:  Math.hypot(ax - bx, ay - by) || 1,
          scale0: pokerPanelScales[name] || 1,
        };
        el.classList.add('zooming');
      }
    }, { passive: false });

    el.addEventListener('touchmove', (ev) => {
      if (pinch && ev.touches.length >= 2) {
        ev.preventDefault();
        const ax = ev.touches[0].clientX, ay = ev.touches[0].clientY;
        const bx = ev.touches[1].clientX, by = ev.touches[1].clientY;
        const dist = Math.hypot(ax - bx, ay - by) || 1;
        const newScale = clamp(pinch.scale0 * (dist / pinch.dist0),
                               MIN_SCALE, MAX_SCALE);
        pokerPanelScales[name] = newScale;
        el.style.setProperty('--panel-scale', newScale.toFixed(3));
      } else if (drag) {
        let t = null;
        for (let i = 0; i < ev.touches.length; i++) {
          if (ev.touches[i].identifier === drag.id) { t = ev.touches[i]; break; }
        }
        if (!t) return;
        ev.preventDefault();
        // Use the SCALED visual width so a pinch-shrunk panel can still be
        // pushed flush against the right/bottom edge. offsetWidth doesn't
        // account for transform: scale().
        const rect = el.getBoundingClientRect();
        const newLeft = clamp(t.clientX - drag.dx, 0,
                              Math.max(0, window.innerWidth  - rect.width));
        const newTop  = clamp(t.clientY - drag.dy, 0,
                              Math.max(0, window.innerHeight - rect.height));
        el.style.left   = `${newLeft}px`;
        el.style.top    = `${newTop}px`;
        el.style.right  = 'auto';
        el.style.bottom = 'auto';
      }
    }, { passive: false });

    function endTouch(ev) {
      // Pinch ends when a finger lifts (drops back below 2 touches).
      if (pinch && ev.touches.length < 2) {
        el.classList.remove('zooming');
        savePokerScales();
        pinch = null;
      }
      if (drag && ev.touches.length === 0) {
        el.style.transition = '';
        el.classList.remove('dragging');
        pokerPanelOffsets[name] = {
          left: parseFloat(el.style.left) || 0,
          top:  parseFloat(el.style.top)  || 0,
        };
        savePokerOffsets();
        drag = null;
      }
    }
    el.addEventListener('touchend',    endTouch, { passive: false });
    el.addEventListener('touchcancel', endTouch, { passive: false });
  }

  attachPokerTouchDrag(panelBoard,    'board');
  attachPokerTouchDrag(panelWin,      'win');
  attachPokerTouchDrag(panelMyHands,  'my-hands');
  attachPokerTouchDrag(panelOppHands, 'opp-hands');
  attachPokerTouchDrag(panelVerdict,  'verdict');
  // Face panels: previously had their own face-anchored drag handler. Now
  // they use the same screen-pinned gestures as every other panel.
  attachPokerTouchDrag(panelHR,   'hr');
  attachPokerTouchDrag(panelExpr, 'expr');

  // ----- Boot -----
  function boot() {
    connect();

    // Try to start the camera immediately. Browsers that need a user gesture
    // (mainly iOS Safari) will throw NotAllowedError; in that case we fall
    // back to a tap-anywhere handler. Android Chrome and desktop browsers
    // typically allow this without a tap.
    let started = false;
    startCamera().then(ok => { started = ok; });

    const onFirstInteraction = async () => {
      if (started) {
        document.removeEventListener('click',    onFirstInteraction);
        document.removeEventListener('touchend', onFirstInteraction);
        return;
      }
      // Request fullscreen on this user gesture (must be from a gesture)
      try {
        const el = document.documentElement;
        const req = el.requestFullscreen || el.webkitRequestFullscreen;
        if (req && !document.fullscreenElement) req.call(el).catch(() => {});
      } catch (_) {}
      const ok = await startCamera();
      if (ok) {
        document.removeEventListener('click',    onFirstInteraction);
        document.removeEventListener('touchend', onFirstInteraction);
        started = true;
      }
    };
    document.addEventListener('click',    onFirstInteraction);
    document.addEventListener('touchend', onFirstInteraction);
  }

  boot();
})();
