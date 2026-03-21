"""
main.py
=======
SyncopateAI – Production-ready Streamlit app that synchronises acoustic
fingerstyle guitar audio (audio_sync.py) with parsed ASCII/text tabs
(tab_engine.py).

Run with::

    streamlit run main.py

Features
--------
* Upload WAV/MP3 → saved to ``static/temp/`` (never Base64-encoded).
* Parse ASCII guitar tab into structured time slices.
* Extract note onsets via :class:`~audio_sync.AudioSyncer` with a
  configurable ``threshold_ignore_sec`` to skip metronome count-ins.
* Many-to-One / One-to-Many onset-to-slice mapping data structure.
* Export / Import sync data as JSON.
* Interactive HTML/JS component (served via ``st.components.v1.html``):

  - Dark-mode, monospace, horizontally scrolling tab.
  - Zoom (+ / −) controls that scale tab spacing proportionally.
  - ``requestAnimationFrame`` playhead anchored to ``audio.currentTime``.
  - Focus-regain catch-up: playhead snaps to correct position when the
    browser tab regains focus after background throttling.
  - Drag-and-drop onset marker repositioning with instant mapping update.
  - Web Audio API click track synthesised at each onset timestamp.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import uuid
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as components

from audio_engine import AudioAnalysisResult
from audio_sync import AudioSyncer
from tab_engine import TabParser
from vision_engine import FrameResult

# ---------------------------------------------------------------------------
# Constants & directory setup
# ---------------------------------------------------------------------------

#: Files uploaded by users are saved here.  Streamlit's static-file server
#: (enabled via .streamlit/config.toml) makes them available at
#: /app/static/temp/<filename> without Base64 encoding.
STATIC_TEMP_DIR = pathlib.Path(__file__).parent / "static" / "temp"
STATIC_TEMP_DIR.mkdir(parents=True, exist_ok=True)

#: Default height (px) for the interactive HTML component inside the app.
COMPONENT_HEIGHT: int = 520

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SyncopateAI",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helper utilities (kept for backward-compatibility with test_main.py)
# ---------------------------------------------------------------------------


def _build_finger_height_csv(
    results: list[FrameResult],
    audio_result: Optional[AudioAnalysisResult] = None,
) -> bytes:
    """
    Convert vision analysis results to a UTF-8 encoded CSV suitable for
    download via ``st.download_button``.

    Columns
    -------
    frame_index, timestamp_sec, avg_finger_height_px, efficiency_score,
    index_x, index_y, middle_x, middle_y, ring_x, ring_y, pinky_x, pinky_y
    [, Audio_Deviation_ms]  – only present when *audio_result* is supplied
                              and contains at least one onset event.

    For each video frame the ``Audio_Deviation_ms`` value is taken from the
    audio onset whose timestamp is closest to the frame's timestamp.  This
    creates a unified dataset that correlates hand technique with timing.
    """
    fingertip_names = ("index", "middle", "ring", "pinky")

    include_audio = (
        audio_result is not None and bool(audio_result.onset_events)
    )

    header = [
        "frame_index",
        "timestamp_sec",
        "avg_finger_height_px",
        "efficiency_score",
    ] + [f"{n}_{axis}" for n in fingertip_names for axis in ("x", "y")]

    if include_audio:
        header.append("Audio_Deviation_ms")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)

    # Pre-extract onset times for fast nearest-neighbour lookup.
    onset_times: list[float] = (
        [e.time_sec for e in audio_result.onset_events] if include_audio else []
    )

    for r in results:
        # Build a lookup so missing fingertips are represented as empty cells
        tip_coords: dict[str, tuple] = {ft.name: (ft.x, ft.y) for ft in r.fingertips}
        row = [
            r.frame_index,
            round(r.timestamp_sec, 6),
            round(r.avg_finger_height, 4) if r.avg_finger_height is not None else "",
            round(r.efficiency_score, 4) if r.efficiency_score is not None else "",
        ]
        for name in fingertip_names:
            if name in tip_coords:
                row += [round(tip_coords[name][0], 2), round(tip_coords[name][1], 2)]
            else:
                row += ["", ""]

        if include_audio:
            # Find the index of the onset whose time is closest to this frame.
            closest_idx = min(
                range(len(onset_times)),
                key=lambda i: abs(onset_times[i] - r.timestamp_sec),
            )
            row.append(round(audio_result.onset_events[closest_idx].deviation_ms, 4))

        writer.writerow(row)

    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Sync-specific helpers
# ---------------------------------------------------------------------------


def _save_audio(uploaded_file: Any) -> tuple[str, str]:
    """Save *uploaded_file* to ``static/temp/`` and return (local_path, url).

    The URL is a root-relative path that Streamlit's static-file server
    exposes at ``/app/static/temp/<filename>``.  Inside an ``st.components.v1.html``
    iframe (same origin), this resolves correctly without CORS issues.

    Parameters
    ----------
    uploaded_file:
        A Streamlit ``UploadedFile`` object (has ``.name`` and ``.read()``).

    Returns
    -------
    tuple[str, str]
        ``(local_path, static_url)`` where *local_path* is the absolute
        filesystem path and *static_url* is the root-relative URL.
    """
    # Use a UUID prefix so simultaneous sessions do not clash.
    safe_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    dest = STATIC_TEMP_DIR / safe_name
    dest.write_bytes(uploaded_file.read())
    url = f"/app/static/temp/{safe_name}"
    return str(dest), url


def build_onset_to_slice_map(
    onset_times: list[float],
    slices: list[dict],
    threshold_ignore_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Build an onset-to-slice mapping that supports Many-to-One and One-to-Many.

    Parameters
    ----------
    onset_times:
        Chronologically sorted onset timestamps in seconds.
    slices:
        Parsed tab time slices (output of :meth:`~tab_engine.TabParser.parse`).
    threshold_ignore_sec:
        Onsets earlier than this value are ignored (count-in mitigation).

    Returns
    -------
    list[dict]
        Each entry has the form::

            {
                "onset_idx":    int,          # index into onset_times
                "onset_time":   float,        # seconds
                "slice_indices": list[int],   # tab slice indices (1-to-many
                                              # when list has >1 element)
            }

        **Many-to-One**: multiple entries share the same ``slice_indices[0]``.
        **One-to-Many**: a single entry lists consecutive slice indices.

    Notes
    -----
    When there are *more* onsets than slices the algorithm performs a
    proportional many-to-one assignment.  When there are *fewer* onsets than
    slices, each onset is given a contiguous range of slices (one-to-many),
    which is the typical situation for hammer-on / pull-off sequences.
    """
    filtered: list[tuple[int, float]] = [
        (i, t) for i, t in enumerate(onset_times) if t >= threshold_ignore_sec
    ]

    if not filtered or not slices:
        return []

    n_o = len(filtered)
    n_s = len(slices)
    mapping: list[dict[str, Any]] = []

    if n_o >= n_s:
        # Many-to-One: proportionally map each onset to a single slice.
        for j, (orig_idx, t) in enumerate(filtered):
            slice_idx = min(round(j / n_o * (n_s - 1)), n_s - 1)
            mapping.append(
                {
                    "onset_idx": orig_idx,
                    "onset_time": round(t, 6),
                    "slice_indices": [slice_idx],
                }
            )
    else:
        # One-to-Many: partition slices evenly across the available onsets.
        for j, (orig_idx, t) in enumerate(filtered):
            s_start = round(j / n_o * n_s)
            s_end = round((j + 1) / n_o * n_s)
            indices = list(range(s_start, s_end)) or [min(s_start, n_s - 1)]
            mapping.append(
                {
                    "onset_idx": orig_idx,
                    "onset_time": round(t, 6),
                    "slice_indices": indices,
                }
            )

    return mapping


# ---------------------------------------------------------------------------
# Interactive HTML component
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
/* ── Reset & base ──────────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0}
body{background:#12121e;color:#e0e0e0;font-family:'Courier New',Courier,monospace;
     font-size:13px;overflow-x:hidden;user-select:none}

/* ── Controls bar ─────────────────────────────────────────── */
#controls{display:flex;align-items:center;gap:6px;padding:8px 12px;
  background:#1a1a30;border-bottom:1px solid #2a2a50;flex-wrap:wrap}
.cb{background:#23234a;color:#e0e0e0;border:1px solid #e94560;border-radius:4px;
  padding:5px 11px;cursor:pointer;font-family:inherit;font-size:12px;
  transition:background .15s,color .15s}
.cb:hover{background:#e94560;color:#fff}
.cb.on{background:#e94560;color:#fff}
#zoom-lbl{color:#8080b0;font-size:11px;min-width:48px;text-align:center}
#time-lbl{color:#8080b0;font-size:11px;margin-left:auto}

/* ── Time ruler ───────────────────────────────────────────── */
#ruler{position:relative;height:22px;background:#0e0e1c;
  border-bottom:1px solid #1e1e40;overflow:hidden;white-space:nowrap}
.r-tick{position:absolute;top:0;width:1px;background:#2a3050;height:10px}
.r-lbl{position:absolute;top:11px;font-size:8px;color:#505080;
  transform:translateX(-50%)}

/* ── Scrollable stage ─────────────────────────────────────── */
#stage-wrap{overflow-x:auto;overflow-y:hidden;position:relative;
  cursor:default;background:#12121e}

/* ── Tab canvas ───────────────────────────────────────────── */
#tab-canvas{position:relative}

/* ── String rows ──────────────────────────────────────────── */
.s-row{display:flex;align-items:center;height:22px;
  border-bottom:1px solid #0a0a18;position:relative}
.s-lbl{width:18px;text-align:center;color:#e94560;font-weight:bold;font-size:12px;
  position:sticky;left:0;background:#12121e;z-index:10;flex-shrink:0}
.s-track{position:relative;height:22px;flex-grow:1}
/* dash baseline */
.s-track::after{content:'';position:absolute;top:50%;left:0;right:0;
  height:1px;background:#1e2a3a;pointer-events:none}

/* ── Note cells ───────────────────────────────────────────── */
.nc{position:absolute;top:3px;height:16px;min-width:16px;text-align:center;
  font-size:11px;line-height:16px;color:#00ccff;background:#162840;
  border-radius:3px;padding:0 3px;transform:translateX(-50%);
  z-index:5;pointer-events:none;transition:background .1s}
.nc.harm{color:#ffd700;background:#2a2010}
.nc.slap{color:#ff6b6b;background:#2a1010}
.nc.slide{color:#a8e6cf;background:#102a1a}
.nc.cur{background:#e94560;color:#fff;z-index:6}

/* ── Onset-markers row ────────────────────────────────────── */
#onset-row{position:relative;height:48px;background:#0e0e1c;
  border-top:2px solid #1e1e40}
.om{position:absolute;width:13px;height:13px;border-radius:50%;
  background:#e94560;border:2px solid #ff8099;top:7px;
  transform:translateX(-50%);cursor:ew-resize;z-index:20;
  transition:background .1s}
.om:hover{background:#ff8099;transform:translateX(-50%) scale(1.3)}
.om.act{background:#00ccff;border-color:#fff}
.om.drag{background:#ffd700;opacity:.9;z-index:30;
  transform:translateX(-50%) scale(1.25)}
.ol{position:absolute;top:25px;font-size:8px;color:#505080;
  transform:translateX(-50%);white-space:nowrap;pointer-events:none}

/* ── Playhead ─────────────────────────────────────────────── */
#ph{position:absolute;top:0;width:2px;background:#00ccff;
  pointer-events:none;z-index:50;box-shadow:0 0 6px #00ccff}
#ph-knob{position:absolute;top:-5px;left:-5px;width:12px;height:12px;
  background:#00ccff;border-radius:50%}
</style>
</head>
<body>

<!-- Controls -->
<div id="controls">
  <button class="cb" id="btn-play">&#9654; Play</button>
  <button class="cb" id="btn-stop">&#9632; Stop</button>
  <button class="cb" id="btn-zi">&#xFF0B; Zoom</button>
  <span id="zoom-lbl">100%</span>
  <button class="cb" id="btn-zo">&#xFF0D; Zoom</button>
  <button class="cb" id="btn-ct" title="Toggle click track">&#128263; Click</button>
  <button class="cb" id="btn-ex">&#128229; Export JSON</button>
  <span id="time-lbl">0.00 s</span>
</div>

<!-- Time ruler -->
<div id="ruler"></div>

<!-- Scrollable stage -->
<div id="stage-wrap">
  <div id="tab-canvas">
    <div id="str-container"></div>
    <div id="ph"><div id="ph-knob"></div></div>
    <div id="onset-row"></div>
  </div>
</div>

<!-- Audio element: src set by JS from injected URL -->
<audio id="aud" preload="auto"></audio>

<script>
/* ============================================================
   DATA – injected by Python
   ============================================================ */
const SLICES        = /*SLICES*/[];
const ONSETS        = /*ONSETS*/[];   // [sec, ...]
let   MAPPING       = /*MAPPING*/[];  // [{onset_idx,onset_time,slice_indices:[...]},...]
const AUDIO_URL     = /*AUDIO_URL*/"";
const TOTAL_DUR     = /*TOTAL_DUR*/0;

/* ============================================================
   MODULE: State  (zoom & coordinate helpers)
   ============================================================ */
const State = (() => {
  const BASE_SW  = 60;   // slice width (px) at zoom=1
  const BASE_PPS = 80;   // pixels per second at zoom=1
  const MIN_Z = 0.25, MAX_Z = 8;
  let z = 1;
  return {
    get zoom()       { return z; },
    get sw()         { return BASE_SW * z; },
    get pps()        { return BASE_PPS * z; },
    get totalW()     {
      return Math.max(SLICES.length * BASE_SW * z + 60,
                      TOTAL_DUR  * BASE_PPS * z + 60, 300);
    },
    sliceX(i)        { return 22 + i * BASE_SW * z; },
    zoomIn()         { z = Math.min(MAX_Z, z * 1.5); },
    zoomOut()        { z = Math.max(MIN_Z, z / 1.5); },
  };
})();

/* ============================================================
   MODULE: TabRenderer
   ============================================================ */
const TabRenderer = (() => {
  const ORDER = ['e','B','G','D','A','E'];

  function cls(tech) {
    if (/harmonic/.test(tech)) return ' harm';
    if (tech === 'slap')       return ' slap';
    if (/slide/.test(tech))    return ' slide';
    return '';
  }

  function render(container) {
    container.innerHTML = '';
    const tw = State.totalW;
    ORDER.forEach(str => {
      const row = document.createElement('div');
      row.className = 's-row';
      row.style.width = (tw + 22) + 'px';

      const lbl = document.createElement('div');
      lbl.className = 's-lbl';
      lbl.textContent = str;
      row.appendChild(lbl);

      const track = document.createElement('div');
      track.className = 's-track';
      track.style.width = tw + 'px';

      SLICES.forEach((sl, idx) => {
        const note = sl.notes.find(n => n.string === str);
        if (!note) return;
        const cell = document.createElement('div');
        cell.className = 'nc' + cls(note.technique);
        cell.dataset.slice = idx;
        cell.style.left = State.sliceX(idx) + 'px';
        cell.textContent = note.fret;
        cell.title = str + ':' + note.fret + ' (' + note.technique + ') slice ' + idx;
        track.appendChild(cell);
      });

      row.appendChild(track);
      container.appendChild(row);
    });
  }

  return { render };
})();

/* ============================================================
   MODULE: ClickTrackEngine  (Web Audio API)
   ============================================================ */
const ClickTrackEngine = (() => {
  let ctx = null, on = false, lastSched = -Infinity;
  const LOOK = 0.12; // schedule window (s)

  function _tick(targetCtxTime) {
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = 1000;
    gain.gain.setValueAtTime(0.35, targetCtxTime);
    gain.gain.exponentialRampToValueAtTime(0.001, targetCtxTime + 0.055);
    osc.start(targetCtxTime);
    osc.stop(targetCtxTime + 0.07);
  }

  function _init() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === 'suspended') ctx.resume();
  }

  return {
    toggle() { _init(); on = !on; return on; },
    get enabled() { return on; },
    reset() { lastSched = -Infinity; },
    schedule(audioEl) {
      if (!on || !ctx) return;
      const now    = ctx.currentTime;
      const audT   = audioEl.currentTime;
      const offset = now - audT;          // AudioContext time − audio time
      ONSETS.forEach(t => {
        if (t > lastSched && t >= audT && t <= audT + LOOK) {
          const ct = t + offset;
          if (ct > now) { _tick(ct); lastSched = t; }
        }
      });
    },
  };
})();

/* ============================================================
   MODULE: DragDropManager
   ============================================================ */
const DragDropManager = (() => {
  let dragIdx = null, startClientX = 0, startLeft = 0;

  function _nearest(xCanvas) {
    let best = 0, bestD = Infinity;
    SLICES.forEach((_, i) => {
      const d = Math.abs(State.sliceX(i) - xCanvas);
      if (d < bestD) { bestD = d; best = i; }
    });
    return best;
  }

  function _snap(idx) {
    const el = document.getElementById('om-' + idx);
    if (!el) return;
    el.classList.remove('drag');
    const xCanvas = parseFloat(el.style.left) || 0;
    const si      = _nearest(xCanvas);
    const newX    = State.sliceX(si);
    el.style.left = newX + 'px';

    // Update label
    const lbl = document.getElementById('ol-' + idx);
    if (lbl) { lbl.style.left = newX + 'px'; lbl.textContent = 's' + si; }

    // Update MAPPING (supports many-to-one by dragging)
    const entry = MAPPING.find(m => m.onset_idx === idx);
    if (entry) entry.slice_indices = [si];

    dragIdx = null;
    document.body.style.cursor = '';
  }

  function attach(el, idx) {
    function onDown(cx) {
      dragIdx    = idx;
      startClientX = cx;
      startLeft  = parseFloat(el.style.left) || 0;
      el.classList.add('drag');
      document.body.style.cursor = 'ew-resize';
    }
    el.addEventListener('mousedown',  e => { e.preventDefault(); onDown(e.clientX); });
    el.addEventListener('touchstart', e => { e.preventDefault(); onDown(e.touches[0].clientX); },
                        { passive: false });
  }

  function initGlobalListeners() {
    function onMove(cx) {
      if (dragIdx === null) return;
      const el = document.getElementById('om-' + dragIdx);
      if (el) el.style.left = (startLeft + cx - startClientX) + 'px';
    }
    function onUp() { if (dragIdx !== null) _snap(dragIdx); }

    document.addEventListener('mousemove',  e => onMove(e.clientX));
    document.addEventListener('touchmove',  e => onMove(e.touches[0].clientX), { passive: true });
    document.addEventListener('mouseup',    onUp);
    document.addEventListener('touchend',   onUp);
  }

  return { attach, initGlobalListeners };
})();

/* ============================================================
   MODULE: OnsetRenderer
   ============================================================ */
const OnsetRenderer = (() => {
  function render(container) {
    container.innerHTML = '';
    container.style.width = State.totalW + 'px';
    MAPPING.forEach(entry => {
      const si = entry.slice_indices[0] ?? 0;
      const x  = State.sliceX(si);

      const mk = document.createElement('div');
      mk.className   = 'om';
      mk.id          = 'om-' + entry.onset_idx;
      mk.dataset.oi  = entry.onset_idx;
      mk.style.left  = x + 'px';
      mk.title       = 'Onset ' + entry.onset_idx + ': ' +
                       entry.onset_time.toFixed(3) + 's → slice ' + si;
      container.appendChild(mk);
      DragDropManager.attach(mk, entry.onset_idx);

      const lbl = document.createElement('div');
      lbl.className  = 'ol';
      lbl.id         = 'ol-' + entry.onset_idx;
      lbl.style.left = x + 'px';
      lbl.textContent = 's' + si;
      container.appendChild(lbl);
    });
  }
  return { render };
})();

/* ============================================================
   MODULE: RulerRenderer
   ============================================================ */
const RulerRenderer = (() => {
  function _interval() {
    const p = State.pps;
    if (p > 200) return 0.1;
    if (p > 80)  return 0.5;
    if (p > 30)  return 1.0;
    return 5.0;
  }
  function render(el) {
    el.innerHTML = '';
    el.style.width = State.totalW + 'px';
    if (TOTAL_DUR <= 0) return;
    const iv = _interval();
    for (let t = 0; t <= TOTAL_DUR + iv; t += iv) {
      const x = t * State.pps + 22;
      const tk = document.createElement('div');
      tk.className  = 'r-tick'; tk.style.left = x + 'px';
      el.appendChild(tk);
      if (Math.round(t / iv) % 5 === 0) {
        const lb = document.createElement('div');
        lb.className  = 'r-lbl'; lb.style.left = x + 'px';
        lb.textContent = t.toFixed(1) + 's';
        el.appendChild(lb);
      }
    }
  }
  return { render };
})();

/* ============================================================
   MODULE: PlayheadManager  (requestAnimationFrame loop)
   ============================================================ */
const PlayheadManager = (() => {
  let rafId = null, audioEl = null, lastSI = -1;

  function _activeSlice() {
    const t = audioEl.currentTime;
    let si = 0;
    for (const entry of MAPPING) {
      if (entry.onset_time <= t) si = entry.slice_indices[0] ?? 0;
      else break;
    }
    return si;
  }

  function _update() {
    const t  = audioEl.currentTime;
    const ph = document.getElementById('ph');
    if (ph) ph.style.left = (t * State.pps + 22) + 'px';

    const td = document.getElementById('time-lbl');
    if (td) td.textContent = t.toFixed(2) + ' s';

    const si = _activeSlice();
    if (si !== lastSI) {
      // Unhighlight old cells
      document.querySelectorAll('.nc.cur').forEach(c => c.classList.remove('cur'));
      // Highlight all cells in slice si (and any slice_indices)
      const activeSlices = new Set();
      MAPPING.forEach(m => {
        if (m.slice_indices[0] === si) m.slice_indices.forEach(x => activeSlices.add(x));
      });
      activeSlices.add(si);
      activeSlices.forEach(s => {
        document.querySelectorAll('.nc[data-slice="' + s + '"]')
                .forEach(c => c.classList.add('cur'));
      });
      // Onset marker highlight
      document.querySelectorAll('.om.act').forEach(m => m.classList.remove('act'));
      MAPPING.forEach(m => {
        if (m.slice_indices.includes(si)) {
          const mk = document.getElementById('om-' + m.onset_idx);
          if (mk) mk.classList.add('act');
        }
      });
      // Auto-scroll: keep playhead centred
      const wrap = document.getElementById('stage-wrap');
      if (wrap) wrap.scrollLeft = Math.max(0, State.sliceX(si) - wrap.clientWidth / 2);
      lastSI = si;
    }

    ClickTrackEngine.schedule(audioEl);
    rafId = requestAnimationFrame(_update);
  }

  return {
    start(aud) {
      audioEl = aud; lastSI = -1;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(_update);
    },
    stop() { if (rafId) { cancelAnimationFrame(rafId); rafId = null; } },
    reset() { lastSI = -1; },
    /** Called when browser tab regains focus – instant catch-up. */
    catchUp() {
      if (!audioEl) return;
      const t  = audioEl.currentTime;
      const ph = document.getElementById('ph');
      if (ph) ph.style.left = (t * State.pps + 22) + 'px';
      const wrap = document.getElementById('stage-wrap');
      if (wrap) {
        const si = _activeSlice();
        wrap.scrollLeft = Math.max(0, State.sliceX(si) - wrap.clientWidth / 2);
      }
    },
  };
})();

/* ============================================================
   MODULE: ExportManager
   ============================================================ */
const ExportManager = (() => ({
  export() {
    const payload = {
      onsets:         ONSETS,
      slices:         SLICES,
      mapping:        MAPPING,
      total_duration: TOTAL_DUR,
      exported_at:    new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)],
                          { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'syncopate_sync_' + Date.now() + '.json';
    a.click(); URL.revokeObjectURL(url);
  },
}))();

/* ============================================================
   Initialisation
   ============================================================ */
(function init() {
  const aud   = document.getElementById('aud');
  const wrap  = document.getElementById('stage-wrap');
  const strC  = document.getElementById('str-container');
  const orC   = document.getElementById('onset-row');
  const ruler = document.getElementById('ruler');
  const ph    = document.getElementById('ph');

  if (AUDIO_URL) aud.src = AUDIO_URL;

  function fullRender() {
    const tw = State.totalW;
    const tc = document.getElementById('tab-canvas');
    tc.style.width  = tw + 'px';
    tc.style.height = (22 * 6 + 48 + 2) + 'px';
    if (ph) ph.style.height = (22 * 6 + 48 + 2) + 'px';

    TabRenderer.render(strC);
    OnsetRenderer.render(orC);
    RulerRenderer.render(ruler);
    document.getElementById('zoom-lbl').textContent =
      Math.round(State.zoom * 100) + '%';
  }

  DragDropManager.initGlobalListeners();
  fullRender();

  /* ── Button handlers ──────────────────────────── */
  document.getElementById('btn-play').addEventListener('click', () => {
    if (!aud.src || aud.src === window.location.href) return;
    aud.play()
       .then(() => { PlayheadManager.start(aud); ClickTrackEngine.reset(); })
       .catch(err => console.warn('Audio play failed:', err));
  });

  document.getElementById('btn-stop').addEventListener('click', () => {
    aud.pause(); aud.currentTime = 0;
    PlayheadManager.stop(); PlayheadManager.reset(); ClickTrackEngine.reset();
    if (ph) ph.style.left = '22px';
    document.getElementById('time-lbl').textContent = '0.00 s';
    document.querySelectorAll('.nc.cur').forEach(c => c.classList.remove('cur'));
    document.querySelectorAll('.om.act').forEach(m => m.classList.remove('act'));
  });

  document.getElementById('btn-zi').addEventListener('click', () => {
    State.zoomIn(); fullRender();
  });
  document.getElementById('btn-zo').addEventListener('click', () => {
    State.zoomOut(); fullRender();
  });

  const ctBtn = document.getElementById('btn-ct');
  ctBtn.addEventListener('click', () => {
    const on = ClickTrackEngine.toggle();
    ctBtn.textContent = (on ? '\u{1F50A}' : '\u{1F507}') + ' Click';
    ctBtn.classList.toggle('on', on);
  });

  document.getElementById('btn-ex').addEventListener('click', () =>
    ExportManager.export()
  );

  /* ── Visibility change: catch up on focus return ── */
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) PlayheadManager.catchUp();
  });

  aud.addEventListener('ended', () => {
    PlayheadManager.stop(); PlayheadManager.reset(); ClickTrackEngine.reset();
  });
})();
</script>
</body>
</html>
"""


def _make_html_component(
    slices: list,
    onsets: list[float],
    mapping: list[dict],
    audio_url: str,
    total_duration: float,
) -> str:
    """Substitute Python data into the HTML template and return the full HTML string."""
    return (
        _HTML_TEMPLATE
        .replace("/*SLICES*/",    json.dumps(slices))
        .replace("/*ONSETS*/",    json.dumps(onsets))
        .replace("/*MAPPING*/",   json.dumps(mapping))
        .replace("/*AUDIO_URL*/", json.dumps(audio_url))
        .replace("/*TOTAL_DUR*/", str(total_duration))
    )


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------


def main() -> None:
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎸 SyncopateAI")
        st.markdown("**Analysis Parameters**")

        threshold_ignore_sec = st.slider(
            "Count-in threshold (s)",
            min_value=0.0, max_value=5.0, value=2.0, step=0.1,
            help=(
                "Onsets detected before this time are ignored to skip "
                "metronome count-ins."
            ),
        )
        bpm = st.number_input(
            "Target BPM", min_value=20, max_value=300, value=120, step=1,
        )
        wait_frames = st.slider(
            "Onset wait (frames)", min_value=1, max_value=30, value=8,
            help="Minimum frames between successive onsets (anti-double-trigger).",
        )
        pre_max = st.slider(
            "Pre-max (frames)", min_value=0, max_value=10, value=3,
        )
        component_height = st.slider(
            "Component height (px)", min_value=300, max_value=900,
            value=COMPONENT_HEIGHT, step=50,
        )

        st.divider()
        st.markdown("**Import Sync Data**")
        import_file = st.file_uploader(
            "Upload sync JSON", type=["json"], key="import_json",
        )

    # ── Session state initialisation ─────────────────────────────────────
    ss = st.session_state
    for key, default in [
        ("onset_times",    []),
        ("tab_slices",     []),
        ("sync_mapping",   []),
        ("audio_url",      ""),
        ("total_duration", 0.0),
    ]:
        if key not in ss:
            ss[key] = default

    # ── Handle JSON import ────────────────────────────────────────────────
    if import_file is not None:
        try:
            data = json.loads(import_file.read())
            ss["onset_times"]    = data.get("onsets", [])
            ss["tab_slices"]     = data.get("slices", [])
            ss["sync_mapping"]   = data.get("mapping", [])
            ss["total_duration"] = data.get("total_duration", 0.0)
            st.sidebar.success("✅ Sync data imported.")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            st.sidebar.error(f"Import failed: {exc}")

    # ── Main layout ───────────────────────────────────────────────────────
    st.title("🎸 SyncopateAI – Guitar Tab Sync")
    st.markdown(
        "Upload your recording and paste your ASCII tab, then click "
        "**Analyse & Sync** to build the interactive synchronised view."
    )

    col_audio, col_tab = st.columns([1, 2])

    with col_audio:
        st.subheader("🎵 Audio File")
        audio_upload = st.file_uploader(
            "Upload WAV or MP3",
            type=["wav", "mp3", "flac", "ogg"],
            help=(
                "File is saved to static/temp/ and served directly by Streamlit "
                "(no Base64 encoding)."
            ),
        )
        if audio_upload and audio_upload.name:
            st.info(f"Uploaded: **{audio_upload.name}**")

    with col_tab:
        st.subheader("🎸 Guitar Tab")
        raw_tab = st.text_area(
            "Paste ASCII tab",
            height=220,
            placeholder=(
                "e|--0-------<12>-------[12]-|\n"
                "B|--1--------12---------1---|\n"
                "G|--0---------0---------0---|\n"
                "D|--2---------2---------2---|\n"
                "A|--0---------0---------0---|\n"
                "E|--x---------x---------x---|"
            ),
        )

    # ── Analyse & Sync ────────────────────────────────────────────────────
    if st.button("🔬 Analyse & Sync", type="primary"):
        if not raw_tab.strip():
            st.warning("⚠️ Please paste a guitar tab before analysing.")
        elif audio_upload is None:
            st.warning("⚠️ Please upload an audio file before analysing.")
        else:
            with st.spinner("Saving audio file …"):
                _, audio_url = _save_audio(audio_upload)
                ss["audio_url"] = audio_url

            with st.spinner("Parsing tab …"):
                parser = TabParser()
                ss["tab_slices"] = parser.parse(raw_tab)

            with st.spinner("Extracting onsets …"):
                syncer = AudioSyncer(
                    wait=wait_frames,
                    pre_max=pre_max,
                    threshold_ignore_sec=threshold_ignore_sec,
                )
                try:
                    saved_path = str(
                        STATIC_TEMP_DIR
                        / audio_url.split("/")[-1]
                    )
                    ss["onset_times"] = syncer.extract_onsets(saved_path)
                except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                    st.error(f"Onset extraction failed: {exc}")
                    ss["onset_times"] = []

            with st.spinner("Building onset-to-slice mapping …"):
                ss["sync_mapping"] = build_onset_to_slice_map(
                    ss["onset_times"],
                    ss["tab_slices"],
                    threshold_ignore_sec=threshold_ignore_sec,
                )

            # Estimate duration from onsets (fallback if librosa metadata
            # is not directly available here)
            if ss["onset_times"]:
                ss["total_duration"] = ss["onset_times"][-1] + 2.0
            else:
                ss["total_duration"] = 10.0

            st.success(
                f"✅  {len(ss['tab_slices'])} tab slices · "
                f"{len(ss['onset_times'])} onsets detected · "
                f"{len(ss['sync_mapping'])} mapping entries"
            )

    st.divider()

    # ── Export sync data (Python-side, uses session_state mapping) ────────
    col_ex, col_info = st.columns([1, 3])
    with col_ex:
        if ss["sync_mapping"] or ss["tab_slices"]:
            export_payload = json.dumps(
                {
                    "onsets":         ss["onset_times"],
                    "slices":         ss["tab_slices"],
                    "mapping":        ss["sync_mapping"],
                    "total_duration": ss["total_duration"],
                },
                indent=2,
            ).encode("utf-8")
            st.download_button(
                "📥 Export Sync Data (JSON)",
                data=export_payload,
                file_name="syncopate_sync.json",
                mime="application/json",
            )
    with col_info:
        if ss["sync_mapping"]:
            st.caption(
                f"Mapping: {len(ss['sync_mapping'])} onsets across "
                f"{len(ss['tab_slices'])} tab slices.  "
                f"Duration: {ss['total_duration']:.2f} s"
            )

    # ── Interactive HTML component ─────────────────────────────────────────
    if ss["tab_slices"]:
        st.subheader("🎼 Interactive Tab Sync View")
        st.markdown(
            "Use **▶ Play** to start playback.  "
            "Drag **onset markers** (red dots) to re-snap them to a different "
            "tab slice.  Click **📥 Export JSON** inside the player to save "
            "your adjusted mapping."
        )
        html_src = _make_html_component(
            slices=ss["tab_slices"],
            onsets=ss["onset_times"],
            mapping=ss["sync_mapping"],
            audio_url=ss["audio_url"],
            total_duration=ss["total_duration"],
        )
        components.html(html_src, height=component_height, scrolling=False)
    else:
        st.info(
            "⬆️  Paste a tab, upload audio, and click **Analyse & Sync** to "
            "see the interactive view here."
        )

    # ── Debug / raw data expander ─────────────────────────────────────────
    if ss["tab_slices"]:
        with st.expander("🔍 Raw parsed tab slices"):
            st.json(ss["tab_slices"])
        if ss["sync_mapping"]:
            with st.expander("🔗 Onset-to-slice mapping"):
                st.json(ss["sync_mapping"])


if __name__ == "__main__":
    main()

