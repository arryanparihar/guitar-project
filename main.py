"""
main.py
=======
SyncopateAI – Interactive Guitar Tab Player dashboard.

Run with:
    streamlit run main.py

Features
--------
* Paste a raw ASCII guitar tab and optionally upload a WAV/MP3 audio file.
* AudioSyncer maps detected audio onsets onto TabParser time slices 1-to-1.
* Sidebar controls: Mode toggle, Global Offset, Playback Speed, Look-Ahead.
* Embedded HTML/JS/CSS Guitar Tab Player:
    - Dark-mode, Courier New monospace UI.
    - Horizontal-scrolling flex container with 6-string vertical alignment.
    - requestAnimationFrame playback loop with auto-scrollIntoView.
    - A-B looping via Shift+click on two tab slices.
    - Sync Refinement Mode with draggable green-dot onset handles.
* Backward-compatible ``_build_finger_height_csv`` helper (used by tests).
"""

from __future__ import annotations

import base64
import csv
import io
import json
import re
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from audio_engine import AudioAnalysisResult
from audio_sync import AudioSyncer, SyncedSlice
from tab_engine import TabParser
from vision_engine import FrameResult

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
# Regex shared with _parse_tab_for_display (mirrors tab_engine._STRING_LINE_RE)
# ---------------------------------------------------------------------------

_DISPLAY_LINE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z][#b]?)\|(?P<content>.*)$"
)

# ---------------------------------------------------------------------------
# Helper: backward-compatible CSV builder (tests depend on this signature)
# ---------------------------------------------------------------------------


def _build_finger_height_csv(
    results: list[FrameResult],
    audio_result: Optional[AudioAnalysisResult] = None,
) -> bytes:
    """Convert vision analysis results to a UTF-8 encoded CSV for download.

    Columns
    -------
    frame_index, timestamp_sec, avg_finger_height_px, efficiency_score,
    index_x, index_y, middle_x, middle_y, ring_x, ring_y, pinky_x, pinky_y
    [, Audio_Deviation_ms]  – only present when *audio_result* is supplied
                              and contains at least one onset event.

    For each video frame the ``Audio_Deviation_ms`` value is taken from the
    audio onset whose timestamp is closest to the frame's timestamp.
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

    onset_times: list[float] = (
        [e.time_sec for e in audio_result.onset_events] if include_audio else []
    )

    for r in results:
        tip_coords: dict[str, tuple] = {
            ft.name: (ft.x, ft.y) for ft in r.fingertips
        }
        row = [
            r.frame_index,
            round(r.timestamp_sec, 6),
            round(r.avg_finger_height, 4) if r.avg_finger_height is not None else "",
            round(r.efficiency_score, 4) if r.efficiency_score is not None else "",
        ]
        for name in fingertip_names:
            if name in tip_coords:
                row += [
                    round(tip_coords[name][0], 2),
                    round(tip_coords[name][1], 2),
                ]
            else:
                row += ["", ""]

        if include_audio:
            closest_idx = min(
                range(len(onset_times)),
                key=lambda i: abs(onset_times[i] - r.timestamp_sec),
            )
            row.append(
                round(audio_result.onset_events[closest_idx].deviation_ms, 4)
            )

        writer.writerow(row)

    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Tab display parser  (for embedding in the JS player)
# ---------------------------------------------------------------------------


def _parse_tab_for_display(raw_tab: str) -> list[list[dict[str, str]]]:
    """Parse raw tab text into blocks suitable for JavaScript rendering.

    Returns
    -------
    List of blocks, where each block is a list of row dicts::

        {"label": "e|", "content": "--0-------<12>--|"}

    Only lines that contain at least one ``-`` in the content part (after
    the ``|``) are included; all other lines are silently ignored.
    """
    lines = raw_tab.splitlines()
    blocks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []

    for line in lines:
        m = _DISPLAY_LINE_RE.match(line)
        if m and "-" in m.group("content"):
            current.append(
                {
                    "label": m.group("name") + "|",
                    "content": m.group("content"),
                }
            )
        else:
            if current:
                blocks.append(current)
                current = []

    if current:
        blocks.append(current)

    return blocks


# ---------------------------------------------------------------------------
# Embedded CSS for the player
# ---------------------------------------------------------------------------

_PLAYER_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #121212;
    color: #e0e0e0;
    font-family: 'Courier New', Courier, monospace;
    font-size: 14px;
    line-height: 1.5;
    overflow-x: hidden;
}
#player-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 10px;
}
/* ── Controls bar ─────────────────────────────────────────── */
#controls {
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}
#play-btn, #stop-btn {
    border: none;
    border-radius: 6px;
    padding: 7px 18px;
    font-size: 15px;
    cursor: pointer;
    color: #fff;
    font-family: inherit;
}
#play-btn            { background: #388e3c; }
#play-btn:hover      { background: #4caf50; }
#play-btn.paused     { background: #1565c0; }
#play-btn.paused:hover { background: #1976d2; }
#stop-btn            { background: #424242; }
#stop-btn:hover      { background: #616161; }
#progress-bar-wrap {
    flex: 1;
    min-width: 80px;
    height: 8px;
    background: #333;
    border-radius: 4px;
    cursor: pointer;
    position: relative;
}
#progress-bar {
    height: 100%;
    background: #4caf50;
    border-radius: 4px;
    width: 0%;
    pointer-events: none;
}
#ab-marker-a, #ab-marker-b {
    position: absolute;
    top: -3px;
    width: 2px;
    height: 14px;
    display: none;
    pointer-events: none;
}
#ab-marker-a { background: #ff9800; }
#ab-marker-b { background: #f44336; }
#time-display {
    color: #888;
    font-size: 12px;
    min-width: 90px;
    white-space: nowrap;
}
#ab-info {
    font-size: 11px;
    color: #ff9800;
    min-width: 170px;
}
#clear-ab-btn {
    background: transparent;
    border: 1px solid #ff9800;
    color: #ff9800;
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 11px;
    cursor: pointer;
    display: none;
    font-family: inherit;
}
#clear-ab-btn.visible { display: inline-block; }
#mode-badge {
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid #388e3c;
    color: #4caf50;
    white-space: nowrap;
}
#mode-badge.refinement {
    color: #ff9800;
    border-color: #ff9800;
}
/* ── Tab display area ─────────────────────────────────────── */
#tab-display {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 14px 14px 8px;
    overflow-x: auto;
}
.tab-system {
    display: inline-block;
    min-width: max-content;
}
.tab-row {
    display: flex;
    align-items: center;
    height: 1.6em;
    white-space: nowrap;
}
.string-label {
    color: #666;
    user-select: none;
    min-width: 2ch;
    padding-right: 2px;
    flex-shrink: 0;
}
.tab-char {
    display: inline-block;
    width: 1ch;
    text-align: center;
    cursor: default;
    color: #ccc;
    border-radius: 2px;
}
.tab-char.active         { background: #2e7d32; color: #fff; }
.tab-char.ab-start       { background: #3e2800; border-left:  2px solid #ff9800; }
.tab-char.ab-end         { background: #3e2800; border-right: 2px solid #f44336; }
.tab-char.ab-region      { background: #3e2800; }
.tab-char:hover          { background: #2a2a2a; }
body.shift-mode .tab-char           { cursor: crosshair; }
body.shift-mode .tab-char:hover     { background: #3e3a00; }
/* ── Refinement handles ───────────────────────────────────── */
.handle-row {
    position: relative;
    height: 22px;
    margin-top: 2px;
}
.onset-handle {
    position: absolute;
    width: 12px;
    height: 12px;
    background: #4caf50;
    border: 2px solid #2e7d32;
    border-radius: 50%;
    top: 5px;
    cursor: grab;
    z-index: 20;
    transition: transform 0.1s, background 0.1s;
}
.onset-handle:hover      { transform: scale(1.4); background: #81c784; }
.onset-handle.dragging   { cursor: grabbing; transform: scale(1.5); background: #a5d6a7; }
/* ── System separator ─────────────────────────────────────── */
.system-sep {
    display: flex;
    align-items: center;
    color: #444;
    font-size: 11px;
    padding: 6px 0;
    gap: 6px;
}
.system-sep::after {
    content: '';
    flex: 1;
    border-bottom: 1px dashed #2a2a2a;
}
.player-msg      { color: #666; font-size: 12px; font-style: italic; text-align: center; padding: 20px; }
.player-msg.warn { color: #ff9800; }
"""

# ---------------------------------------------------------------------------
# Embedded JS for the player
# ---------------------------------------------------------------------------

_PLAYER_JS = r"""
// ============================================================
// Config (JSON embedded in a <script type="application/json">)
// ============================================================
const CFG            = JSON.parse(document.getElementById('player-config').textContent);
const TAB_BLOCKS     = CFG.tabBlocks;
const INIT_SLICES    = CFG.syncedSlices;
const AUDIO_SRC      = CFG.audioSrc;
const MODE           = CFG.mode;
const PLAYBACK_SPEED = CFG.playbackSpeed;
const LOOK_AHEAD_MS  = CFG.lookAheadMs;

// ============================================================
// Mutable state
// ============================================================
let mapping       = INIT_SLICES.map(s => Object.assign({}, s));
let currentSliceI = -1;
let isPlaying     = false;
let rafId         = null;
let shiftDown     = false;
const ab = { startSec: null, endSec: null, step: 0 };
const audio = new Audio();

// ============================================================
// Audio setup
// ============================================================
function setupAudio() {
    if (!AUDIO_SRC) return;
    audio.src          = AUDIO_SRC;
    audio.playbackRate = PLAYBACK_SPEED;
    audio.preload      = 'auto';
    audio.addEventListener('ended', stopPlayback);
}

// ============================================================
// Build tab DOM
// ============================================================
function buildTabDisplay() {
    const container = document.getElementById('tab-display');
    if (!TAB_BLOCKS || TAB_BLOCKS.length === 0) {
        container.innerHTML = '<p class="player-msg">No valid tab content detected.</p>';
        return;
    }
    TAB_BLOCKS.forEach((block, blockIdx) => {
        const sys = document.createElement('div');
        sys.className = 'tab-system';
        sys.dataset.system = blockIdx;

        const wrap = document.createElement('div');
        wrap.className = 'rows-wrapper';

        block.forEach(row => {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'tab-row';

            const lbl = document.createElement('span');
            lbl.className = 'string-label';
            lbl.textContent = row.label;
            rowDiv.appendChild(lbl);

            for (let col = 0; col < row.content.length; col++) {
                const sp = document.createElement('span');
                sp.className    = 'tab-char';
                sp.dataset.col  = col;
                sp.dataset.block = blockIdx;
                sp.textContent  = row.content[col] || ' ';
                rowDiv.appendChild(sp);
            }
            wrap.appendChild(rowDiv);
        });
        sys.appendChild(wrap);

        if (MODE === 'refinement') {
            const hr = document.createElement('div');
            hr.className = 'handle-row';
            hr.dataset.system = blockIdx;
            sys.appendChild(hr);
        }
        container.appendChild(sys);

        if (blockIdx < TAB_BLOCKS.length - 1) {
            const sep = document.createElement('div');
            sep.className = 'system-sep';
            sep.textContent = '\u00a7 system ' + (blockIdx + 2);
            container.appendChild(sep);
        }
    });

    // Shift+click for A-B loop
    document.addEventListener('keydown', e => {
        if (e.key === 'Shift') { shiftDown = true; document.body.classList.add('shift-mode'); }
    });
    document.addEventListener('keyup', e => {
        if (e.key === 'Shift') { shiftDown = false; document.body.classList.remove('shift-mode'); }
    });
    container.addEventListener('click', e => {
        if (!shiftDown) return;
        const sp = e.target.closest('.tab-char');
        if (sp) handleAbClick(parseInt(sp.dataset.col));
    });

    if (MODE === 'refinement') {
        requestAnimationFrame(() => requestAnimationFrame(buildHandles));
    }
}

// ============================================================
// A-B Loop
// ============================================================
function onsetTimeForCol(col) {
    for (const s of mapping) {
        if (s.time_index === col && s.onset_time_sec !== null) return s.onset_time_sec;
    }
    return null;
}

function handleAbClick(col) {
    const t = onsetTimeForCol(col);
    if (t === null) return;
    if (ab.step === 0 || ab.step === 2) {
        ab.startSec = t; ab.endSec = null; ab.step = 1;
        document.getElementById('ab-info').textContent =
            'A = ' + t.toFixed(2) + 's \u2014 Shift+click B';
        updateAbVisuals();
    } else {
        if (t > ab.startSec) { ab.endSec = t; }
        else { ab.endSec = ab.startSec; ab.startSec = t; }
        ab.step = 2;
        document.getElementById('ab-info').textContent =
            'Loop ' + ab.startSec.toFixed(2) + 's \u2192 ' + ab.endSec.toFixed(2) + 's';
        document.getElementById('clear-ab-btn').classList.add('visible');
        updateAbVisuals();
    }
}

function clearAbLoop() {
    ab.startSec = null; ab.endSec = null; ab.step = 0;
    document.getElementById('ab-info').textContent =
        AUDIO_SRC ? 'Shift+click two slices to set A-B loop' : 'Upload audio to enable playback';
    document.getElementById('clear-ab-btn').classList.remove('visible');
    updateAbVisuals();
}

function closestSlice(targetSec) {
    let best = null;
    for (const s of mapping) {
        if (s.onset_time_sec === null) continue;
        if (best === null ||
            Math.abs(s.onset_time_sec - targetSec) < Math.abs(best.onset_time_sec - targetSec)) {
            best = s;
        }
    }
    return best;
}

function updateAbVisuals() {
    document.querySelectorAll('.tab-char').forEach(el =>
        el.classList.remove('ab-start', 'ab-end', 'ab-region'));

    if (ab.startSec === null) { updateProgressAbMarkers(); return; }

    const sSlice = closestSlice(ab.startSec);
    const eSlice = ab.endSec !== null ? closestSlice(ab.endSec) : null;

    if (sSlice) {
        document.querySelectorAll('.tab-char[data-col="' + sSlice.time_index + '"]')
            .forEach(el => el.classList.add('ab-start'));
    }
    if (eSlice) {
        document.querySelectorAll('.tab-char[data-col="' + eSlice.time_index + '"]')
            .forEach(el => el.classList.add('ab-end'));
        const lo = Math.min(sSlice.slice_index, eSlice.slice_index) + 1;
        const hi = Math.max(sSlice.slice_index, eSlice.slice_index) - 1;
        for (let i = lo; i <= hi; i++) {
            if (i >= 0 && i < mapping.length) {
                document.querySelectorAll('.tab-char[data-col="' + mapping[i].time_index + '"]')
                    .forEach(el => el.classList.add('ab-region'));
            }
        }
    }
    updateProgressAbMarkers();
}

function updateProgressAbMarkers() {
    const dur = audio.duration || 0;
    const mA = document.getElementById('ab-marker-a');
    const mB = document.getElementById('ab-marker-b');
    if (!dur || ab.startSec === null) {
        mA.style.display = 'none'; mB.style.display = 'none'; return;
    }
    mA.style.display = 'block';
    mA.style.left    = (ab.startSec / dur * 100) + '%';
    if (ab.endSec !== null) {
        mB.style.display = 'block';
        mB.style.left    = (ab.endSec / dur * 100) + '%';
    } else {
        mB.style.display = 'none';
    }
}

// ============================================================
// requestAnimationFrame playback loop
// ============================================================
function playbackLoop() {
    if (!isPlaying) return;

    const nowSec      = audio.currentTime;
    const effectiveMs = nowSec * 1000 + LOOK_AHEAD_MS;

    // A-B loop enforcement
    if (ab.step === 2 && ab.endSec !== null && nowSec >= ab.endSec) {
        audio.currentTime = ab.startSec || 0;
    }

    // Find last slice whose onset <= effectiveTime
    let newI = -1;
    for (let i = 0; i < mapping.length; i++) {
        const s = mapping[i];
        if (s.onset_time_sec !== null && s.onset_time_sec * 1000 <= effectiveMs) newI = i;
    }

    if (newI !== currentSliceI) {
        if (currentSliceI >= 0) {
            document.querySelectorAll('.tab-char[data-col="' + mapping[currentSliceI].time_index + '"]')
                .forEach(el => el.classList.remove('active'));
        }
        currentSliceI = newI;
        if (currentSliceI >= 0) {
            const col = mapping[currentSliceI].time_index;
            const els = document.querySelectorAll('.tab-char[data-col="' + col + '"]');
            els.forEach(el => el.classList.add('active'));
            if (els.length > 0) {
                els[0].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
            }
        }
    }

    updateTimeDisplay();
    rafId = requestAnimationFrame(playbackLoop);
}

// ============================================================
// Controls
// ============================================================
function startPlayback() {
    if (!AUDIO_SRC) {
        document.getElementById('ab-info').textContent =
            '\u26a0 No audio loaded \u2014 upload a WAV/MP3 file.';
        return;
    }
    audio.play().then(() => {
        isPlaying = true;
        const btn = document.getElementById('play-btn');
        btn.textContent = '\u23f8 Pause';
        btn.classList.add('paused');
        rafId = requestAnimationFrame(playbackLoop);
    }).catch(err => {
        document.getElementById('ab-info').textContent =
            '\u26a0 Playback error: ' + err.message;
    });
}

function pausePlayback() {
    audio.pause();
    isPlaying = false;
    const btn = document.getElementById('play-btn');
    btn.textContent = '\u25b6 Play';
    btn.classList.remove('paused');
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
}

function stopPlayback() {
    pausePlayback();
    audio.currentTime = 0;
    currentSliceI = -1;
    document.querySelectorAll('.tab-char.active').forEach(el => el.classList.remove('active'));
    updateTimeDisplay();
}

function togglePlay() { if (isPlaying) pausePlayback(); else startPlayback(); }

function updateTimeDisplay() {
    const cur = audio.currentTime || 0;
    const dur = audio.duration    || 0;
    const pct = dur > 0 ? (cur / dur * 100) : 0;
    document.getElementById('time-display').textContent =
        cur.toFixed(1) + 's / ' + dur.toFixed(1) + 's';
    document.getElementById('progress-bar').style.width = pct + '%';
}

function onProgressClick(e) {
    if (!audio.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct  = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audio.currentTime = pct * audio.duration;
    updateTimeDisplay();
}

// ============================================================
// Sync Refinement Mode – draggable onset handles
// ============================================================
let dragState = null;

function buildHandles() {
    mapping.forEach(slice => {
        if (slice.onset_time_sec === null) return;

        const refSpan = document.querySelector('.tab-char[data-col="' + slice.time_index + '"]');
        if (!refSpan) return;

        const sys = refSpan.closest('.tab-system');
        if (!sys) return;

        const handleRow = sys.querySelector('.handle-row');
        if (!handleRow) return;

        const handle = document.createElement('div');
        handle.className = 'onset-handle';
        handle.dataset.sliceIndex = slice.slice_index;
        handle.title = 'Onset ' + slice.slice_index + ': t=' +
                       slice.onset_time_sec.toFixed(3) + 's\nDrag to remap';

        positionHandle(handle, refSpan, handleRow);
        setupHandleDrag(handle, slice, handleRow);
        handleRow.appendChild(handle);
    });
}

function positionHandle(handle, refSpan, handleRow) {
    const sr = refSpan.getBoundingClientRect();
    const rr = handleRow.getBoundingClientRect();
    handle.style.left = Math.max(0, sr.left - rr.left + sr.width / 2 - 6) + 'px';
}

function setupHandleDrag(handle, slice, handleRow) {
    handle.addEventListener('pointerdown', e => {
        e.preventDefault();
        handle.setPointerCapture(e.pointerId);
        handle.classList.add('dragging');
        dragState = { handle, slice, handleRow };
    });

    handle.addEventListener('pointermove', e => {
        if (!dragState || dragState.slice !== slice) return;

        // Temporarily hide so elementFromPoint looks through it
        handle.style.visibility = 'hidden';
        const under = document.elementFromPoint(e.clientX, e.clientY);
        handle.style.visibility = '';

        if (under && under.classList.contains('tab-char')) {
            const newCol = parseInt(under.dataset.col);
            const oldCol = slice.time_index;
            if (newCol !== oldCol) {
                // Remove active from old col if this was the current slice
                if (currentSliceI === slice.slice_index) {
                    document.querySelectorAll('.tab-char[data-col="' + oldCol + '"]')
                        .forEach(el => el.classList.remove('active'));
                }
                slice.time_index = newCol;
                // Immediate visual feedback
                if (currentSliceI === slice.slice_index) {
                    document.querySelectorAll('.tab-char[data-col="' + newCol + '"]')
                        .forEach(el => el.classList.add('active'));
                }
                positionHandle(handle, under, handleRow);
            }
        }
    });

    handle.addEventListener('pointerup', () => {
        if (dragState && dragState.slice === slice) {
            handle.classList.remove('dragging');
            dragState = null;
        }
    });
}

// ============================================================
// Init
// ============================================================
function init() {
    setupAudio();
    buildTabDisplay();

    document.getElementById('play-btn').addEventListener('click', togglePlay);
    document.getElementById('stop-btn').addEventListener('click', stopPlayback);
    document.getElementById('clear-ab-btn').addEventListener('click', clearAbLoop);
    document.getElementById('progress-bar-wrap').addEventListener('click', onProgressClick);

    const badge = document.getElementById('mode-badge');
    if (MODE === 'refinement') {
        badge.textContent = '\u2699 Sync Refinement Mode';
        badge.classList.add('refinement');
    } else {
        badge.textContent = '\u25b6 Playback Mode';
    }

    document.getElementById('ab-info').textContent =
        AUDIO_SRC ? 'Shift+click two slices to set A-B loop' : 'Upload audio to enable playback';
    updateTimeDisplay();
}

window.addEventListener('DOMContentLoaded', init);
"""


def _build_player_html(
    tab_blocks: list[list[dict[str, str]]],
    synced_slices: list[SyncedSlice],
    audio_src: str,
    *,
    mode: str,
    playback_speed: float,
    look_ahead_ms: int,
) -> str:
    """Build the complete self-contained HTML/JS/CSS Guitar Tab Player.

    Parameters
    ----------
    tab_blocks:
        Parsed tab blocks from :func:`_parse_tab_for_display`.
    synced_slices:
        Onset-to-slice mapping from :class:`AudioSyncer`.
    audio_src:
        A data-URI for the audio (``"data:audio/...;base64,..."``), or an
        empty string when no audio has been uploaded.
    mode:
        ``"playback"`` or ``"refinement"``.
    playback_speed:
        Audio playback rate (0.5 – 1.0).
    look_ahead_ms:
        Milliseconds added to the current playback position before finding
        the active tab slice (the highlight fires slightly early).
    """
    config: dict = {
        "tabBlocks": tab_blocks,
        "syncedSlices": [
            {
                "slice_index": s.slice_index,
                "time_index": s.time_index,
                "onset_time_sec": s.onset_time_sec,
            }
            for s in synced_slices
        ],
        "audioSrc": audio_src,
        "mode": mode,
        "playbackSpeed": playback_speed,
        "lookAheadMs": look_ahead_ms,
    }
    config_json: str = json.dumps(config, ensure_ascii=True)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<style>{_PLAYER_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '<div id="player-container">\n'
        '  <div id="controls">\n'
        '    <button id="play-btn">\u25b6 Play</button>\n'
        '    <button id="stop-btn">\u25a0 Stop</button>\n'
        '    <div id="progress-bar-wrap" title="Click to seek">\n'
        '      <div id="progress-bar"></div>\n'
        '      <div id="ab-marker-a"></div>\n'
        '      <div id="ab-marker-b"></div>\n'
        "    </div>\n"
        '    <span id="time-display">0.0s / 0.0s</span>\n'
        '    <span id="ab-info"></span>\n'
        '    <button id="clear-ab-btn">\u2715 Clear A-B</button>\n'
        '    <span id="mode-badge"></span>\n'
        "  </div>\n"
        '  <div id="tab-display"></div>\n'
        "</div>\n"
        '<script type="application/json" id="player-config">'
        f"{config_json}"
        "</script>\n"
        f"<script>\n{_PLAYER_JS}\n</script>\n"
        "</body>\n"
        "</html>"
    )


# ---------------------------------------------------------------------------
# Main Streamlit dashboard
# ---------------------------------------------------------------------------

_SAMPLE_TAB = (
    "e|--0-------<12>-------[12]-|\n"
    "B|--1--------12---------1---|\n"
    "G|--0---------0---------0---|\n"
    "D|--2---------2---------2---|\n"
    "A|--0---------0---------0---|\n"
    "E|--x---------x---------x---|"
)

#: Warn (but do not block) when audio exceeds this size.
_AUDIO_WARN_MB: float = 10.0
#: Hard limit – mirrors AudioSyncer._MAX_FILE_BYTES.
_AUDIO_MAX_MB: float = 50.0


def main() -> None:
    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎸 Controls")
        st.divider()

        mode: str = st.radio(
            "Mode",
            options=["Playback Mode", "Sync Refinement Mode"],
            index=0,
            help=(
                "**Playback Mode** – follow the highlighted tab in real time.\n\n"
                "**Sync Refinement Mode** – drag the green dot handles to "
                "re-link audio onsets to different tab columns."
            ),
        )
        mode_key = "playback" if mode == "Playback Mode" else "refinement"

        st.divider()

        global_offset_ms: int = st.slider(
            "Global Offset (ms)",
            min_value=-500,
            max_value=500,
            value=0,
            step=5,
            help=(
                "Shift every onset timestamp by this amount. "
                "Positive = later; negative = earlier."
            ),
        )
        playback_speed: float = st.slider(
            "Playback Speed",
            min_value=0.5,
            max_value=1.0,
            value=1.0,
            step=0.05,
            format="%.2fx",
            help="Controls `audio.playbackRate` in the player.",
        )
        look_ahead_ms: int = st.slider(
            "Look-Ahead (ms)",
            min_value=0,
            max_value=500,
            value=0,
            step=10,
            help=(
                "Highlight each tab slice this many milliseconds *before* "
                "its audio onset fires."
            ),
        )

    # ── Main layout ──────────────────────────────────────────────────────────
    st.title("🎸 SyncopateAI – Guitar Tab Player")
    st.caption(
        "Paste a guitar tab, upload an audio file, then press **▶ Play**. "
        "Hold **Shift** and click two tab slices to set an A-B loop region."
    )

    col_tab, col_audio = st.columns([3, 2], gap="medium")

    with col_tab:
        raw_tab: str = st.text_area(
            "Guitar Tab",
            value=_SAMPLE_TAB,
            height=220,
            placeholder=_SAMPLE_TAB,
            help=(
                "Standard ASCII guitar tab. Lines must follow "
                "`STRING|content`, e.g. `e|--0--5--|`."
            ),
        )

    with col_audio:
        uploaded_file = st.file_uploader(
            "Audio File (WAV / MP3)",
            type=["wav", "mp3"],
            help=f"Upload the audio for this tab. Maximum {int(_AUDIO_MAX_MB)} MB.",
        )

    st.divider()

    # ── Parse tab ────────────────────────────────────────────────────────────
    parser = TabParser()
    tab_slices: list = []
    tab_blocks: list = []
    if raw_tab.strip():
        tab_slices = parser.parse(raw_tab)
        tab_blocks = _parse_tab_for_display(raw_tab)
        if not tab_slices:
            st.warning(
                "⚠️ No valid tab string lines detected. "
                "Ensure lines follow the `STRING|content` format."
            )

    # ── Process audio ────────────────────────────────────────────────────────
    onset_times: list[float] = []
    audio_src: str = ""

    if uploaded_file is not None:
        audio_bytes: bytes = uploaded_file.read()
        size_mb: float = len(audio_bytes) / 1_048_576

        if size_mb > _AUDIO_MAX_MB:
            st.error(
                f"⛔ Audio file too large ({size_mb:.1f} MB). "
                f"Maximum is {int(_AUDIO_MAX_MB)} MB."
            )
        else:
            if size_mb > _AUDIO_WARN_MB:
                st.warning(
                    f"⚠️ Audio file is {size_mb:.1f} MB – large files may "
                    "take a moment to load in the player."
                )

            suffix = (
                ".mp3" if uploaded_file.name.lower().endswith(".mp3") else ".wav"
            )
            try:
                syncer = AudioSyncer()
                onset_times = syncer.analyse_bytes(audio_bytes, suffix=suffix)
                st.success(
                    f"✅ **{len(onset_times)}** onsets detected in "
                    f"**{uploaded_file.name}** ({size_mb:.2f} MB)."
                )
            except ValueError as exc:
                st.error(f"⛔ Audio analysis failed: {exc}")
            except OSError as exc:
                st.error(f"⛔ Could not read audio file: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"⛔ Unexpected audio error: {exc}")

            # Encode as base64 data-URI for the embedded player
            b64 = base64.b64encode(audio_bytes).decode("ascii")
            mime = "audio/mpeg" if suffix == ".mp3" else "audio/wav"
            audio_src = f"data:{mime};base64,{b64}"

    # ── Map onsets → slices ──────────────────────────────────────────────────
    synced_slices: list[SyncedSlice] = AudioSyncer.map_to_slices(
        onset_times,
        tab_slices,
        global_offset_ms=float(global_offset_ms),
    )

    # ── Info metrics ─────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Tab slices", len(tab_slices))
    m2.metric("Onsets detected", len(onset_times))
    m3.metric(
        "Linked pairs",
        sum(1 for s in synced_slices if s.onset_time_sec is not None),
    )

    # ── Guitar Tab Player (HTML/JS/CSS) ───────────────────────────────────────
    player_html = _build_player_html(
        tab_blocks,
        synced_slices,
        audio_src,
        mode=mode_key,
        playback_speed=playback_speed,
        look_ahead_ms=look_ahead_ms,
    )
    components.html(player_html, height=560, scrolling=False)

    # ── Expandable details ────────────────────────────────────────────────────
    with st.expander("🔍 Parsed Tab Data (JSON)", expanded=False):
        st.json(tab_slices)

    with st.expander("🔗 Onset → Slice Mapping", expanded=False):
        if synced_slices:
            st.dataframe(
                [
                    {
                        "Slice #": s.slice_index,
                        "Column (time_index)": s.time_index,
                        "Onset (s)": (
                            round(s.onset_time_sec, 4)
                            if s.onset_time_sec is not None
                            else "—"
                        ),
                    }
                    for s in synced_slices
                ],
                use_container_width=True,
            )
        else:
            st.info("No slices parsed yet.")


if __name__ == "__main__":
    main()
