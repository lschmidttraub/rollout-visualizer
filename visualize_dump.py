#!/usr/bin/env python3
"""
Rollout Visualizer: Displays rollout text with tokens colored by teacher/student logprobs.

Usage:
    # Using the ML environment on Clariden:
    uenv run pytorch/v2.8.0:v1 --view=default -- bash -c \
        'source /iopsstor/scratch/cscs/lschmidttraub/venvs/llm-dev/bin/activate && \
         python3 visualize_rollout.py <path_to_jsonl> [-o output.html] [--record N]'

    # The script generates a self-contained HTML file that can be opened in any browser.
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def load_data(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_token_data(tokenizer, records: list[dict]) -> list[dict]:
    """Pre-tokenize all records and build structured data for the HTML."""
    has_topk = "student_response_topk_under_teacher" in records[0]

    all_records = []
    for idx, rec in enumerate(records):
        print(f"  Tokenizing record {idx + 1}/{len(records)}...", file=sys.stderr)
        entry = {
            "step": rec.get("step", "?"),
            "batch_idx": rec.get("batch_idx", "?"),
            "student_prompt": rec.get("student_prompt", ""),
            "teacher_prompt": rec.get("teacher_prompt", ""),
            "has_topk": has_topk,
        }

        # Student response tokens
        student_ids = tokenizer.encode(
            rec["student_response"], add_special_tokens=False
        )
        student_lp_student = rec["student_response_logprobs_under_student"]
        student_lp_teacher = rec["student_response_logprobs_under_teacher"]
        n_s = min(len(student_ids), len(student_lp_student), len(student_lp_teacher))
        student_tokens = []
        student_topk_teacher = (
            rec.get("student_response_topk_under_teacher") if has_topk else None
        )
        student_topk_student = (
            rec.get("student_response_topk_under_student") if has_topk else None
        )
        for i in range(n_s):
            tok = {
                "text": tokenizer.decode([student_ids[i]]),
                "lp_student": student_lp_student[i],
                "lp_teacher": student_lp_teacher[i],
            }
            if student_topk_teacher and i < len(student_topk_teacher):
                tok["topk_teacher"] = student_topk_teacher[i]
            if student_topk_student and i < len(student_topk_student):
                tok["topk_student"] = student_topk_student[i]
            student_tokens.append(tok)
        entry["student_tokens"] = student_tokens

        # Teacher response tokens
        teacher_ids = tokenizer.encode(
            rec["teacher_response"], add_special_tokens=False
        )
        teacher_lp_teacher = rec["teacher_response_logprobs_under_teacher"]
        teacher_lp_student = rec["teacher_response_logprobs_under_student"]
        n_t = min(len(teacher_ids), len(teacher_lp_teacher), len(teacher_lp_student))
        teacher_tokens = []
        teacher_topk_teacher = (
            rec.get("teacher_response_topk_under_teacher") if has_topk else None
        )
        teacher_topk_student = (
            rec.get("teacher_response_topk_under_student") if has_topk else None
        )
        for i in range(n_t):
            tok = {
                "text": tokenizer.decode([teacher_ids[i]]),
                "lp_teacher": teacher_lp_teacher[i],
                "lp_student": teacher_lp_student[i],
            }
            if teacher_topk_teacher and i < len(teacher_topk_teacher):
                tok["topk_teacher"] = teacher_topk_teacher[i]
            if teacher_topk_student and i < len(teacher_topk_student):
                tok["topk_student"] = teacher_topk_student[i]
            teacher_tokens.append(tok)
        entry["teacher_tokens"] = teacher_tokens

        all_records.append(entry)

    return all_records


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rollout Visualizer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: #f6f8fa;
    color: #1f2328;
    display: flex;
    height: 100vh;
    overflow: hidden;
}

/* Sidebar */
.sidebar {
    width: 260px;
    min-width: 260px;
    background: #ffffff;
    border-right: 1px solid #d1d9e0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.sidebar-header {
    padding: 16px;
    border-bottom: 1px solid #d1d9e0;
    font-size: 14px;
    font-weight: 600;
    color: #0969da;
}

.sidebar-info {
    padding: 8px 16px;
    border-bottom: 1px solid #d1d9e0;
    font-size: 11px;
    color: #656d76;
}

.record-list {
    overflow-y: auto;
    flex: 1;
}

.record-item {
    padding: 10px 16px;
    border-bottom: 1px solid #eaeef2;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.15s;
}

.record-item:hover { background: #f0f4f8; }
.record-item.active { background: #ddf4ff; border-left: 3px solid #0969da; }

.record-item .step { color: #0969da; font-weight: 600; }
.record-item .meta { color: #656d76; font-size: 11px; margin-top: 2px; }

/* Main content */
.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Controls bar */
.controls {
    padding: 12px 20px;
    background: #ffffff;
    border-bottom: 1px solid #d1d9e0;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
}

.control-group {
    display: flex;
    align-items: center;
    gap: 6px;
}

.control-group label {
    font-size: 11px;
    color: #656d76;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.btn-group {
    display: flex;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #d1d9e0;
}

.btn-group button {
    padding: 6px 12px;
    background: #f6f8fa;
    color: #1f2328;
    border: none;
    font-size: 12px;
    font-family: inherit;
    cursor: pointer;
    transition: background 0.15s;
    border-right: 1px solid #d1d9e0;
}

.btn-group button:last-child { border-right: none; }
.btn-group button:hover { background: #eaeef2; }
.btn-group button.active { background: #0969da; color: #fff; }

.scale-control {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
}

.scale-control input[type=range] {
    width: 100px;
    accent-color: #0969da;
}

.scale-control .val {
    font-size: 11px;
    color: #0969da;
    min-width: 70px;
}

/* Color legend */
.legend {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 8px;
}

.legend-bar {
    width: 120px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #d1d9e0;
}

.legend-label {
    font-size: 10px;
    color: #656d76;
}

/* Content area */
.content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.section {
    margin-bottom: 24px;
}

.section-header {
    font-size: 12px;
    font-weight: 600;
    color: #656d76;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #eaeef2;
}

.prompt-box {
    background: #ffffff;
    border: 1px solid #d1d9e0;
    border-radius: 8px;
    padding: 14px;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 200px;
    overflow-y: auto;
    color: #656d76;
}

.token-display {
    background: #ffffff;
    border: 1px solid #d1d9e0;
    border-radius: 8px;
    padding: 16px;
    font-size: 14px;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-word;
}

.token {
    position: relative;
    cursor: default;
    border-radius: 2px;
    transition: outline 0.1s;
}

.token:hover {
    outline: 1px solid #0969da;
    z-index: 1;
}

/* Tooltip */
.tooltip {
    display: none;
    position: fixed;
    background: #ffffff;
    border: 1px solid #d1d9e0;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 11px;
    line-height: 1.5;
    z-index: 1000;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    max-width: 520px;
}

.tooltip .tt-topk-grid {
    display: flex;
    gap: 14px;
    margin-top: 4px;
}

.tooltip .tt-topk-col {
    flex: 1;
    min-width: 0;
}

.tooltip .tt-token {
    font-weight: 700;
    color: #1f2328;
    margin-bottom: 4px;
    font-size: 12px;
    word-break: break-all;
}

.tooltip .tt-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
}

.tooltip .tt-label { color: #656d76; }
.tooltip .tt-value { color: #0969da; font-weight: 600; }
.tooltip .tt-value.neg { color: #cf222e; }
.tooltip .tt-value.pos { color: #1a7f37; }

.tooltip .tt-divider {
    border-top: 1px solid #eaeef2;
    margin: 5px 0;
}

.tooltip .tt-topk-header {
    font-size: 10px;
    font-weight: 600;
    color: #656d76;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 3px;
}

.tooltip .tt-topk-row {
    display: flex;
    gap: 8px;
    font-size: 10.5px;
    line-height: 1.6;
}

.tooltip .tt-topk-row.actual {
    font-weight: 700;
    color: #0969da;
}

.tooltip .tt-topk-rank {
    color: #8b949e;
    min-width: 18px;
    text-align: right;
}

.tooltip .tt-topk-token {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.tooltip .tt-topk-prob {
    min-width: 44px;
    text-align: right;
    color: #656d76;
}

/* Stats bar */
.stats-bar {
    padding: 8px 20px;
    background: #ffffff;
    border-top: 1px solid #d1d9e0;
    font-size: 11px;
    color: #656d76;
    display: flex;
    gap: 20px;
}

.stats-bar .stat-val { color: #0969da; font-weight: 600; }
</style>
</head>
<body>

<div class="sidebar">
    <div class="sidebar-header">Rollout Visualizer</div>
    <div class="sidebar-info" id="fileInfo"></div>
    <div class="record-list" id="recordList"></div>
</div>

<div class="main">
    <div class="controls">
        <div class="control-group">
            <label>Response</label>
            <div class="btn-group" id="responseSelector">
                <button data-val="student" class="active">Student</button>
                <button data-val="teacher">Teacher</button>
            </div>
        </div>
        <div class="control-group">
            <label>Color by</label>
            <div class="btn-group" id="colorSelector">
                <button data-val="student" class="active">Student logp</button>
                <button data-val="teacher">Teacher logp</button>
                <button data-val="diff">Δ logp</button>
                <button data-val="h_student">Student H</button>
                <button data-val="h_teacher">Teacher H</button>
                <button data-val="h_diff">ΔH</button>
                <button data-val="kl">KL</button>
            </div>
        </div>
        <div class="control-group">
            <label>Filter</label>
            <div class="btn-group" id="filterSelector">
                <button data-val="none" class="active">None</button>
                <button data-val="entropy">Entropy</button>
                <button data-val="type_aware">Type-aware</button>
                <button data-val="h_diff">ΔH</button>
            </div>
        </div>
        <div class="control-group" id="retentionGroup" style="display:none;">
            <label>Keep</label>
            <input type="range" id="retentionSlider" min="0.01" max="1.0" value="0.5" step="0.01" style="width:100px;accent-color:#1a7f37;">
            <span class="val" id="retentionVal" style="color:#1a7f37;font-weight:600;font-size:11px;min-width:42px;">50%</span>
        </div>
        <div class="scale-control">
            <label style="font-size:11px;color:#656d76;">Scale</label>
            <input type="range" id="scaleSlider" min="-3" max="1.4" value="0.7" step="0.02">
            <span class="val" id="scaleVal">−5</span>
        </div>
        <div class="legend">
            <span class="legend-label" id="legendLow">-5</span>
            <canvas class="legend-bar" id="legendBar" width="120" height="14"></canvas>
            <span class="legend-label" id="legendHigh">0</span>
        </div>
    </div>

    <div class="content" id="content">
        <div class="section">
            <div class="section-header" id="promptHeader">Prompt</div>
            <div class="prompt-box" id="promptBox"></div>
        </div>
        <div class="section">
            <div class="section-header" id="responseHeader">Response</div>
            <div class="token-display" id="tokenDisplay"></div>
        </div>
    </div>

    <div class="stats-bar" id="statsBar"></div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
// ===== DATA (injected by Python) =====
const RECORDS = __RECORDS_JSON__;
const FILE_INFO = __FILE_INFO_JSON__;

// ===== STATE =====
let state = {
    recordIdx: 0,
    response: 'student',  // 'student' | 'teacher'
    // 'student' | 'teacher' | 'diff' | 'h_student' | 'h_teacher' | 'h_diff' | 'kl'
    colorBy: 'student',
    scale: 5,
    autoScale: true,       // recompute scale from data on each mode/record change
    filter: 'none',        // 'none' | 'entropy' | 'type_aware'
    retention: 0.5,        // fraction of tokens to keep when filtering
};

const SELECTED_COLOR = '#bce8c5';     // light green for selected tokens
const UNSELECTED_COLOR = '#f0f1f3';   // muted gray for unselected tokens
let currentFilterMask = null;          // last-rendered selection mask, for tooltip

// ===== COLOR FUNCTIONS =====
// Diverging green-red gradient via a light neutral midpoint.
// t=0 (bad) → red, t=0.5 → light neutral, t=1 (good) → green.
function gradient(t) {
    t = Math.max(0, Math.min(1, t));
    // Endpoints chosen to look clean on a white background
    const red = [222, 130, 130];
    const mid = [245, 240, 225];
    const grn = [125, 190, 145];
    let r, g, b;
    if (t < 0.5) {
        const s = t * 2;
        r = Math.round(red[0] + (mid[0] - red[0]) * s);
        g = Math.round(red[1] + (mid[1] - red[1]) * s);
        b = Math.round(red[2] + (mid[2] - red[2]) * s);
    } else {
        const s = (t - 0.5) * 2;
        r = Math.round(mid[0] + (grn[0] - mid[0]) * s);
        g = Math.round(mid[1] + (grn[1] - mid[1]) * s);
        b = Math.round(mid[2] + (grn[2] - mid[2]) * s);
    }
    return `rgb(${r},${g},${b})`;
}

const MISSING_COLOR = '#e6e6e6';

function logprobColor(lp, scale) {
    // lp ∈ [-scale, 0] mapped to red→green
    return gradient((lp + scale) / scale);
}

function diffColor(diff, scale) {
    // diff ∈ [-scale, +scale] mapped to red→neutral→green
    return gradient((diff / scale + 1) / 2);
}

function sequentialColor(value, scale) {
    // value ∈ [0, scale]; light at 0, deep blue at scale.
    const t = Math.max(0, Math.min(1, value / scale));
    const lo = [245, 248, 253];   // near-white
    const hi = [110, 145, 195];   // softened blue
    const r = Math.round(lo[0] + (hi[0] - lo[0]) * t);
    const g = Math.round(lo[1] + (hi[1] - lo[1]) * t);
    const b = Math.round(lo[2] + (hi[2] - lo[2]) * t);
    return `rgb(${r},${g},${b})`;
}

// ===== ENTROPY / KL FROM TOPK =====
function entropyFromTopk(topk) {
    if (!topk || topk.length === 0) return null;
    let h = 0;
    for (const e of topk) {
        const p = Math.exp(e.logprob);
        if (p > 0) h += -p * e.logprob;
    }
    return h;
}

function klFromTopks(topkP, topkQ) {
    // Approximate KL(P || Q) using the topks. For tokens in P's topk that
    // aren't in Q's topk, fall back to (Q's lowest topk logprob − 2) as a soft floor.
    if (!topkP || !topkQ || topkP.length === 0 || topkQ.length === 0) return null;
    const qMap = new Map();
    let qMin = 0;
    for (const e of topkQ) {
        qMap.set(e.token_id, e.logprob);
        if (e.logprob < qMin) qMin = e.logprob;
    }
    const qFloor = qMin - 2;
    let kl = 0;
    for (const e of topkP) {
        const p = Math.exp(e.logprob);
        const lpQ = qMap.has(e.token_id) ? qMap.get(e.token_id) : qFloor;
        kl += p * (e.logprob - lpQ);
    }
    return Math.max(0, kl);
}

function getEntropyTeacher(tok) { return entropyFromTopk(tok.topk_teacher); }
function getEntropyStudent(tok) { return entropyFromTopk(tok.topk_student); }

function getKL(tok) {
    // KL(student || teacher). Requires both topks → only computable on teacher response
    // since student_response_topk_under_student isn't dumped.
    return klFromTopks(tok.topk_student, tok.topk_teacher);
}

// ===== RENDERING =====
function renderRecordList() {
    const list = document.getElementById('recordList');
    list.innerHTML = '';
    RECORDS.forEach((rec, i) => {
        const div = document.createElement('div');
        div.className = 'record-item' + (i === state.recordIdx ? ' active' : '');
        div.innerHTML = `<div><span class="step">Step ${rec.step}</span> &middot; Batch ${rec.batch_idx}</div>
            <div class="meta">${rec.student_tokens.length} student / ${rec.teacher_tokens.length} teacher tokens</div>`;
        div.onclick = () => { state.recordIdx = i; state.autoScale = true; render(); };
        list.appendChild(div);
    });
    document.getElementById('fileInfo').textContent = FILE_INFO;
}

function getTokens() {
    const rec = RECORDS[state.recordIdx];
    return state.response === 'student' ? rec.student_tokens : rec.teacher_tokens;
}

function getEntropyDiff(tok) {
    const hs = getEntropyStudent(tok);
    const ht = getEntropyTeacher(tok);
    if (hs === null || ht === null) return null;
    return ht - hs;
}

// ===== TOKEN SELECTION FILTERS =====
// Mirrors verl/trainer/ppo/core_algos.py :: compute_token_filter_mask.
// Returns an array of booleans (one per token) marking selected tokens, or
// null if the filter is off / not applicable. Tokens missing the required
// data are excluded from the active set and never selected.
function computeFilterMask() {
    if (state.filter === 'none') return null;
    const tokens = getTokens();
    const n = tokens.length;
    const scores = new Array(n).fill(null);

    if (state.filter === 'entropy') {
        for (let i = 0; i < n; i++) {
            const h = getEntropyStudent(tokens[i]);
            if (h !== null && isFinite(h)) scores[i] = h;
        }
    } else if (state.filter === 'h_diff') {
        // Rank tokens by H_student − H_teacher (student more uncertain than teacher).
        for (let i = 0; i < n; i++) {
            const hs = getEntropyStudent(tokens[i]);
            const ht = getEntropyTeacher(tokens[i]);
            if (hs !== null && ht !== null && isFinite(hs) && isFinite(ht)) {
                scores[i] = hs - ht;
            }
        }
    } else if (state.filter === 'type_aware') {
        // s_t = 1 − (1 − ĥ_t)(1 − δ̂_t), with min-max normalization over
        // the active tokens (those where both H_S and KL are computable).
        const hs = new Array(n).fill(null);
        const ds = new Array(n).fill(null);
        for (let i = 0; i < n; i++) {
            const h = getEntropyStudent(tokens[i]);
            const d = getKL(tokens[i]);
            if (h !== null && d !== null && isFinite(h) && isFinite(d)) {
                hs[i] = h;
                ds[i] = d;
            }
        }
        let hMin = Infinity, hMax = -Infinity, dMin = Infinity, dMax = -Infinity;
        for (let i = 0; i < n; i++) {
            if (hs[i] === null) continue;
            if (hs[i] < hMin) hMin = hs[i];
            if (hs[i] > hMax) hMax = hs[i];
            if (ds[i] < dMin) dMin = ds[i];
            if (ds[i] > dMax) dMax = ds[i];
        }
        const hRange = Math.max(1e-8, hMax - hMin);
        const dRange = Math.max(1e-8, dMax - dMin);
        for (let i = 0; i < n; i++) {
            if (hs[i] === null) continue;
            const hHat = Math.min(1, Math.max(0, (hs[i] - hMin) / hRange));
            const dHat = Math.min(1, Math.max(0, (ds[i] - dMin) / dRange));
            scores[i] = 1 - (1 - hHat) * (1 - dHat);
        }
    }

    // Top-k selection by score.
    const valid = [];
    for (let i = 0; i < n; i++) if (scores[i] !== null) valid.push(scores[i]);
    if (valid.length === 0) return new Array(n).fill(false);
    const nKeep = Math.max(1, Math.floor(state.retention * valid.length));
    const sortedDesc = valid.slice().sort((a, b) => b - a);
    const threshold = sortedDesc[Math.min(nKeep - 1, sortedDesc.length - 1)];

    // Tie-break: keep at most nKeep entries even if many tokens share the
    // threshold (matches torch.topk's deterministic behavior).
    const mask = new Array(n).fill(false);
    let kept = 0;
    // First pass: tokens strictly above the threshold are always selected.
    for (let i = 0; i < n; i++) {
        if (scores[i] !== null && scores[i] > threshold) {
            mask[i] = true;
            kept++;
        }
    }
    // Second pass: tokens at the threshold, in input order, until we hit nKeep.
    for (let i = 0; i < n && kept < nKeep; i++) {
        if (scores[i] !== null && scores[i] === threshold && !mask[i]) {
            mask[i] = true;
            kept++;
        }
    }
    return mask;
}

function getColorValue(tok) {
    // Returns the scalar used for coloring this token under the current mode.
    // May return null if the data needed isn't available.
    switch (state.colorBy) {
        case 'student':   return tok.lp_student;
        case 'teacher':   return tok.lp_teacher;
        case 'diff':      return tok.lp_teacher - tok.lp_student;  // T − S
        case 'h_student': return getEntropyStudent(tok);
        case 'h_teacher': return getEntropyTeacher(tok);
        case 'h_diff':    return getEntropyDiff(tok);              // T − S
        case 'kl':        return getKL(tok);
    }
    return null;
}

function isNonNegMode() {
    return state.colorBy === 'h_student' || state.colorBy === 'h_teacher' || state.colorBy === 'kl';
}

function isDiffMode() {
    return state.colorBy === 'diff' || state.colorBy === 'h_diff';
}

function colorForValue(value) {
    if (value === null || value === undefined || !isFinite(value)) return MISSING_COLOR;
    if (isDiffMode()) return diffColor(value, state.scale);
    if (isNonNegMode()) return sequentialColor(value, state.scale);
    // Student/Teacher logp: non-positive; intensity grows with magnitude.
    return sequentialColor(-value, state.scale);
}

function percentile(sorted, p) {
    if (sorted.length === 0) return 0;
    const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * p));
    return sorted[idx];
}

function computeAutoScale() {
    const tokens = getTokens();
    const values = [];
    for (const tok of tokens) {
        const v = getColorValue(tok);
        if (v !== null && v !== undefined && isFinite(v)) values.push(v);
    }
    if (values.length === 0) {
        // No data for this mode (e.g., Student H / KL / ΔH on the student response,
        // because student_response_topk_under_student isn't dumped). Pick a
        // sensible default rather than carrying over the previous mode's scale.
        if (state.colorBy === 'student' || state.colorBy === 'teacher') return 5;
        if (isDiffMode()) return 1;
        return 1;  // entropy / kl
    }

    let raw;
    if (state.colorBy === 'student' || state.colorBy === 'teacher') {
        // Logprobs are ≤ 0; take magnitude of the 5th percentile (most negative tail).
        const sorted = values.slice().sort((a, b) => a - b);
        raw = -percentile(sorted, 0.05);
    } else if (isDiffMode()) {
        const abs = values.map(Math.abs).sort((a, b) => a - b);
        raw = percentile(abs, 0.95);
    } else {
        // h_student, h_teacher, kl: non-negative
        const sorted = values.slice().sort((a, b) => a - b);
        raw = percentile(sorted, 0.95);
    }

    if (!isFinite(raw) || raw <= 0) raw = 0.01;
    // Clamp to the slider's log-space range [10^-3, 10^1.4].
    return Math.max(0.001, Math.min(25, raw));
}

function formatScale(s) {
    // Choose precision based on magnitude
    if (s >= 10) return s.toFixed(1);
    if (s >= 1) return s.toFixed(2);
    if (s >= 0.1) return s.toFixed(3);
    return s.toFixed(4);
}

function render() {
    const rec = RECORDS[state.recordIdx];
    const tokens = getTokens();

    if (state.autoScale) {
        state.scale = computeAutoScale();
    }
    // Slider operates in log10 space.
    document.getElementById('scaleSlider').value = Math.log10(state.scale);
    const sign = isNonNegMode() ? '' : (isDiffMode() ? '±' : '−');
    document.getElementById('scaleVal').textContent =
        `${sign}${formatScale(state.scale)}` + (state.autoScale ? ' (auto)' : '');

    document.querySelectorAll('.record-item').forEach((el, i) => {
        el.classList.toggle('active', i === state.recordIdx);
    });

    const promptHeader = document.getElementById('promptHeader');
    const promptBox = document.getElementById('promptBox');
    if (state.response === 'student') {
        promptHeader.textContent = 'Student Prompt';
        promptBox.textContent = rec.student_prompt;
    } else {
        promptHeader.textContent = 'Teacher Prompt';
        promptBox.textContent = rec.teacher_prompt;
    }

    const filterMask = computeFilterMask();
    currentFilterMask = filterMask;
    let nSelected = 0;
    if (filterMask) for (const b of filterMask) if (b) nSelected++;

    const responseHeader = document.getElementById('responseHeader');
    const labels = {
        student:   'student logprob',
        teacher:   'teacher logprob',
        diff:      'Δ logprob (teacher − student)',
        h_student: 'student entropy',
        h_teacher: 'teacher entropy',
        h_diff:    'ΔH (teacher − student)',
        kl:        'KL(student ‖ teacher)',
    };
    if (filterMask) {
        const filterLabel =
            state.filter === 'entropy' ? 'entropy' :
            state.filter === 'h_diff' ? 'ΔH (student − teacher)' :
            'type-aware';
        responseHeader.textContent =
            `${state.response} response — ${filterLabel} filter, keep top ${(state.retention * 100).toFixed(0)}%`;
    } else {
        responseHeader.textContent = `${state.response} response — colored by ${labels[state.colorBy]}`;
    }

    const display = document.getElementById('tokenDisplay');
    display.innerHTML = '';
    const fragment = document.createDocumentFragment();

    let sumLp = 0, minLp = 0, maxLp = -Infinity;
    let sumHs = 0, nHs = 0;
    let sumHt = 0, nHt = 0;
    let sumKL = 0, nKL = 0;
    let nColored = 0;

    tokens.forEach((tok, i) => {
        const v = getColorValue(tok);
        const span = document.createElement('span');
        span.className = 'token';
        span.textContent = tok.text;
        span.dataset.idx = i;
        if (filterMask) {
            span.style.backgroundColor = filterMask[i] ? SELECTED_COLOR : UNSELECTED_COLOR;
        } else {
            span.style.backgroundColor = colorForValue(v);
        }
        span.style.color = '#000';

        if (v !== null && v !== undefined && isFinite(v)) nColored++;

        fragment.appendChild(span);

        const ownLp = state.response === 'student' ? tok.lp_student : tok.lp_teacher;
        sumLp += ownLp;
        minLp = Math.min(minLp, ownLp);
        maxLp = Math.max(maxLp, ownLp);

        const hs = getEntropyStudent(tok);
        if (hs !== null && isFinite(hs)) { sumHs += hs; nHs++; }
        const ht = getEntropyTeacher(tok);
        if (ht !== null && isFinite(ht)) { sumHt += ht; nHt++; }
        const kl = getKL(tok);
        if (kl !== null && isFinite(kl)) { sumKL += kl; nKL++; }
    });

    display.appendChild(fragment);

    const n = tokens.length;
    const meanLp = n > 0 ? sumLp / n : 0;
    const ppl = n > 0 ? Math.exp(-sumLp / n) : 0;
    let statsHtml =
        `<span>Tokens: <span class="stat-val">${n}</span></span>` +
        `<span>Mean logp: <span class="stat-val">${meanLp.toFixed(3)}</span></span>` +
        `<span>PPL: <span class="stat-val">${ppl.toFixed(2)}</span></span>` +
        `<span>Min: <span class="stat-val">${minLp.toFixed(3)}</span></span>` +
        `<span>Max: <span class="stat-val">${maxLp.toFixed(3)}</span></span>`;
    if (nHs > 0) statsHtml += `<span>Mean H<sub>S</sub>: <span class="stat-val">${(sumHs / nHs).toFixed(3)}</span></span>`;
    if (nHt > 0) statsHtml += `<span>Mean H<sub>T</sub>: <span class="stat-val">${(sumHt / nHt).toFixed(3)}</span></span>`;
    if (nHs > 0 && nHt > 0) {
        const meanDh = sumHt / nHt - sumHs / nHs;
        statsHtml += `<span>Mean ΔH: <span class="stat-val">${meanDh >= 0 ? '+' : ''}${meanDh.toFixed(3)}</span></span>`;
    }
    if (nKL > 0) statsHtml += `<span>Mean KL: <span class="stat-val">${(sumKL / nKL).toFixed(3)}</span></span>`;
    if (nColored === 0 && n > 0 && !filterMask) {
        statsHtml += `<span style="color:#cf222e;">No data for this mode in the ${state.response} response (student topk not dumped on student response)</span>`;
    }
    if (filterMask) {
        const pct = n > 0 ? (nSelected / n * 100).toFixed(1) : '0.0';
        statsHtml += `<span style="color:#1a7f37;">Selected: <span class="stat-val" style="color:#1a7f37;">${nSelected} / ${n} (${pct}%)</span></span>`;
        if (nSelected === 0 && state.response === 'student') {
            statsHtml += `<span style="color:#cf222e;">Filter unavailable on student response (needs student topk)</span>`;
        }
    }
    document.getElementById('statsBar').innerHTML = statsHtml;

    updateLegend();
}

function updateLegend() {
    const canvas = document.getElementById('legendBar');
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (state.filter !== 'none') {
        // Two-tone legend: unselected (left) → selected (right).
        ctx.fillStyle = UNSELECTED_COLOR;
        ctx.fillRect(0, 0, Math.floor(w / 2), h);
        ctx.fillStyle = SELECTED_COLOR;
        ctx.fillRect(Math.floor(w / 2), 0, w - Math.floor(w / 2), h);
        document.getElementById('legendLow').textContent = 'unsel';
        document.getElementById('legendHigh').textContent = 'sel';
        return;
    }

    const sLabel = formatScale(state.scale);
    let lowLabel, highLabel;
    if (isNonNegMode()) {
        lowLabel = '0';
        highLabel = sLabel;
        for (let x = 0; x < w; x++) {
            ctx.fillStyle = sequentialColor((x / (w - 1)) * state.scale, state.scale);
            ctx.fillRect(x, 0, 1, h);
        }
    } else if (isDiffMode()) {
        lowLabel = `−${sLabel}`;
        highLabel = `+${sLabel}`;
        for (let x = 0; x < w; x++) {
            ctx.fillStyle = gradient(x / (w - 1));
            ctx.fillRect(x, 0, 1, h);
        }
    } else {
        // Student/Teacher logp: blue sequential, intense on the left (very negative).
        lowLabel = `−${sLabel}`;
        highLabel = '0';
        for (let x = 0; x < w; x++) {
            const intensity = (1 - x / (w - 1)) * state.scale;
            ctx.fillStyle = sequentialColor(intensity, state.scale);
            ctx.fillRect(x, 0, 1, h);
        }
    }

    document.getElementById('legendLow').textContent = lowLabel;
    document.getElementById('legendHigh').textContent = highLabel;
}

// ===== TOOLTIP =====
function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/ /g, '&middot;').replace(/\n/g, '↵').replace(/\t/g, '→');
}

const tooltip = document.getElementById('tooltip');
document.getElementById('tokenDisplay').addEventListener('mousemove', (e) => {
    const span = e.target.closest('.token');
    if (!span) { tooltip.style.display = 'none'; return; }

    const idx = parseInt(span.dataset.idx);
    const tokens = getTokens();
    const tok = tokens[idx];
    if (!tok) { tooltip.style.display = 'none'; return; }

    const lpS = tok.lp_student;
    const lpT = tok.lp_teacher;
    const diff = lpT - lpS;
    const probS = Math.exp(lpS);
    const probT = Math.exp(lpT);

    const hS = getEntropyStudent(tok);
    const hT = getEntropyTeacher(tok);
    const kl = getKL(tok);

    let html = `
        <div class="tt-token">"${escapeHtml(tok.text)}"</div>
        <div class="tt-row"><span class="tt-label">Student logp:</span>
            <span class="tt-value ${lpS < -2 ? 'neg' : 'pos'}">${lpS.toFixed(4)}</span></div>
        <div class="tt-row"><span class="tt-label">Student prob:</span>
            <span class="tt-value">${(probS * 100).toFixed(2)}%</span></div>
        <div class="tt-row"><span class="tt-label">Teacher logp:</span>
            <span class="tt-value ${lpT < -2 ? 'neg' : 'pos'}">${lpT.toFixed(4)}</span></div>
        <div class="tt-row"><span class="tt-label">Teacher prob:</span>
            <span class="tt-value">${(probT * 100).toFixed(2)}%</span></div>
        <div class="tt-row"><span class="tt-label">Δ logp (T−S):</span>
            <span class="tt-value ${diff > 0 ? 'pos' : 'neg'}">${diff > 0 ? '+' : ''}${diff.toFixed(4)}</span></div>`;
    if (hS !== null) {
        html += `<div class="tt-row"><span class="tt-label">Student H:</span>
            <span class="tt-value">${hS.toFixed(4)}</span></div>`;
    }
    if (hT !== null) {
        html += `<div class="tt-row"><span class="tt-label">Teacher H:</span>
            <span class="tt-value">${hT.toFixed(4)}</span></div>`;
    }
    if (hS !== null && hT !== null) {
        const dh = hT - hS;
        html += `<div class="tt-row"><span class="tt-label">ΔH (T−S):</span>
            <span class="tt-value ${dh > 0 ? 'pos' : 'neg'}">${dh > 0 ? '+' : ''}${dh.toFixed(4)}</span></div>`;
    }
    if (kl !== null) {
        html += `<div class="tt-row"><span class="tt-label">KL(stu‖tea):</span>
            <span class="tt-value">${kl.toFixed(4)}</span></div>`;
    }
    html += `<div class="tt-row"><span class="tt-label">Token idx:</span>
            <span class="tt-value">${idx}</span></div>`;
    if (state.filter !== 'none' && currentFilterMask) {
        const isSelected = currentFilterMask[idx];
        html += `<div class="tt-row"><span class="tt-label">Filter:</span>
            <span class="tt-value ${isSelected ? 'pos' : 'neg'}">${isSelected ? 'selected' : 'not selected'}</span></div>`;
    }

    // Top-k alternatives — both student and teacher, side by side.
    const renderTopkColumn = (topk, label, actualLp) => {
        if (!topk || topk.length === 0) {
            return `<div class="tt-topk-col"><div class="tt-topk-header">${label}</div>` +
                `<div class="tt-topk-row" style="color:#8b949e;">(no data)</div></div>`;
        }
        let actualRank = null;
        for (let j = 0; j < topk.length; j++) {
            if (Math.abs(topk[j].logprob - actualLp) < 1e-4) { actualRank = j; break; }
        }
        let h = `<div class="tt-topk-col">` +
            `<div class="tt-topk-header">${label}${actualRank !== null ? ` (#${actualRank + 1})` : ' (∉ top-k)'}</div>`;
        for (let j = 0; j < topk.length; j++) {
            const alt = topk[j];
            const isActual = j === actualRank;
            const prob = Math.exp(alt.logprob) * 100;
            h += `<div class="tt-topk-row${isActual ? ' actual' : ''}">` +
                `<span class="tt-topk-rank">${j + 1}.</span>` +
                `<span class="tt-topk-token">${escapeHtml(alt.token)}</span>` +
                `<span class="tt-topk-prob">${prob >= 1 ? prob.toFixed(1) : prob.toFixed(2)}%</span>` +
                `</div>`;
        }
        h += `</div>`;
        return h;
    };

    if (tok.topk_student || tok.topk_teacher) {
        html += `<div class="tt-divider"></div>`;
        html += `<div class="tt-topk-grid">`;
        html += renderTopkColumn(tok.topk_student, 'Student', tok.lp_student);
        html += renderTopkColumn(tok.topk_teacher, 'Teacher', tok.lp_teacher);
        html += `</div>`;
    }

    tooltip.innerHTML = html;
    tooltip.style.display = 'block';

    let tx = e.clientX + 12, ty = e.clientY + 12;
    const rect = tooltip.getBoundingClientRect();
    if (tx + rect.width > window.innerWidth) tx = e.clientX - rect.width - 12;
    if (ty + rect.height > window.innerHeight) ty = e.clientY - rect.height - 12;
    tooltip.style.left = tx + 'px';
    tooltip.style.top = ty + 'px';
});

document.getElementById('tokenDisplay').addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
});

// ===== CONTROLS =====
function setupBtnGroup(id, stateKey) {
    const group = document.getElementById(id);
    group.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
            group.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state[stateKey] = btn.dataset.val;
            // Mode / response change → re-enable auto-scale
            state.autoScale = true;
            render();
        });
    });
}

setupBtnGroup('responseSelector', 'response');
setupBtnGroup('colorSelector', 'colorBy');

// Filter selector — toggle the retention slider's visibility based on choice.
(function () {
    const group = document.getElementById('filterSelector');
    const retentionGroup = document.getElementById('retentionGroup');
    group.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
            group.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.filter = btn.dataset.val;
            retentionGroup.style.display = state.filter === 'none' ? 'none' : '';
            render();
        });
    });
})();

document.getElementById('retentionSlider').addEventListener('input', (e) => {
    state.retention = parseFloat(e.target.value);
    document.getElementById('retentionVal').textContent = `${(state.retention * 100).toFixed(0)}%`;
    render();
});

document.getElementById('scaleSlider').addEventListener('input', (e) => {
    state.scale = Math.pow(10, parseFloat(e.target.value));
    state.autoScale = false;
    render();
});

// ===== INIT =====
renderRecordList();
render();
</script>
</body>
</html>"""


def generate_html(records_data: list[dict], file_info: str, output_path: str):
    records_json = json.dumps(records_data)
    file_info_json = json.dumps(file_info)

    html_content = HTML_TEMPLATE.replace("__RECORDS_JSON__", records_json).replace(
        "__FILE_INFO_JSON__", file_info_json
    )

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Written to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize rollout dumps with token-level logprob coloring"
    )
    parser.add_argument("input", help="Path to JSONL rollout dump file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML path (default: <input_stem>_viz.html)",
    )
    parser.add_argument(
        "--record",
        type=int,
        default=None,
        help="Only process a specific record index (0-based)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model id for the tokenizer (default: Qwen/Qwen3-8B).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_suffix("")) + "_viz.html"

    print(f"Loading tokenizer: {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading data from {input_path}...", file=sys.stderr)
    records = load_data(str(input_path))

    if args.record is not None:
        if 0 <= args.record < len(records):
            records = [records[args.record]]
        else:
            print(
                f"Error: record index {args.record} out of range (0-{len(records)-1})",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Processing {len(records)} records...", file=sys.stderr)
    records_data = build_token_data(tokenizer, records)

    file_info = f"{input_path.name} — {len(records)} records"
    generate_html(records_data, file_info, output_path)
    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
