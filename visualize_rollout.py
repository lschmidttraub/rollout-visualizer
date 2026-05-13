#!/usr/bin/env python3
"""
Rollout Visualizer: Browse rollouts from an evaluation-style dump.

Each record is expected to have at minimum: input, output, gts, pred, acc, score.
Optional fields like step, incorrect_format, truncated, feedback are surfaced
in the metadata bar when present.

Usage:
    python3 visualize_rollout.py <path_to_jsonl> [-o output.html]

The script generates a self-contained HTML file that can be opened in any browser.
"""

import argparse
import json
import sys
from pathlib import Path


def load_data(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rollout Viewer</title>
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
    width: 300px;
    min-width: 300px;
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
    padding: 6px 16px;
    border-bottom: 1px solid #d1d9e0;
    font-size: 11px;
    color: #656d76;
}

.filter-bar {
    padding: 10px 14px;
    border-bottom: 1px solid #d1d9e0;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.filter-bar label {
    font-size: 10px;
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
    flex: 1;
    padding: 6px 8px;
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

.record-list {
    overflow-y: auto;
    flex: 1;
}

.record-item {
    padding: 10px 14px;
    border-bottom: 1px solid #eaeef2;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.15s;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}

.record-item:hover { background: #f0f4f8; }
.record-item.active { background: #ddf4ff; border-left: 3px solid #0969da; padding-left: 11px; }

.record-item .badge {
    flex-shrink: 0;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 5px;
}
.record-item .badge.correct { background: #1a7f37; }
.record-item .badge.incorrect { background: #cf222e; }
.record-item .badge.unknown { background: #8b949e; }

.record-item .info { flex: 1; min-width: 0; }
.record-item .idx { color: #0969da; font-weight: 600; }
.record-item .meta { color: #656d76; font-size: 11px; margin-top: 4px; word-break: break-word; }

.record-item .ratio {
    font-size: 11px;
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 3px;
}
.record-item .ratio.correct { background: #dcffe4; color: #1a7f37; }
.record-item .ratio.incorrect { background: #ffebe9; color: #cf222e; }
.record-item .ratio.mixed { background: #fff8c5; color: #9a6700; }

.record-item .dots {
    display: inline-flex;
    gap: 2px;
    margin-left: 4px;
    vertical-align: middle;
}
.dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
}
.dot.correct { background: #1a7f37; }
.dot.incorrect { background: #cf222e; }
.dot.unknown { background: #8b949e; }

/* Rollout cards (Detail view) */
.rollout-card {
    background: #ffffff;
    border: 1px solid #d1d9e0;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
}
.rollout-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 12px;
}
.rollout-head .rollout-idx { font-weight: 600; color: #1f2328; margin-right: 12px; }
.rollout-head .rollout-pred { color: #656d76; margin-right: 12px; }
.rollout-head .rollout-pred .correct { color: #1a7f37; font-weight: 600; }
.rollout-head .rollout-pred .incorrect { color: #cf222e; font-weight: 600; }
.rollout-head .rollout-result { font-weight: 700; }
.rollout-head .rollout-result.correct { color: #1a7f37; }
.rollout-head .rollout-result.incorrect { color: #cf222e; }
.rollout-head .rollout-flags { color: #bf8700; font-size: 11px; }

/* Jump-to bar inside Detail view */
.jump-bar {
    position: sticky;
    top: -20px;  /* cancels the content padding so it sits flush */
    background: #f6f8fa;
    margin: -20px -20px 14px -20px;
    padding: 10px 20px;
    border-bottom: 1px solid #d1d9e0;
    display: flex;
    gap: 6px;
    align-items: center;
    flex-wrap: wrap;
    z-index: 5;
}
.jump-bar .jump-label {
    font-size: 10px;
    color: #656d76;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-right: 4px;
}
.jump-bar button {
    padding: 3px 9px;
    border: 1px solid #d1d9e0;
    border-radius: 4px;
    background: #ffffff;
    font-size: 12px;
    font-family: inherit;
    cursor: pointer;
    font-weight: 600;
    color: #1f2328;
    transition: background 0.1s;
}
.jump-bar button:hover { background: #f0f4f8; }
.jump-bar button.correct { color: #1a7f37; border-color: #b4e0c0; background: #f0fbf4; }
.jump-bar button.incorrect { color: #cf222e; border-color: #f5c0bc; background: #fff4f3; }
.jump-bar button.correct:hover { background: #dcffe4; }
.jump-bar button.incorrect:hover { background: #ffebe9; }

/* Main content */
.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.metadata-bar {
    padding: 10px 20px;
    background: #ffffff;
    border-bottom: 1px solid #d1d9e0;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    font-size: 12px;
}

.metadata-bar .meta-cell {
    display: flex;
    gap: 6px;
    align-items: center;
}

.metadata-bar .meta-label {
    color: #656d76;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metadata-bar .meta-value {
    color: #0969da;
    font-weight: 600;
}

.metadata-bar .meta-value.correct { color: #1a7f37; }
.metadata-bar .meta-value.incorrect { color: #cf222e; }
.metadata-bar .meta-value.warn { color: #bf8700; }

.content-toolbar {
    padding: 8px 20px;
    background: #ffffff;
    border-bottom: 1px solid #d1d9e0;
    display: flex;
    gap: 10px;
    align-items: center;
}

.content-toolbar label {
    font-size: 10px;
    color: #656d76;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.compare-table {
    width: 100%;
    font-size: 12px;
    border-collapse: collapse;
    background: #ffffff;
    border: 1px solid #d1d9e0;
    border-radius: 6px;
    overflow: hidden;
}
.compare-table th, .compare-table td {
    padding: 8px 10px;
    text-align: left;
    border-bottom: 1px solid #eaeef2;
    vertical-align: top;
}
.compare-table th {
    background: #f6f8fa;
    font-weight: 600;
    color: #656d76;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    position: sticky;
    top: 0;
    z-index: 1;
}
.compare-table tr.row { cursor: pointer; }
.compare-table tr.row:hover { background: #f0f4f8; }
.compare-table tr.row.active { background: #ddf4ff; }
.compare-table .badge {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.compare-table .badge.correct { background: #1a7f37; }
.compare-table .badge.incorrect { background: #cf222e; }
.compare-table .badge.unknown { background: #8b949e; }
.compare-table td.correct { color: #1a7f37; font-weight: 600; }
.compare-table td.incorrect { color: #cf222e; font-weight: 600; }
.compare-table td.idx { color: #0969da; font-weight: 600; white-space: nowrap; }
.compare-table td.gt, .compare-table td.pred {
    font-family: inherit;
    word-break: break-word;
    max-width: 240px;
}
.compare-table td.ratio {
    font-weight: 700;
    white-space: nowrap;
}
.compare-table td.ratio.correct { color: #1a7f37; }
.compare-table td.ratio.incorrect { color: #cf222e; }

.section {
    margin-bottom: 20px;
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

.text-box {
    background: #ffffff;
    border: 1px solid #d1d9e0;
    border-radius: 8px;
    padding: 14px;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    color: #1f2328;
}

.text-box.muted { color: #656d76; }
.empty-state {
    padding: 40px;
    text-align: center;
    color: #8b949e;
    font-size: 13px;
}
</style>
</head>
<body>

<div class="sidebar">
    <div class="sidebar-header">Rollout Viewer</div>
    <div class="sidebar-info" id="fileInfo"></div>
    <div class="filter-bar">
        <div>
            <label>Correctness (per prompt)</label>
            <div class="btn-group" id="correctnessFilter">
                <button data-val="all" class="active">All</button>
                <button data-val="all_correct">All✓</button>
                <button data-val="mixed">Mixed</button>
                <button data-val="all_incorrect">All✗</button>
            </div>
        </div>
        <div>
            <label>Flags</label>
            <div class="btn-group" id="flagsFilter">
                <button data-val="any" class="active">Any</button>
                <button data-val="truncated">Truncated</button>
                <button data-val="badformat">Bad format</button>
            </div>
        </div>
    </div>
    <div class="record-list" id="recordList"></div>
</div>

<div class="main">
    <div class="content-toolbar">
        <label>View</label>
        <div class="btn-group" id="viewSelector">
            <button data-val="detail" class="active">Detail</button>
            <button data-val="compare">Compare (GT vs Pred)</button>
        </div>
    </div>
    <div class="metadata-bar" id="metadataBar"></div>
    <div class="content" id="content">
        <div class="empty-state">Select a record from the sidebar</div>
    </div>
</div>

<script>
const RECORDS = __RECORDS_JSON__;
const FILE_INFO = __FILE_INFO_JSON__;

function escapeHtml(s) {
    return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function isCorrect(rec) {
    if (rec.acc !== undefined && rec.acc !== null) return rec.acc >= 0.5;
    if (rec.score !== undefined && rec.score !== null) return rec.score >= 0.5;
    return null;
}

// ===== GROUPING =====
// Group rollouts by their input (prompt). Each prompt is expected to have a
// handful of rollouts (e.g. 8 for n=8 generation).
const GROUPS = (() => {
    const byInput = new Map();
    const order = [];
    for (let i = 0; i < RECORDS.length; i++) {
        const r = RECORDS[i];
        const k = r.input || '';
        if (!byInput.has(k)) {
            byInput.set(k, { input: k, rollouts: [], indices: [] });
            order.push(k);
        }
        const g = byInput.get(k);
        g.rollouts.push(r);
        g.indices.push(i);
    }
    return order.map(k => byInput.get(k));
})();

function groupStats(g) {
    let nCorrect = 0, nTotal = 0;
    for (const r of g.rollouts) {
        const c = isCorrect(r);
        if (c !== null) { nTotal++; if (c) nCorrect++; }
    }
    return { nCorrect, nTotal };
}

let state = {
    selectedGroupIdx: null,
    correctness: 'all',   // 'all' | 'all_correct' | 'mixed' | 'all_incorrect'
    flags: 'any',         // 'any' | 'truncated' | 'badformat'
    view: 'detail',       // 'detail' | 'compare'
};

function filteredGroupIndices() {
    const out = [];
    for (let i = 0; i < GROUPS.length; i++) {
        const g = GROUPS[i];
        const { nCorrect, nTotal } = groupStats(g);
        if (state.correctness === 'all_correct' && !(nTotal > 0 && nCorrect === nTotal)) continue;
        if (state.correctness === 'all_incorrect' && !(nTotal > 0 && nCorrect === 0)) continue;
        if (state.correctness === 'mixed' && !(nCorrect > 0 && nCorrect < nTotal)) continue;
        if (state.flags === 'truncated' && !g.rollouts.some(r => r.truncated)) continue;
        if (state.flags === 'badformat' && !g.rollouts.some(r => r.incorrect_format)) continue;
        out.push(i);
    }
    return out;
}

function dotRow(rollouts) {
    return rollouts.map(r => {
        const c = isCorrect(r);
        const cls = c === true ? 'correct' : c === false ? 'incorrect' : 'unknown';
        return `<span class="dot ${cls}"></span>`;
    }).join('');
}

function renderGroupList() {
    const list = document.getElementById('recordList');
    list.innerHTML = '';
    const idxs = filteredGroupIndices();
    if (idxs.length === 0) {
        list.innerHTML = '<div class="empty-state">No prompts match the current filters.</div>';
    } else {
        idxs.forEach(gi => {
            const g = GROUPS[gi];
            const { nCorrect, nTotal } = groupStats(g);
            const div = document.createElement('div');
            div.className = 'record-item' + (gi === state.selectedGroupIdx ? ' active' : '');

            let preview = (g.input || '').replace(/\s+/g, ' ').trim();
            if (preview.length > 90) preview = preview.slice(0, 90) + '…';

            const summaryCls = nCorrect === nTotal ? 'correct' :
                                nCorrect === 0 ? 'incorrect' : 'mixed';

            div.innerHTML =
                `<div class="info">` +
                    `<div>` +
                        `<span class="idx">P${gi}</span> ` +
                        `<span class="ratio ${summaryCls}">${nCorrect}/${nTotal}</span> ` +
                        `<span class="dots">${dotRow(g.rollouts)}</span>` +
                    `</div>` +
                    `<div class="meta">${escapeHtml(preview)}</div>` +
                `</div>`;
            div.onclick = () => { state.selectedGroupIdx = gi; render(); };
            list.appendChild(div);
        });
    }
    document.getElementById('fileInfo').textContent =
        `${FILE_INFO} · showing ${idxs.length} / ${GROUPS.length} prompts`;
}

function renderGroupDetail(g, gi) {
    const { nCorrect, nTotal } = groupStats(g);
    const summaryCls = nCorrect === nTotal ? 'correct' :
                        nCorrect === 0 ? 'incorrect' : 'warn';

    const metaBar = document.getElementById('metadataBar');
    const sampleStep = g.rollouts.find(r => r.step !== undefined);
    const cells = [];
    cells.push(`<div class="meta-cell"><span class="meta-label">Prompt</span><span class="meta-value">P${gi}</span></div>`);
    if (sampleStep) cells.push(`<div class="meta-cell"><span class="meta-label">Step</span><span class="meta-value">${sampleStep.step}</span></div>`);
    cells.push(`<div class="meta-cell"><span class="meta-label">Score</span><span class="meta-value ${summaryCls}">${nCorrect} / ${nTotal} correct</span></div>`);
    cells.push(`<div class="meta-cell"><span class="meta-label">GT</span><span class="meta-value">${escapeHtml(String(g.rollouts[0].gts ?? ''))}</span></div>`);
    metaBar.innerHTML = cells.join('');

    let html = `<div class="section" id="prompt-section">
        <div class="section-header">Prompt</div>
        <div class="text-box muted">${escapeHtml(g.input || '')}</div>
    </div>`;

    // Jump-to bar — sticky row of buttons, one per rollout, colored by outcome.
    const jumpButtons = g.rollouts.map((rec, k) => {
        const c = isCorrect(rec);
        const cls = c === true ? 'correct' : c === false ? 'incorrect' : '';
        const mark = c === true ? '✓' : c === false ? '✗' : '?';
        return `<button data-rollout="${k}" class="${cls}" title="Pred: ${escapeHtml(String(rec.pred ?? ''))}">R${k + 1} ${mark}</button>`;
    }).join('');
    html += `<div class="jump-bar">
        <span class="jump-label">Jump to</span>
        <button data-rollout="prompt">↑ Prompt</button>
        ${jumpButtons}
    </div>`;

    g.rollouts.forEach((rec, k) => {
        const correct = isCorrect(rec);
        const predCls = correct === true ? 'correct' : correct === false ? 'incorrect' : '';
        const flagBits = [];
        if (rec.truncated) flagBits.push('truncated');
        if (rec.incorrect_format) flagBits.push('bad format');
        if (rec.truncated_and_missing_answer) flagBits.push('no answer');

        html += `<div class="rollout-card" id="rollout-${k}">
            <div class="rollout-head">
                <div>
                    <span class="rollout-idx">Rollout ${k + 1} <span style="color:#8b949e;">(#${g.indices[k]})</span></span>
                    <span class="rollout-pred">Pred: <span class="${predCls}">${escapeHtml(String(rec.pred ?? ''))}</span></span>
                    <span class="rollout-result ${predCls}">${correct === true ? '✓' : correct === false ? '✗' : '?'}</span>
                </div>
                ${flagBits.length ? `<div class="rollout-flags">[${flagBits.join(', ')}]</div>` : ''}
            </div>
            <div class="text-box">${escapeHtml(rec.output || '')}</div>
            ${rec.feedback ? `<div class="text-box muted" style="margin-top:6px;">${escapeHtml(rec.feedback)}</div>` : ''}
        </div>`;
    });

    const contentEl = document.getElementById('content');
    contentEl.innerHTML = html;

    // Wire up jump-bar buttons.
    contentEl.querySelectorAll('.jump-bar button').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.rollout === 'prompt'
                ? document.getElementById('prompt-section')
                : document.getElementById(`rollout-${btn.dataset.rollout}`);
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
}

function renderCompare() {
    document.getElementById('metadataBar').innerHTML = '';
    document.getElementById('metadataBar').style.display = 'none';

    const idxs = filteredGroupIndices();
    if (idxs.length === 0) {
        document.getElementById('content').innerHTML =
            '<div class="empty-state">No prompts match the current filters.</div>';
        return;
    }
    const hasStep = RECORDS.some(r => r.step !== undefined);
    const maxN = Math.max(...idxs.map(gi => GROUPS[gi].rollouts.length));

    const headerPredCells = [];
    for (let k = 0; k < maxN; k++) headerPredCells.push(`<th>P${k + 1}</th>`);

    const rows = idxs.map(gi => {
        const g = GROUPS[gi];
        const { nCorrect, nTotal } = groupStats(g);
        const step = g.rollouts[0].step;
        const summaryCls = nCorrect === nTotal ? 'correct' :
                            nCorrect === 0 ? 'incorrect' : '';
        const predCells = [];
        for (let k = 0; k < maxN; k++) {
            const r = g.rollouts[k];
            if (!r) { predCells.push('<td></td>'); continue; }
            const c = isCorrect(r);
            const cls = c === true ? 'correct' : c === false ? 'incorrect' : '';
            predCells.push(`<td class="pred ${cls}" title="${escapeHtml(String(r.pred ?? ''))}">${escapeHtml(String(r.pred ?? ''))}</td>`);
        }
        return `<tr class="row ${gi === state.selectedGroupIdx ? 'active' : ''}" data-gi="${gi}">` +
            `<td class="idx">P${gi}</td>` +
            (hasStep ? `<td>${step !== undefined ? step : ''}</td>` : '') +
            `<td class="gt">${escapeHtml(String(g.rollouts[0].gts ?? ''))}</td>` +
            predCells.join('') +
            `<td class="ratio ${summaryCls}">${nCorrect}/${nTotal}</td>` +
            `</tr>`;
    }).join('');

    document.getElementById('content').innerHTML =
        `<table class="compare-table">
            <thead><tr>
                <th>#</th>
                ${hasStep ? '<th>Step</th>' : ''}
                <th>GT</th>
                ${headerPredCells.join('')}
                <th>✓/N</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>`;

    document.querySelectorAll('.compare-table tr.row').forEach(tr => {
        tr.addEventListener('click', () => {
            state.selectedGroupIdx = parseInt(tr.dataset.gi);
            state.view = 'detail';
            document.querySelectorAll('#viewSelector button').forEach(b => {
                b.classList.toggle('active', b.dataset.val === 'detail');
            });
            render();
        });
    });
}

function render() {
    renderGroupList();
    if (state.view === 'compare') {
        renderCompare();
        return;
    }
    document.getElementById('metadataBar').style.display = '';
    if (state.selectedGroupIdx !== null && GROUPS[state.selectedGroupIdx]) {
        renderGroupDetail(GROUPS[state.selectedGroupIdx], state.selectedGroupIdx);
    } else {
        document.getElementById('content').innerHTML =
            '<div class="empty-state">Select a prompt from the sidebar</div>';
        document.getElementById('metadataBar').innerHTML = '';
    }
}

function setupBtnGroup(id, stateKey) {
    const group = document.getElementById(id);
    group.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
            group.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state[stateKey] = btn.dataset.val;
            const idxs = filteredGroupIndices();
            if (state.selectedGroupIdx !== null && !idxs.includes(state.selectedGroupIdx)) {
                state.selectedGroupIdx = idxs.length > 0 ? idxs[0] : null;
            }
            render();
        });
    });
}

setupBtnGroup('correctnessFilter', 'correctness');
setupBtnGroup('flagsFilter', 'flags');
setupBtnGroup('viewSelector', 'view');

const initialIdxs = filteredGroupIndices();
if (initialIdxs.length > 0) state.selectedGroupIdx = initialIdxs[0];
render();
</script>
</body>
</html>"""


def generate_html(records: list[dict], file_info: str, output_path: str) -> None:
    records_json = json.dumps(records)
    file_info_json = json.dumps(file_info)
    html_content = HTML_TEMPLATE.replace("__RECORDS_JSON__", records_json).replace(
        "__FILE_INFO_JSON__", file_info_json
    )
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"Written to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize rollout dumps with correct/incorrect filtering"
    )
    parser.add_argument("input", help="Path to JSONL rollout file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output HTML path (default: <input_stem>_viz.html)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = args.output or (str(input_path.with_suffix("")) + "_viz.html")

    print(f"Loading {input_path}...", file=sys.stderr)
    records = load_data(str(input_path))

    n_correct = sum(1 for r in records if r.get("acc", r.get("score", 0)) >= 0.5)
    n_prompts = len({r.get("input", "") for r in records})
    file_info = (
        f"{input_path.name} — {n_prompts} prompts, {len(records)} rollouts "
        f"({n_correct} correct / {len(records) - n_correct} incorrect)"
    )

    generate_html(records, file_info, output_path)
    print(f"Done — {len(records)} records.", file=sys.stderr)


if __name__ == "__main__":
    main()
