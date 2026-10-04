// EON mode in one figure, all from the model run in data/eon.json: the searches from the starting
// state, the energy along one process, the searches and steps at the start of the run, the first
// kinetic Monte Carlo step, and the energy of the current state against simulated time.

import { Figure, HERO_TYPE as TYPE, scale } from '../lib.mjs';
import { duration, energy, eon, events, LEVELS, rate, stateNumber } from '../landscape.mjs';
import { drawSurface } from '../surface.mjs';

const START = 'state-000000';
const searches = eon.attempts.filter((a) => a.state === START);
// The search that found the process to state 2.
const shown = searches.find((a) => a.status === 'new' && a.product === 'state-000002');
const process = eon.processes[shown.process];
const run = events();
const firstStep = run.find((e) => e.kind === 'step' && e.index === 0);
// The timeline ends after this many steps.
const STEPS_SHOWN = 3;
const timeline = run.slice(0, run.findIndex((e) => e.kind === 'step' && e.index === STEPS_SHOWN - 1) + 1);

// Every search from the starting state fits in this view.
const WIDE = { x0: -3.62, x1: -0.62, y0: -1.0, y1: 0.976 };
const PW = 545;
const PH = Math.round((PW * (WIDE.y1 - WIDE.y0)) / (WIDE.x1 - WIDE.x0));
const ROW_A = 78;
const ROW_B = ROW_A + PH + 34 + 106;
const ROW_C = ROW_B + 212 + 92;
const BOTTOM_H = 318;

/** Dark text on a light bar, white on a dark one. */
const on = (color) => {
  const [r, g, b] = [1, 3, 5].map((k) => parseInt(color.slice(k, k + 2), 16) / 255);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 ? '#1f2328' : '#ffffff';
};

function searchesPanel(fig, rect) {
  const t = fig.t;
  const { sx, sy } = drawSurface(fig, 'eon-mode-searches', rect, { domain: WIDE, field: energy, levels: LEVELS, rx: 8 });
  const at = ([x, y]) => [sx(x), sy(y)];
  const halo = fig.tint(t.contour, 0.18);
  const clip = fig.clipRect('eon-mode-climbs', rect.x, rect.y, rect.w, rect.h, 8);
  const trail = (points, color, o = {}) => {
    // Stop at the ring of the marker the path ends on.
    const drawn = points.map(at);
    const end = drawn.at(-1);
    const kept = drawn.slice(0, drawn.findLastIndex(([px, py]) => Math.hypot(px - end[0], py - end[1]) > 12) + 1);
    const last = kept.at(-1);
    const reach = Math.hypot(end[0] - last[0], end[1] - last[1]);
    kept.push([end[0] - ((end[0] - last[0]) / reach) * 9, end[1] - ((end[1] - last[1]) / reach) * 9]);
    fig.arrow(kept, { color, sw: o.sw ?? 2, size: 8, dash: o.dash });
  };

  // Every other search's climb, faint: green when it reached a saddle, red when it failed.
  fig.group({ clip }, () => {
    for (const search of searches) {
      if (search === shown) continue;
      const good = search.status !== 'failed';
      fig.path(Figure.polyline(search.climb.map(at)), { stroke: good ? t.refine : t.critical, sw: 1.4, join: 'round', opacity: good ? 0.55 : 0.45 });
    }
  });

  // One search in full: the push, the dimer climb to a saddle, and both relaxations.
  const [cx, cy] = at(eon.states[START].position);
  const [qx, qy] = at(shown.climb[0]);
  const reach = Math.hypot(qx - cx, qy - cy);
  fig.arrow([[cx + ((qx - cx) / reach) * 10, cy + ((qy - cy) / reach) * 10], [qx - ((qx - cx) / reach) * 8, qy - ((qy - cy) / reach) * 8]], { color: t.event, sw: 2.5, size: 9 });
  trail(shown.climb, t.refine, { sw: 3 });
  for (const side of shown.relax) trail(side, t.event, { dash: '2 5', sw: 2.25 });
  fig.circle(qx, qy, 6, { fill: t.bg, stroke: t.ink, sw: 1.75 });

  // The two saddles, with how many searches reached each, and the states either side.
  const counts = new Map();
  for (const search of searches) if (search.process) counts.set(search.process, (counts.get(search.process) ?? 0) + 1);
  for (const [id, count] of counts) {
    const p = eon.processes[id];
    const [px, py] = at(p.saddle);
    fig.saddle(px, py, 8);
    fig.text(px + 16, py + (p.saddle[1] > 0 ? -12 : 26), `×${count}`, { size: TYPE.small, weight: 700, halo });
  }
  for (const id of [START, 'state-000001', 'state-000002']) {
    const [px, py] = at(eon.states[id].position);
    fig.circle(px, py, 6.5, { fill: t.ink, stroke: t.bg, sw: 2 });
    fig.text(px + (id === START ? -2 : 0), py + 31, String(stateNumber(id)), { size: TYPE.body, weight: 700, anchor: 'middle', halo });
  }
  fig.glyph('check', cx - 24, cy - 2, 15, t.good);

  fig.keys(rect.x + 4, rect.y + rect.h + 34, [
    [(x, y) => fig.arrow([[x - 11, y], [x + 11, y]], { color: t.event, sw: 2.5, size: 8 }), 'Push'],
    [(x, y) => fig.line(x - 11, y, x + 11, y, { stroke: t.refine, sw: 3, cap: 'round' }), 'Climb'],
    [(x, y) => fig.line(x - 11, y, x + 11, y, { stroke: t.event, sw: 2.25, dash: '2 5', cap: 'round' }), 'Relax'],
    [(x, y) => fig.line(x - 11, y, x + 11, y, { stroke: t.critical, sw: 2, cap: 'round', opacity: 0.7 }), 'Failed'],
  ]);
}

function processPanel(fig, rect) {
  const t = fig.t;
  // Energy along the process: the two relaxations from the saddle, joined there.
  const path = [...shown.relax[0].slice().reverse(), ...shown.relax[1].slice(1)];
  const lengths = path.reduce((acc, p, i) => [...acc, i ? acc.at(-1) + Math.hypot(p[0] - path[i - 1][0], p[1] - path[i - 1][1]) : 0], []);
  const e0 = eon.states[START].energy_eV;
  const [left, right, top, bottom] = [rect.x + 92, rect.x + rect.w - 12, rect.y + 16, rect.y + rect.h - 70];
  const x = scale(0, lengths.at(-1), left, right);
  const y = scale(-0.45, 0.55, bottom, top);
  for (const tick of [-0.4, 0, 0.4]) {
    fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
    fig.text(left - 12, y(tick), tick === 0 ? '0' : tick.toFixed(1), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
  }
  fig.line(left, bottom, right, bottom, { stroke: t.axis });
  for (let tick = 0; tick <= lengths.at(-1); tick += 0.5) {
    fig.line(x(tick), bottom, x(tick), bottom + 6, { stroke: t.axis });
    fig.text(x(tick), bottom + 32, tick ? tick.toFixed(1) : '0', { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  }
  fig.text((left + right) / 2, bottom + 68, 'Path (Å)', { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  fig.text(rect.x + 16, (top + bottom) / 2, 'E − E(0) (eV)', { size: TYPE.small, anchor: 'middle', fill: t.ink2, rotate: -90 });

  const points = path.map((p, i) => [x(lengths[i]), y(energy(p[0], p[1]) - e0)]);
  fig.path(Figure.smooth(points, 0.35), { stroke: t.refine, sw: 2.75 });
  const peak = shown.relax[0].length - 1;
  for (const [i, label] of [[0, '0'], [points.length - 1, String(stateNumber(shown.product))]]) {
    const [px, py] = points[i];
    fig.circle(px, py, 6.5, { fill: t.ink, stroke: t.bg, sw: 2 });
    fig.text(px, py + 34, label, { size: TYPE.body, weight: 700, anchor: 'middle' });
  }
  fig.saddle(...points[peak], 9.5);

  // The barrier and prefactor the search returned.
  const bx = points[peak][0] + 34;
  fig.line(points[0][0] + 10, y(0), bx + 8, y(0), { stroke: t.ink3, sw: 1, dash: '4 4' });
  fig.line(points[peak][0] + 12, points[peak][1], bx + 8, points[peak][1], { stroke: t.ink3, sw: 1, dash: '4 4' });
  fig.arrow([[bx, y(0)], [bx, points[peak][1] + 1]], { color: t.ink, sw: 1.5, size: 8, both: true });
  const exponent = Math.floor(Math.log10(process.prefactor_s));
  const mantissa = (process.prefactor_s / 10 ** exponent).toFixed(1);
  // Below the curve on the left, which stays above zero until it falls toward state 2.
  fig.text(left + 22, y(-0.16), [{ t: 'ΔE‡ ', weight: 700 }, `${process.barrier_eV.toFixed(3)} eV`], { size: TYPE.body, middle: true });
  fig.text(left + 22, y(-0.31), [{ t: 'ν ', weight: 700 }, `${mantissa} × 10`, { t: String(exponent), sup: true }, ' s', { t: '−1', sup: true }], { size: TYPE.body, middle: true });
}

function timelinePanel(fig, top) {
  const t = fig.t;
  // Searches and steps fill [x0, x1]; the run continues to the right edge.
  const [x0, x1, edge] = [230, 1080, 1164];
  const lanes = [
    { y: top + 24, label: 'Searches', color: t.event },
    { y: top + 88, label: 'Confidence', color: t.ink3 },
    { y: top + 150, label: 'State', color: t.md },
  ];
  for (const lane of lanes) {
    fig.circle(44, lane.y, 6, { fill: lane.color });
    fig.text(58, lane.y, lane.label, { size: TYPE.body, weight: 650, middle: true });
  }
  fig.line(x0, lanes[0].y, edge, lanes[0].y, { stroke: t.grid });

  // Searches advance by one pitch; a step takes a fixed gap.
  const count = timeline.filter((e) => e.kind === 'search').length;
  const gap = 26;
  const pitch = (x1 - x0 - STEPS_SHOWN * gap) / count;
  let cursor = x0;
  const xs = timeline.map((e) => {
    const width = e.kind === 'search' ? pitch : gap;
    cursor += width;
    return cursor - width / 2;
  });
  const mark = (status, x, y) => {
    if (status === 'new') fig.circle(x, y, 6.5, { fill: t.refine });
    else if (status === 'repeat') fig.line(x, y - 8, x, y + 8, { stroke: t.ink3, sw: 1.75, cap: 'round' });
    else if (status === 'failed') fig.glyph('cross', x, y, 11, t.critical);
    else fig.circle(x, y, 5, { fill: t.warn });
  };

  // Legend on the heading line, right-aligned.
  const items = [
    [(x, y) => mark('new', x, y), 'New'],
    [(x, y) => mark('repeat', x, y), 'Repeat'],
    [(x, y) => mark('failed', x, y), 'Failed'],
    [(x, y) => fig.line(x, y - 12, x, y + 12, { stroke: t.ink3, sw: 1.25, dash: '4 4' }), 'KMC step'],
  ];
  const width = items.reduce((sum, [, label]) => sum + 30 + fig.measure(label, { size: TYPE.small }) + 32, -32);
  fig.keys(1164 - width, top - 46, items);

  // Step boundaries, with the clock's advance under each.
  const sy = lanes[2].y;
  timeline.forEach((e, i) => {
    if (e.kind !== 'step') return;
    fig.line(xs[i], lanes[0].y - 22, xs[i], sy + 22, { stroke: t.ink3, sw: 1.25, dash: '4 4' });
    fig.text(xs[i], sy + 52, `+${duration(e.step.dt_s)}`, { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  });
  timeline.forEach((e, i) => e.kind === 'search' && mark(e.attempt.status, xs[i], lanes[0].y));

  // Confidence of the current state after each search, on a 0..1 band.
  const [cTop, cBottom] = [lanes[1].y - 22, lanes[1].y + 22];
  const cy = (v) => cBottom - v * (cBottom - cTop);
  const target = eon.akmc.confidence;
  fig.line(x0, cy(target), x1, cy(target), { stroke: t.md, sw: 1.25, dash: '5 4' });
  fig.text(x1 + 12, cy(target), String(target), { size: TYPE.small, weight: 700, middle: true, fill: t.md });
  fig.line(x0, cBottom, x1, cBottom, { stroke: t.grid });
  let segment = [];
  const flush = () => {
    if (segment.length > 1) fig.path(Figure.polyline(segment), { stroke: t.ink2, sw: 2, join: 'round' });
    segment = [];
  };
  timeline.forEach((e, i) => (e.kind === 'step' ? flush() : segment.push([xs[i], cy(e.confidence)])));
  flush();

  // The current state as a bar per visit, then the run continues.
  const bar = 32;
  let from = x0;
  timeline.forEach((e, i) => {
    if (e.kind !== 'step') return;
    const to = xs[i] - gap / 2 + 2;
    fig.rect(from, sy - bar / 2, to - from, bar, { rx: 7, fill: t.md });
    fig.text((from + to) / 2, sy, `State ${stateNumber(e.step.from)}`, { size: TYPE.small, weight: 650, anchor: 'middle', middle: true, fill: '#ffffff' });
    from = xs[i] + gap / 2 - 2;
  });
  fig.path(`M${from + 7} ${sy - bar / 2} H${edge - 14} L${edge} ${sy} L${edge - 14} ${sy + bar / 2} H${from + 7} a7 7 0 0 1 -7 -7 v${-(bar - 14)} a7 7 0 0 1 7 -7 z`, { fill: t.md, opacity: 0.35 });
}

function stepPanel(fig, rect) {
  const t = fig.t;
  const { window } = firstStep;
  const total = firstStep.step.total_rate_s;
  const [pick] = firstStep.step.random;
  const colors = window.map((_, i) => t.ramp[[1, 2, 0][i % 3]]);

  // The pick: a random number on the summed rates, each process as wide as its rate.
  const [pl, pr, py] = [rect.x + 4, rect.x + rect.w - 4, rect.y + 66];
  const share = scale(0, total, pl, pr);
  let acc = 0;
  window.forEach((p, i) => {
    const [a, b] = [share(acc), share(acc + rate(p))];
    fig.rect(a + (i ? 2 : 0), py - 24, b - a - (i ? 2 : 0), 48, { rx: 6, fill: colors[i] });
    fig.text((a + b) / 2, py, `→ ${stateNumber(p.product)}`, { size: TYPE.body, weight: 700, anchor: 'middle', middle: true, fill: on(colors[i]) });
    fig.text((a + b) / 2, py + 56, `${p.barrier_eV.toFixed(3)} eV`, { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    acc += rate(p);
  });
  const ux = share(pick * total);
  fig.line(ux, py - 54, ux, py - 28, { stroke: t.ink, sw: 2 });
  fig.head(ux, py - 26, Math.PI / 2, { color: t.ink, size: 10 });
  fig.text(ux + 10, py - 44, `u = ${pick.toFixed(2)}`, { size: TYPE.small, weight: 650 });

  // The clock: an exponential waiting time with mean 1/Σk.
  const [cl, cr, ct, cb] = [rect.x + 4, rect.x + rect.w - 4, rect.y + 162, rect.y + rect.h - 70];
  const mean = 1 / total;
  const tx = scale(0, 4 * mean, cl, cr);
  const ty = scale(0, 1, cb, ct);
  const density = Array.from({ length: 81 }, (_, i) => [tx((4 * mean * i) / 80), ty(Math.exp(-i / 20))]);
  fig.path(Figure.polyline([[cl, cb], ...density, [cr, cb]]) + ' Z', { fill: fig.tint(t.md, 0.14) });
  fig.path(Figure.polyline(density), { stroke: t.md, sw: 2.25, join: 'round' });
  fig.line(cl, cb, cr, cb, { stroke: t.axis });
  const dt = firstStep.step.dt_s;
  const [dx, dy] = [tx(dt), ty(Math.exp(-dt / mean))];
  fig.line(dx, cb, dx, dy, { stroke: t.ink, sw: 2 });
  fig.circle(dx, dy, 6, { fill: t.ink, stroke: t.bg, sw: 2 });
  fig.text(dx + 14, dy - 14, `Δt = ${duration(dt)}`, { size: TYPE.body, weight: 700, halo: true });
  for (const k of [0, 1, 2, 3]) {
    fig.line(tx(k * mean), cb, tx(k * mean), cb + 6, { stroke: t.axis });
    fig.text(tx(k * mean), cb + 32, k === 0 ? '0' : k === 1 ? '1/Σk' : `${k}/Σk`, { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  }
  fig.text(cr, cb + 68, 'Waiting time', { size: TYPE.small, anchor: 'end', fill: t.ink2 });
}

function timePanel(fig, rect) {
  const t = fig.t;
  const [left, right, top, bottom] = [rect.x + 92, rect.x + rect.w - 12, rect.y + 16, rect.y + rect.h - 70];
  const times = eon.steps.map((s) => s.time_s);
  const [d0, d1] = [Math.floor(Math.log10(times[0])) - 1, Math.ceil(Math.log10(times.at(-1)))];
  const x = scale(d0, d1, left, right);
  const e0 = eon.states[START].energy_eV;
  const visited = [START, ...eon.steps.map((s) => s.to)];
  const energies = visited.map((id) => eon.states[id].energy_eV - e0);
  const lo = Math.floor(Math.min(...energies) * 4) / 4;
  const y = scale(lo, 0.25, bottom, top);
  for (let v = 0; v >= lo; v -= 0.5) {
    fig.line(left, y(v), right, y(v), { stroke: t.grid });
    fig.text(left - 12, y(v), v === 0 ? '0' : v.toFixed(1), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
  }
  fig.line(left, bottom, right, bottom, { stroke: t.axis });
  const labels = { '-6': '1 µs', '-3': '1 ms', 0: '1 s' };
  for (let d = d0; d <= d1; d++) {
    fig.line(x(d), bottom, x(d), bottom + 6, { stroke: t.axis });
    if (labels[d]) fig.text(x(d), bottom + 32, labels[d], { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  }
  fig.text((left + right) / 2, bottom + 68, 'Simulated time', { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  fig.text(rect.x + 16, (top + bottom) / 2, 'E − E(0) (eV)', { size: TYPE.small, anchor: 'middle', fill: t.ink2, rotate: -90 });

  const stairs = [[x(d0), y(energies[0])]];
  times.forEach((time, i) => stairs.push([x(Math.log10(time)), y(energies[i])], [x(Math.log10(time)), y(energies[i + 1])]));
  stairs.push([right, y(energies.at(-1))]);
  fig.path(Figure.polyline(stairs), { stroke: t.md, sw: 2.5, join: 'round' });

  // The state on each step of the descent; after state 6 the run only trades 6 and 4. A plateau
  // too short to hold its number gets it just past the step that ends it.
  const plateaus = [[x(d0), 0], ...times.slice(0, 4).map((time, i) => [x(Math.log10(time)), i + 1])];
  plateaus.forEach(([px, i], k) => {
    const next = k < 4 ? x(Math.log10(times[k])) : right;
    const label = String(stateNumber(visited[i]));
    if (next - px < 12) fig.text(next + 10, y(energies[i]) - 4, label, { size: TYPE.small, weight: 700, halo: true });
    else fig.text((px + Math.min(next, px + 60)) / 2 + (k ? 6 : 0), y(energies[i]) - 14, label, { size: TYPE.small, weight: 700, anchor: 'middle', halo: true });
  });
  fig.text(right, top + 10, `${eon.steps.length} steps, ${Math.round(times.at(-1))} s`, { size: TYPE.body, weight: 700, anchor: 'end', halo: true });
}

export default {
  name: 'eon-mode',
  title: 'EON mode',
  label: 'EON mode',
  alt:
    'Five panels on the EON model run. a, every search from the starting state on the model landscape, with one ' +
    'search in full. It pushes away from state 0, climbs to a saddle with the dimer method, and relaxes to state 0 ' +
    'and state 2. Thirteen searches reach one saddle, eleven reach the other, and the rest fail. b, the energy along ' +
    'that process with a barrier of 0.437 eV and a prefactor of 2.7 × 10¹³ per second. c, searches in the first ' +
    'three states, each a new process, a repeat, or a failure. The confidence rises with each repeat in a row and a ' +
    'kinetic Monte Carlo step follows when it reaches 0.95. d, the first step. Its two processes share the summed ' +
    'rate, a random number picks the one to state 1, and the clock advances by 350 ns drawn from an exponential ' +
    'distribution. e, the energy of the current state against simulated time, 12 steps over 96 s.',
  caption:
    'ReactionFlow in EON mode with default settings, 12 steps at 300 K, on a two-dimensional model landscape. ' +
    'Every number comes from that run.',
  width: 1200,
  height: ROW_C + BOTTOM_H + 24,
  draw(fig) {
    fig.heading(36, ROW_A - 26, 'a', 'Searches from state 0');
    fig.heading(619, ROW_A - 26, 'b', 'One process');
    searchesPanel(fig, { x: 36, y: ROW_A, w: PW, h: PH });
    processPanel(fig, { x: 619, y: ROW_A, w: PW, h: PH + 34 });
    fig.heading(36, ROW_B - 36, 'c', 'Search until confident');
    timelinePanel(fig, ROW_B);
    fig.heading(36, ROW_C - 26, 'd', 'Kinetic Monte Carlo step');
    fig.heading(619, ROW_C - 26, 'e', 'Simulated time');
    stepPanel(fig, { x: 36, y: ROW_C, w: PW, h: BOTTOM_H });
    timePanel(fig, { x: 619, y: ROW_C, w: PW, h: BOTTOM_H });
  },
};
