// MD mode in one figure: a bond change confirmed by the detector (data/detection.json), the
// refinement stages and barrier of a refine_pathway() run on the model surface (data/pathway.json),
// and how occurrences of the same change fall into one reaction class.

import { readFileSync } from 'node:fs';

import { Figure, HERO_TYPE as TYPE, scale } from '../lib.mjs';
import { DOMAIN, drawSurface, pathway } from '../surface.mjs';

const detection = JSON.parse(readFileSync(new URL('../../data/detection.json', import.meta.url)));
const [relaxR, relaxP, neb, climb, sideA, sideB] = pathway.runs.map((run) => run.history);
const saddleIndex = pathway.frequency.saddle_index;
const mode = pathway.frequency.primary_mode[1];
const imaginary = Math.abs(Math.min(...pathway.frequency.frequencies_cm1));

// Refinement stages: the default view with room above the arch for the stage titles.
const STAGE_DOMAIN = { ...DOMAIN, y1: DOMAIN.y1 + 0.42 };
const [STAGE_W, STAGE_GAP] = [270, 16];
const STAGE_H = Math.round((STAGE_W * (STAGE_DOMAIN.y1 - STAGE_DOMAIN.y0)) / (STAGE_DOMAIN.x1 - STAGE_DOMAIN.x0));

// Rows: the top of each panel's content, below its heading.
const PLOT_H = 206;
const ROW_A = 92;
const ROW_B = ROW_A + PLOT_H + 70 + 92;
const ROW_C = ROW_B + STAGE_H + 34 + 98;
const BOTTOM_H = 332;

function bondDetection(fig, top) {
  const t = fig.t;
  const [left, right, bottom] = [128, 1000, top + PLOT_H];
  const frames = detection.frames;
  const x = scale(-0.6, frames.length - 0.4, left, right);
  const y = scale(0.88, 1.72, bottom, top);
  const [form, brk] = [detection.form_scale, detection.break_scale];
  const formed = detection.events[0].confirmed_frame;
  const candidate = detection.candidates[0].observed_frame;

  for (const tick of [1.0, 1.2, 1.4, 1.6]) {
    fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
    fig.text(left - 12, y(tick), tick.toFixed(1), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
  }
  fig.line(left, bottom, right, bottom, { stroke: t.axis });
  for (const f of frames) {
    fig.line(x(f.frame), bottom, x(f.frame), bottom + 6, { stroke: t.axis });
    if (f.frame % 2 === 0) fig.text(x(f.frame), bottom + 32, String(f.frame), { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  }
  fig.text((left + right) / 2, bottom + 68, 'Observation frame', { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  fig.text(50, (top + bottom) / 2, ['Distance / (r', { t: 'i', sub: true }, ' + r', { t: 'j', sub: true }, ')'], {
    size: TYPE.small,
    anchor: 'middle',
    fill: t.ink2,
    rotate: -90,
  });

  // The hysteresis gap between the two thresholds.
  fig.rect(left, y(brk), right - left, y(form) - y(brk), { fill: t.panel2 });
  for (const [value, label] of [[brk, `Break ≥ ${brk.toFixed(2)}`], [form, `Form ≤ ${form.toFixed(2)}`]]) {
    fig.line(left, y(value), right, y(value), { stroke: t.ink3, sw: 1.25, dash: '6 5' });
    fig.text(right + 12, y(value), label, { size: TYPE.small, weight: 650, middle: true });
  }
  fig.text(left + 12, y((form + brk) / 2), 'Hysteresis', { size: TYPE.small, fill: t.ink2, middle: true });

  // Trace, then the markers: hollow before the bond, orange while a crossing is pending, filled once bonded.
  fig.path(Figure.polyline(frames.map((f) => [x(f.frame), y(f.ratio)])), { stroke: t.md, sw: 2.25, join: 'round', cap: 'round' });
  for (const f of frames) {
    const [cx, cy] = [x(f.frame), y(f.ratio)];
    fig.circle(cx, cy, 8.5, { fill: t.bg });
    if (f.persistence_count) fig.circle(cx, cy, 6.5, { fill: t.event });
    else if (f.bonded) fig.circle(cx, cy, 6.5, { fill: t.md });
    else fig.circle(cx, cy, 5.25, { fill: t.bg, stroke: t.md, sw: 2.25 });
  }

  // The frame that confirms the bond and the frame that makes it a candidate, with leaders from above.
  const note = (frame, height, label, anchor) => {
    const [cx, cy] = [x(frame), y(frames[frame].ratio)];
    fig.line(cx, y(height) + 8, cx, cy - 11, { stroke: t.ink3, sw: 1 });
    fig.text(cx + (anchor === 'end' ? 6 : -6), y(height), label, { size: TYPE.small, weight: 650, anchor, halo: true });
  };
  note(formed, 1.47, 'Bond formed', 'end');
  note(candidate, 1.47, 'Candidate', 'start');
  fig.text(x(5) - 14, y(frames[5].ratio) + 32, 'No event', { size: TYPE.small, fill: t.ink2, anchor: 'end', halo: true });

  // Legend on the heading line, right-aligned.
  const items = [
    [(cx, cy) => fig.circle(cx, cy, 5.25, { fill: t.bg, stroke: t.md, sw: 2.25 }), 'Not bonded'],
    [(cx, cy) => fig.circle(cx, cy, 6.5, { fill: t.event }), 'Pending'],
    [(cx, cy) => fig.circle(cx, cy, 6.5, { fill: t.md }), 'Bonded'],
  ];
  const width = items.reduce((sum, [, label]) => sum + 30 + fig.measure(label, { size: TYPE.small }) + 32, -32);
  fig.keys(1164 - width, top - 51, items);
}

function refinementStages(fig, top) {
  const t = fig.t;
  const [w, h, gap, domain] = [STAGE_W, STAGE_H, STAGE_GAP, STAGE_DOMAIN];
  const titles = ['Relax', 'NEB', 'CI-NEB', 'Checks'];
  const halo = fig.tint(t.contour, 0.18);

  titles.forEach((title, index) => {
    const rect = { x: 36 + index * (w + gap), y: top, w, h };
    const { sx, sy } = drawSurface(fig, 'md-mode-stage', rect, { domain, rx: 8 });
    const at = ([px, py]) => [sx(px), sy(py)];
    const dot = (p, r, fill) => fig.circle(...at(p), r, { fill, stroke: t.bg, sw: 2 });
    const hollow = (p, r, stroke) => fig.circle(...at(p), r, { fill: t.bg, stroke, sw: 1.75 });
    const diamond = (p, r) => {
      const [cx, cy] = at(p);
      fig.polygon([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], { fill: t.bg, stroke: t.ink, sw: 1.75, join: 'round' });
    };
    const trail = (history, color, o = {}) => {
      // Follow the optimizer's steps, stopping at the ring of the marker the path ends on.
      const points = history.map((step) => at(step[0]));
      const end = points.at(-1);
      const shown = points.slice(0, points.findLastIndex(([px, py]) => Math.hypot(px - end[0], py - end[1]) > 14) + 1);
      const last = shown.at(-1);
      const reach = Math.hypot(end[0] - last[0], end[1] - last[1]);
      shown.push([end[0] - ((end[0] - last[0]) / reach) * 8, end[1] - ((end[1] - last[1]) / reach) * 8]);
      fig.arrow(shown, { color, sw: 1.75, size: 7, dash: o.dash });
    };
    const band = (images, color, o = {}) =>
      fig.path(Figure.polyline(images.map(at)), { stroke: color, sw: o.sw ?? 2.25, join: 'round', cap: 'round', dash: o.dash, opacity: o.opacity });
    const endpoints = () => {
      for (const [p, label, dx] of [[pathway.images[0], 'R', -19], [pathway.images.at(-1), 'P', 19]]) {
        dot(p, 5.5, t.ink);
        const [cx, cy] = at(p);
        fig.text(cx + dx, cy + 22, label, { size: TYPE.body, weight: 700, anchor: 'middle', halo });
      }
    };

    if (index === 0) {
      trail(relaxR, t.ink);
      trail(relaxP, t.ink);
      diamond(pathway.snapshots.reactant, 6);
      diamond(pathway.snapshots.product, 6);
      endpoints();
    }
    if (index === 1) {
      for (const step of [5, 10, 16]) band(neb[step], t.refine, { sw: 1.25, opacity: 0.45 });
      band(neb[0], t.ink2, { sw: 1.5, dash: '5 4' });
      neb[0].slice(1, -1).forEach((p) => hollow(p, 3.5, t.ink2));
      band(neb.at(-1), t.refine);
      neb.at(-1).slice(1, -1).forEach((p) => dot(p, 4.5, t.refine));
      endpoints();
    }
    if (index === 2) {
      band(climb.at(-1), t.refine);
      climb.at(-1).slice(1, -1).forEach((p, i) => i + 1 !== saddleIndex && dot(p, 4.5, t.refine));
      hollow(climb[0][saddleIndex], 4.5, t.refine);
      fig.saddle(...at(climb.at(-1)[saddleIndex]), 8);
      const [from, to] = [at(climb[0][saddleIndex]), at(climb.at(-1)[saddleIndex])];
      fig.curve([from[0], from[1] - 8], [from[0] - 2, from[1] - 26], [to[0] + 6, to[1] - 30], [to[0] + 1, to[1] - 13], { color: t.ink, sw: 1.5, size: 6 });
      endpoints();
    }
    if (index === 3) {
      band(climb.at(-1), t.refine, { sw: 1.5, opacity: 0.4 });
      trail(sideA, t.event, { dash: '2 5' });
      trail(sideB, t.event, { dash: '2 5' });
      const saddle = at(pathway.images[saddleIndex]);
      const reach = 0.32 * (w / (domain.x1 - domain.x0));
      fig.arrow(
        [[saddle[0] - mode[0] * reach, saddle[1] + mode[1] * reach], [saddle[0] + mode[0] * reach, saddle[1] - mode[1] * reach]],
        { color: t.event, sw: 2.25, size: 8, both: true },
      );
      fig.saddle(...saddle, 8);
      endpoints();
      // Midway between the endpoints, in the open space under the arch.
      const middle = (at(pathway.images[0])[0] + at(pathway.images.at(-1))[0]) / 2;
      fig.text(middle, saddle[1] + 64, `${imaginary.toFixed(0)}i cm⁻¹`, { size: TYPE.small, weight: 650, anchor: 'middle', halo });
    }

    fig.badge(rect.x + 26, rect.y + 26, index + 1, { r: 13 });
    fig.text(rect.x + 47, rect.y + 26, title, { size: TYPE.body, weight: 700, middle: true, halo });
  });

  const ky = top + h + 34;
  fig.keys(40, ky, [
    [(x, y) => fig.polygon([[x, y - 6], [x + 6, y], [x, y + 6], [x - 6, y]], { fill: t.bg, stroke: t.ink, sw: 1.75, join: 'round' }), 'MD snapshot'],
    [(x, y) => { fig.line(x - 11, y, x + 11, y, { stroke: t.ink2, sw: 1.5, dash: '5 4' }); fig.circle(x, y, 3.5, { fill: t.bg, stroke: t.ink2, sw: 1.75 }); }, 'Interpolated'],
    [(x, y) => { fig.line(x - 11, y, x + 11, y, { stroke: t.refine, sw: 2.25 }); fig.circle(x, y, 4.5, { fill: t.refine }); }, 'Converged band'],
    [(x, y) => fig.saddle(x, y, 7, { sw: 0 }), 'Saddle'],
    [(x, y) => fig.arrow([[x - 11, y], [x + 11, y]], { color: t.event, sw: 2, size: 6, both: true }), 'Imaginary mode'],
  ]);
}

function barrier(fig, rect) {
  const t = fig.t;
  const [left, right, top, bottom] = [rect.x + 90, rect.x + rect.w - 8, rect.y + 10, rect.y + rect.h - 72];
  const x = scale(-0.35, 6.35, left, right);
  const y = scale(-1.2, 0.8, bottom, top);
  for (const tick of [-1, -0.5, 0, 0.5]) {
    fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
    fig.text(left - 12, y(tick), tick === 0 ? '0' : tick.toFixed(1), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
  }
  fig.line(left, bottom, right, bottom, { stroke: t.axis });
  ['R', '1', '2', '3', '4', '5', 'P'].forEach((label, i) =>
    fig.text(x(i), bottom + 32, label, { size: TYPE.small, anchor: 'middle', fill: t.ink2, weight: i % 6 ? 400 : 700 }),
  );
  fig.text((left + right) / 2, bottom + 68, 'Image', { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  fig.text(rect.x + 16, (top + bottom) / 2, 'E − E(R) (eV)', { size: TYPE.small, anchor: 'middle', fill: t.ink2, rotate: -90 });
  const zero = pathway.energies_eV[0];
  const points = pathway.energies_eV.map((e, i) => [x(i), y(e - zero)]);
  fig.path(Figure.polyline(points), { stroke: t.refine, sw: 2.75, join: 'round' });
  points.forEach(([px, py], i) => {
    if (i === saddleIndex) fig.saddle(px, py, 9.5);
    else fig.circle(px, py, i % 6 ? 5.5 : 6, { fill: i % 6 ? t.refine : t.ink, stroke: t.bg, sw: 2 });
  });
  // The barrier, drawn clear of the falling side of the band.
  const bx = x(4.3);
  fig.line(x(0) + 10, y(0), bx + 8, y(0), { stroke: t.ink3, sw: 1, dash: '4 4' });
  fig.line(points[saddleIndex][0] + 12, points[saddleIndex][1], bx + 8, points[saddleIndex][1], { stroke: t.ink3, sw: 1, dash: '4 4' });
  fig.arrow([[bx, y(0)], [bx, points[saddleIndex][1] + 1]], { color: t.ink, sw: 1.5, size: 8, both: true });
  fig.text(bx + 12, y(pathway.barrier_eV / 2) - 15, 'ΔE‡', { size: TYPE.body, weight: 700, middle: true });
  fig.text(bx + 12, y(pathway.barrier_eV / 2) + 17, `${pathway.barrier_eV.toFixed(2)} eV`, { size: TYPE.body, weight: 700, middle: true });
}

/**
 * Change graph of acetonitrile <-> ketenimine (a hydrogen moves from the methyl carbon to nitrogen).
 * variant: 'forward' | 'reverse' | 'loss' (only the C-H bond breaks); bend and flip redraw the
 * same graph with a different geometry.
 */
function changeGraph(fig, x, y, s, { variant = 'forward', bend = 0, flip = false, ids = null } = {}) {
  const t = fig.t;
  const arm = (length, degrees) => [length * Math.cos((degrees * Math.PI) / 180), length * Math.sin((degrees * Math.PI) / 180)];
  const c2 = [58, 0];
  const n = [c2[0] + arm(58, bend)[0], c2[1] + arm(58, bend)[1]];
  const moving = [n[0] / 2, flip ? 52 : -54];
  const nodes = [
    ['C', [0, 0]], ['C', c2], ['N', n], ['H', [-36, -36]], ['H', [-36, 36]], ['H', moving],
  ].map(([el, [px, py]]) => [el, [x + px * s, y + py * s]]);
  const edges = [
    [0, 1, 'same'], [1, 2, 'same'], [0, 3, 'same'], [0, 4, 'same'],
    [0, 5, variant === 'reverse' ? 'formed' : 'broken'],
    ...(variant === 'loss' ? [] : [[5, 2, variant === 'reverse' ? 'broken' : 'formed']]),
  ];
  for (const [a, b, kind] of edges) {
    const [p, q] = [nodes[a][1], nodes[b][1]];
    const style = {
      same: { stroke: t.ink3, sw: 2 * s },
      formed: { stroke: t.refine, sw: 4.5 * s },
      broken: { stroke: t.event, sw: 4.5 * s, dash: `${6 * s} ${5 * s}` },
    }[kind];
    fig.line(...p, ...q, { ...style, cap: kind === 'broken' ? 'butt' : 'round' });
  }
  nodes.forEach(([el, [px, py]], i) => {
    const r = (el === 'H' ? 11 : 14) * s;
    fig.circle(px, py, r, { fill: t.bg, stroke: t.ink2, sw: 1.5 });
    fig.text(px, py, el, { size: (el === 'H' ? 14 : 17) * s, weight: 700, anchor: 'middle', middle: true });
    if (ids) {
      const [below, side] = [r + 14, r + 6];
      const [dx, dy, anchor] = [[0, below, 'middle'], [0, below, 'middle'], [0, below, 'middle'], [-side, 0, 'end'], [-side, 0, 'end'], [side, -4, 'start']][i];
      fig.text(px + dx, py + dy, String(ids[i]), { size: 19, mono: true, fill: t.ink3, anchor, middle: true });
    }
  });
}

function reactionClasses(fig, rect) {
  const t = fig.t;
  const graph = { x: rect.x, y: rect.y, w: 216, h: rect.h };
  fig.card(graph.x, graph.y, graph.w, graph.h, { fill: t.panel });
  changeGraph(fig, graph.x + 80, graph.y + 120, 0.98, { ids: [17, 18, 19, 40, 41, 42] });
  [
    [{ stroke: t.ink3, sw: 2.5 }, 'Unchanged'],
    [{ stroke: t.refine, sw: 5 }, 'Formed'],
    [{ stroke: t.event, sw: 5, dash: '7 5' }, 'Broken'],
  ].forEach(([style, label], i) => {
    const ly = graph.y + graph.h - 92 + i * 32;
    fig.line(graph.x + 20, ly, graph.x + 52, ly, { ...style, cap: style.dash ? 'butt' : 'round' });
    fig.text(graph.x + 64, ly, label, { size: TYPE.small, fill: t.ink2, middle: true });
  });

  // The same reaction whatever the numbering, direction, or geometry; a different change is not.
  const cells = [
    { label: 'Renumbered', same: true, dx: 90, options: { ids: [23, 9, 77, 31, 150, 64] } },
    { label: 'Reversed', same: true, options: { variant: 'reverse' } },
    { label: 'New geometry', same: true, dy: 54, options: { bend: -30, flip: true } },
    { label: 'Other change', same: false, options: { variant: 'loss' } },
  ];
  const [cx0, gap] = [graph.x + graph.w + 12, 12];
  const [cw, ch] = [(rect.x + rect.w - cx0 - gap) / 2, (rect.h - gap) / 2];
  cells.forEach((cell, i) => {
    const [cx, cy] = [cx0 + (i % 2) * (cw + gap), rect.y + Math.floor(i / 2) * (ch + gap)];
    const color = cell.same ? t.good : t.critical;
    fig.card(cx, cy, cw, ch, { fill: t.bg });
    changeGraph(fig, cx + (cell.dx ?? 72), cy + (cell.dy ?? 68), 0.88, cell.options);
    fig.text(cx + 12, cy + ch - 18, cell.label, { size: TYPE.small, fill: t.ink2 });
    fig.circle(cx + cw - 24, cy + ch - 27, 14, { fill: fig.tint(color, 0.16), stroke: color, sw: 1.25 });
    fig.text(cx + cw - 24, cy + ch - 27, cell.same ? '=' : '≠', { size: TYPE.body, weight: 700, anchor: 'middle', middle: true });
  });
}

export default {
  name: 'md-mode',
  title: 'MD mode',
  label: 'MD mode',
  alt:
    'Four panels on MD mode. a, the distance of one atom pair over 22 observations in units of the summed covalent ' +
    'radii. A single frame below the formation threshold of 1.15 is ignored, three frames in a row form the bond, ' +
    'and three stable frames after that make a reaction candidate. b, refinement on a model energy surface in four ' +
    'stages, endpoint relaxation, NEB, climbing image, and saddle checks with a 476i cm⁻¹ mode. c, the energy of ' +
    'each image with a barrier of 0.62 eV. d, a reaction stored as a change graph, which matches renumbered, ' +
    'reversed, and reshaped copies of itself but not a different change.',
  caption:
    'a, one C–N pair passed through <code>BondChangeDetector</code> and <code>ReactionTracker</code> with the default ' +
    'settings. b and c, a <code>refine_pathway()</code> run with default settings on a two-dimensional model ' +
    'potential. d, the hydrogen shift between acetonitrile and ketenimine as a change graph.',
  width: 1200,
  height: ROW_C + BOTTOM_H + 24,
  draw(fig) {
    fig.heading(36, ROW_A - 40, 'a', 'Bond detection');
    bondDetection(fig, ROW_A);
    fig.heading(36, ROW_B - 26, 'b', 'Pathway refinement');
    refinementStages(fig, ROW_B);
    fig.heading(36, ROW_C - 26, 'c', 'Barrier');
    fig.heading(486, ROW_C - 26, 'd', 'Reaction classes');
    barrier(fig, { x: 36, y: ROW_C, w: 430, h: BOTTOM_H });
    reactionClasses(fig, { x: 486, y: ROW_C + 2, w: 678, h: BOTTOM_H - 6 });
  },
};
