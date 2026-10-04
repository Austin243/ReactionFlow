// What a campaign chooses and how it runs, in one figure: the built-in models (data/models.json),
// fixed-cell NEB against variable-cell SSNEB with real SSNEB output (data/ssneb.json), a real
// stop-and-restore against a run that never stopped (data/restart.json), and trajectories across a
// resubmitted job (schematic timings).

import { readFileSync } from 'node:fs';

import { Figure, HERO_TYPE as TYPE, scale } from '../lib.mjs';

const catalog = JSON.parse(readFileSync(new URL('../../data/models.json', import.meta.url)));
const ssneb = JSON.parse(readFileSync(new URL('../../data/ssneb.json', import.meta.url)));
const restart = JSON.parse(readFileSync(new URL('../../data/restart.json', import.meta.url)));

const NAMES = {
  mace: 'MACE', mace_field: 'MACE-FIELD', uma: 'UMA', aimnet2: 'AIMNet2', orb: 'ORB', mattersim: 'MatterSim',
  chgnet: 'CHGNet', sevennet: 'SevenNet', nep: 'NEP89', ani1xnr: 'ANI-1xnr',
};
const missing = catalog.backends.filter((backend) => !NAMES[backend.backend]).map((backend) => backend.backend);
if (missing.length) throw new Error(`campaigns figure has no name for: ${missing.join(', ')}`);
const families = catalog.backends
  .map((backend) => ({ name: NAMES[backend.backend], count: backend.models.length }))
  .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));

const [LEFT, RIGHT, W] = [36, 619, 545];
const ROW_A = 80;
const TOP_H = 350;
const ROW_B = ROW_A + TOP_H + 100;
const BOTTOM_H = 358;

function models(fig, rect) {
  const t = fig.t;
  const pitch = 34;
  const nameRight = rect.x + 158;
  const x = scale(0, Math.max(...families.map((f) => f.count)), nameRight + 14, rect.x + rect.w - 48);
  families.forEach((family, i) => {
    const y = rect.y + 16 + i * pitch;
    fig.text(nameRight, y, family.name, { size: TYPE.small, weight: 650, anchor: 'end', middle: true });
    fig.rect(nameRight + 14, y - 11, x(family.count) - nameRight - 14, 22, { rx: 4, fill: t.md });
    fig.text(x(family.count) + 8, y, String(family.count), { size: TYPE.small, fill: t.ink2, middle: true });
  });
  // Any other model comes through one of two adapters, drawn in the room beside the short bars.
  [['Any ASE', 'calculator'], ['Custom', 'adapter']].forEach((rows, i) => {
    const [bx, by, bw, bh] = [rect.x + 306, rect.y + 128 + i * 106, rect.w - 306, 90];
    fig.card(bx, by, bw, bh, { fill: t.bg, stroke: t.ink3, dash: '5 4' });
    fig.lines(bx + 20, by + 37, rows, { size: TYPE.body, weight: 700, lh: 30 });
  });
}

function ensembles(fig, rect) {
  const t = fig.t;
  // The band's cells: fixed for NEB, relaxing image by image for SSNEB.
  [['NEB', 'ΔE‡', false], ['SSNEB', 'ΔH‡', true]].forEach(([method, barrier, variable], row) => {
    const baseline = rect.y + 40 + row * 62;
    fig.text(rect.x, baseline - 17, method, { size: TYPE.body, weight: 700, middle: true });
    for (let i = 0; i < 5; i++) {
      const cx = rect.x + 156 + i * 54;
      const size = variable ? 28 + i * 3.8 : 36;
      const shear = variable ? i * 1.9 : 0;
      fig.polygon(
        [[cx - size / 2 + shear, baseline - size], [cx + size / 2 + shear, baseline - size], [cx + size / 2, baseline], [cx - size / 2, baseline]],
        { fill: t.bg, stroke: i === 2 ? t.refine : t.ink3, sw: i === 2 ? 2.25 : 1.5, join: 'round' },
      );
      const spread = 4 + i * 1.8;
      fig.circle(cx - spread + shear / 2, baseline - size / 2, 3.4, { fill: t.ink2 });
      fig.circle(cx + spread + shear / 2, baseline - size / 2, 3.4, { fill: t.ink2 });
    }
    fig.text(rect.x + rect.w, baseline - 17, barrier, { size: TYPE.body, weight: 700, anchor: 'end', middle: true });
  });

  // SSNEB enthalpy along the band at three pressures, with the barrier at each.
  const [left, right, top, bottom] = [rect.x + 70, rect.x + rect.w - 170, rect.y + 180, rect.y + rect.h - 36];
  const x = scale(-0.3, 6.3, left, right);
  const y = scale(-0.005, 0.105, bottom, top);
  for (const tick of [0, 0.05, 0.1]) {
    fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
    fig.text(left - 12, y(tick), tick ? tick.toFixed(2) : '0', { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
  }
  fig.line(left, bottom, right, bottom, { stroke: t.axis });
  ['R', '1', '2', '3', '4', '5', 'P'].forEach((label, i) =>
    fig.text(x(i), bottom + 30, label, { size: TYPE.small, anchor: 'middle', fill: t.ink2, weight: i % 6 ? 400 : 700 }),
  );
  fig.text(rect.x, top - 30, 'H − H(R) (eV)', { size: TYPE.small, fill: t.ink2, middle: true });
  const [gx, vx] = [right + 30, rect.x + rect.w];
  fig.text(gx + 26, top - 30, 'GPa', { size: TYPE.small, fill: t.ink2, middle: true });
  fig.text(vx, top - 30, 'ΔH‡', { size: TYPE.small, fill: t.ink2, anchor: 'end', middle: true });
  ssneb.results.forEach((result, k) => {
    const points = result.enthalpies_eV.map((h, i) => [x(i), y(h - result.enthalpies_eV[0])]);
    fig.path(Figure.polyline(points), { stroke: t.ramp[k], sw: 2.5, join: 'round' });
    points.forEach(([px, py]) => fig.circle(px, py, 4.5, { fill: t.ramp[k], stroke: t.bg, sw: 1.5 }));
    const ly = top + 12 + (ssneb.results.length - 1 - k) * 34;
    fig.line(gx, ly, gx + 18, ly, { stroke: t.ramp[k], sw: 3, cap: 'round' });
    fig.text(gx + 26, ly, String(result.pressure_GPa), { size: TYPE.small, middle: true });
    fig.text(vx, ly, result.barrier_eV.toFixed(3), { size: TYPE.small, weight: 650, anchor: 'end', middle: true });
  });
}

function exactRestart(fig, rect) {
  const t = fig.t;
  const [left, right] = [rect.x + 64, rect.x + rect.w - 8];
  const steps = restart.steps;
  const x = scale(0, steps.at(-1), left, right);
  const split = restart.split_step;
  const after = steps.indexOf(split);
  const chart = (top, bottom, range, ticks, title) => {
    const y = scale(range[0], range[1], bottom, top);
    for (const tick of ticks) {
      fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
      fig.text(left - 12, y(tick), tick ? String(tick) : '0', { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
    }
    fig.line(left, bottom, right, bottom, { stroke: t.axis });
    fig.text(rect.x, top - 24, title, { size: TYPE.small, fill: t.ink2, middle: true });
    fig.line(x(split), top - 6, x(split), bottom, { stroke: t.ink3, sw: 1.25, dash: '4 4' });
    return y;
  };
  const line = (y, values, color, sw, from = 0) =>
    fig.path(Figure.polyline(values.map((v, i) => [x(steps[i]), y(v)]).slice(from)), { stroke: color, sw, join: 'round', cap: 'round' });

  const vy = chart(rect.y + 34, rect.y + 120, [368, 392], [370, 380, 390], 'Cell volume (Å³)');
  line(vy, restart.reference_volume_A3, t.ink3, 6);
  line(vy, restart.structural_volume_A3, t.event, 2.5, after);
  line(vy, restart.exact_volume_A3, t.md, 2.5);
  fig.text(x(split) + 10, rect.y + 10, 'Restart', { size: TYPE.small, weight: 650, middle: true });

  const dy = chart(rect.y + 188, rect.y + 254, [0, 0.45], [0, 0.4], 'RMS position difference (Å)');
  line(dy, restart.structural_rms_deviation_A, t.event, 2.5, after);
  line(dy, restart.exact_rms_deviation_A, t.md, 3, after);
  for (const tick of [0, 100, 200, 300, 400]) {
    fig.line(x(tick), rect.y + 254, x(tick), rect.y + 260, { stroke: t.axis });
    fig.text(x(tick), rect.y + 286, String(tick), { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
  }
  fig.text((left + right) / 2, rect.y + 318, 'MD step', { size: TYPE.small, anchor: 'middle', fill: t.ink2 });

  fig.keys(rect.x + 4, rect.y + rect.h - 4, [
    [(kx, ky) => fig.line(kx - 11, ky, kx + 11, ky, { stroke: t.ink3, sw: 6, cap: 'round' }), 'Uninterrupted'],
    [(kx, ky) => fig.line(kx - 11, ky, kx + 11, ky, { stroke: t.md, sw: 3, cap: 'round' }), 'Exact'],
    [(kx, ky) => fig.line(kx - 11, ky, kx + 11, ky, { stroke: t.event, sw: 3, cap: 'round' }), 'Structural'],
  ], { gap: 26 });
}

function trajectories(fig, rect) {
  const t = fig.t;
  // Segment ends are in units of the two job spans (job 1 is 366 long, job 2 is 352).
  const [L, R] = [rect.x + 134, rect.x + rect.w - 12];
  const k = (R - L - 30) / (366 + 352);
  const [end1, start2] = [L + 366 * k, L + 366 * k + 30];
  const [bar, pitch] = [20, 32];
  const top = rect.y + 52;
  const y = (i) => top + i * pitch + (i > 3 ? 14 : 0);
  const lanes = [
    { job1: [['md', 0, 116], ['ref', 116, 162], ['md', 162, 366]], job2: [['md', 0, 204]], done2: 204 },
    { job1: [['md', 0, 366]], job2: [['md', 0, 94], ['ref', 94, 146], ['md', 146, 352]], more: true },
    { job1: [['md', 0, 66], ['ref', 66, 104], ['md', 104, 246], ['ref', 246, 296], ['md', 296, 366]], job2: [['md', 0, 352]], more: true },
    { job1: [['md', 0, 296]], done1: 296, job2: [] },
    { job1: [['md', 0, 206]], error: 206, job2: [['md', 0, 352]], more: true },
    { job1: [['md', 0, 176], ['ref', 176, 236], ['md', 236, 366]], job2: [['md', 0, 174], ['ref', 174, 224], ['md', 224, 352]], more: true },
    { job1: [['md', 0, 366]], job2: [['md', 0, 352]], more: true },
    { job1: [['md', 0, 286], ['ref', 286, 340], ['md', 340, 366]], job2: [['md', 0, 284]], done2: 284 },
  ];

  fig.rect(end1 + 3, top - 20, start2 - end1 - 6, y(7) + bar / 2 + 20 - (top - 20), { fill: t.panel2, rx: 4 });
  for (const [a, b, label] of [[L, end1, 'Job 1'], [start2, R, 'Job 2']]) {
    fig.line(a, top - 30, b, top - 30, { stroke: t.ink3, sw: 1.25 });
    for (const edge of [a, b]) fig.line(edge, top - 35, edge, top - 25, { stroke: t.ink3, sw: 1.25 });
    fig.text((a + b) / 2, top - 42, label, { size: TYPE.small, weight: 650, anchor: 'middle' });
  }
  [0, 4].forEach((first, node) => {
    const [a, b] = [y(first) - bar / 2 - 2, y(first + 3) + bar / 2 + 2];
    fig.path(`M${rect.x + 38} ${a} h-7 V${b} h7`, { stroke: t.ink3, sw: 1.5 });
    fig.text(rect.x + 14, (a + b) / 2, `Node ${node + 1}`, { size: TYPE.small, weight: 650, anchor: 'middle', middle: true, rotate: -90 });
  });
  const mark = (kind, cx, cy, r = 11) => {
    const color = kind === 'check' ? t.good : t.critical;
    fig.circle(cx, cy, r, { fill: fig.tint(color, 0.16), stroke: color, sw: 1.5 });
    fig.glyph(kind, cx, cy, kind === 'check' ? r * 0.9 : r * 0.75, color);
  };
  lanes.forEach((lane, i) => {
    const cy = y(i);
    fig.text(rect.x + 50, cy, `GPU ${i % 4}`, { size: TYPE.small, fill: t.ink2, middle: true });
    const segment = (origin, [kind, a, b]) =>
      fig.rect(origin + a * k + (a ? 1 : 0), cy - bar / 2, (b - a) * k - (a ? 1 : 0), bar, { rx: 4, fill: kind === 'md' ? t.md : t.refine });
    lane.job1.forEach((s) => segment(L, s));
    lane.job2.forEach((s) => segment(start2, s));
    if (lane.more) fig.polygon([[R - 9, cy - bar / 2], [R + 2, cy], [R - 9, cy + bar / 2]], { fill: t.md });
    for (const [origin, at] of [[L, lane.done1], [start2, lane.done2]]) if (at !== undefined) mark('check', origin + at * k + 17, cy);
    if (lane.error !== undefined) mark('cross', L + lane.error * k + 17, cy);
  });

  fig.keys(rect.x + 4, rect.y + rect.h - 4, [
    [(kx, ky) => fig.rect(kx - 11, ky - 8, 22, 16, { rx: 3, fill: t.md }), 'MD'],
    [(kx, ky) => fig.rect(kx - 11, ky - 8, 22, 16, { rx: 3, fill: t.refine }), 'Refinement'],
    [(kx, ky) => mark('check', kx, ky), 'Done'],
    [(kx, ky) => mark('cross', kx, ky), 'Error'],
  ], { gap: 26 });
}

export default {
  name: 'campaigns',
  title: 'Campaigns',
  label: 'Campaigns',
  alt:
    'Four panels on running campaigns. a, the built-in model families with their number of models, from 22 for ' +
    'MACE to one each for ANI-1xnr, MACE-FIELD, and NEP89, plus any ASE calculator or a custom adapter. b, NEB ' +
    'keeps the cell fixed and reports an energy barrier, SSNEB relaxes the cell of every image and reports an ' +
    'enthalpy barrier. The SSNEB barrier of a model system rises from 0.062 eV at 0 GPa to 0.095 eV at 0.8 GPa. ' +
    'c, a run stopped and restored at step 150 against one that never stopped. The exact restart follows the same ' +
    'cell volume with zero position difference, and a restart from atoms, momenta, and cell drifts away by 0.4 Å. ' +
    'd, eight trajectories on two four-GPU nodes over two jobs. Refinement pauses only its own trajectory, an error ' +
    'stops only its own trajectory, and the second job resumes the rest.',
  caption:
    'a, the catalog that <code>reactionflow models</code> prints. b, SSNEB on the volume-coupled double well from ' +
    'the test suite. c, the generic ASE adapter on 32 copper atoms with EMT in NPT at 600 K and 1 GPa. d, schematic ' +
    'timings.',
  width: 1200,
  height: ROW_B + BOTTOM_H + 24,
  draw(fig) {
    fig.heading(LEFT, ROW_A - 28, 'a', 'Models');
    fig.heading(RIGHT, ROW_A - 28, 'b', 'NVT or NPT');
    models(fig, { x: LEFT, y: ROW_A, w: W, h: TOP_H });
    ensembles(fig, { x: RIGHT, y: ROW_A, w: W, h: TOP_H });
    fig.heading(LEFT, ROW_B - 28, 'c', 'Exact restart');
    fig.heading(RIGHT, ROW_B - 28, 'd', 'Many trajectories');
    exactRestart(fig, { x: LEFT, y: ROW_B, w: W, h: BOTTOM_H });
    trajectories(fig, { x: RIGHT, y: ROW_B, w: W, h: BOTTOM_H });
  },
};
