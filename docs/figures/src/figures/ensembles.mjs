// Fixed-cell NEB against variable-cell SSNEB, with real SSNEB output at three pressures
// (data/ssneb.json, the volume-coupled double well from the test suite).

import { readFileSync } from 'node:fs';

import { Figure, scale, TYPE } from '../lib.mjs';

const data = JSON.parse(readFileSync(new URL('../../data/ssneb.json', import.meta.url)));

export default {
  name: 'ensembles',
  title: 'Constant volume and constant pressure',
  alt:
    'Comparison of pathway refinement at constant volume and at constant pressure. At constant volume the cell is ' +
    'fixed, the band is a nudged elastic band, and the barrier is a potential-energy difference. At constant pressure ' +
    'the cell relaxes along the band with solid-state NEB and the barrier is an enthalpy difference. A chart of a model ' +
    'system shows the enthalpy barrier rising from 0.063 to 0.095 eV between 0 and 0.8 GPa while the cell volume grows ' +
    'along the band.',
  caption:
    'The trajectory setting decides the method. A numeric <code>pressure_GPa</code> runs NPT dynamics and refines with ' +
    'variable-cell SSNEB on the enthalpy; <code>null</code> runs NVT and keeps the cell fixed. The chart is real SSNEB ' +
    'output for the volume-coupled double well in the test suite.',
  width: 1200,
  height: 400,
  draw(fig) {
    const t = fig.t;

    // Left: side-by-side comparison.
    const cols = [{ x: 150, title: 'Constant volume', setting: '"pressure_GPa": null' }, { x: 422, title: 'Constant pressure', setting: '"pressure_GPa": 20' }];
    const colW = 260;
    const [top, cardH] = [16, 368];
    cols.forEach((col, c) => {
      fig.card(col.x, top, colW, cardH, { fill: t.panel });
      fig.text(col.x + 18, top + 38, col.title, { size: TYPE.body, weight: 700 });
      fig.text(col.x + 18, top + 70, col.setting, { size: TYPE.small, mono: true, fill: t.ink2 });
      // Five images of the band; the cell is identical on the left and changes on the right.
      const baseline = top + 150;
      for (let i = 0; i < 5; i++) {
        const cx = col.x + 36 + i * 45;
        const size = c ? 28 + i * 3.5 : 36;
        const shear = c ? i * 1.9 : 0;
        fig.polygon(
          [[cx - size / 2 + shear, baseline - size], [cx + size / 2 + shear, baseline - size], [cx + size / 2, baseline], [cx - size / 2, baseline]],
          { fill: t.bg, stroke: i === 2 ? t.refine : t.ink3, sw: i === 2 ? 2.25 : 1.5, join: 'round' },
        );
        const spread = 5 + i * 2;
        fig.circle(cx - spread + shear / 2, baseline - size / 2, 3.8, { fill: t.ink2 });
        fig.circle(cx + spread + shear / 2, baseline - size / 2, 3.8, { fill: t.ink2 });
      }
    });
    const rows = [
      ['Band', 'NEB', 'SSNEB'],
      ['Cell', 'fixed', 'relaxes'],
      ['Barrier', 'ΔE‡', 'ΔH‡, H = E + PV'],
    ];
    rows.forEach(([label, a, b], i) => {
      const ry = top + 222 + i * 56;
      fig.line(36, ry - 28, cols[1].x + colW, ry - 28, { stroke: t.border });
      fig.text(36, ry, label, { size: TYPE.body, weight: 700, middle: true });
      fig.text(cols[0].x + 18, ry, a, { size: TYPE.body, middle: true });
      fig.text(cols[1].x + 18, ry, b, { size: TYPE.body, middle: true });
    });

    // Right: SSNEB results at three pressures.
    const [left, right] = [810, 1016];
    const x = scale(-0.3, 6.3, left, right);
    const results = data.results;
    const hy = scale(-0.005, 0.105, 206, 50);
    fig.text(left - 52, 26, 'Enthalpy, H − H(R) (eV)', { size: TYPE.small, weight: 700, middle: true });
    for (const tick of [0, 0.05, 0.1]) {
      fig.line(left, hy(tick), right, hy(tick), { stroke: t.grid });
      fig.text(left - 10, hy(tick), tick ? tick.toFixed(2) : '0', { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
    }
    // Legend with the barrier at each pressure, highest pressure first to match the curves.
    const [lx, vx] = [1036, 1164];
    fig.text(lx + 26, 74, 'GPa', { size: TYPE.small, fill: t.ink2, middle: true });
    fig.text(vx, 74, 'ΔH‡', { size: TYPE.small, fill: t.ink2, anchor: 'end', middle: true });
    results.forEach((result, k) => {
      const values = result.enthalpies_eV.map((h) => h - result.enthalpies_eV[0]);
      const points = values.map((v, i) => [x(i), hy(v)]);
      fig.path(Figure.polyline(points), { stroke: t.ramp[k], sw: 2.25, join: 'round' });
      points.forEach(([px, py]) => {
        fig.circle(px, py, 6.5, { fill: t.bg });
        fig.circle(px, py, 4.5, { fill: t.ramp[k] });
      });
      const ly = 106 + (results.length - 1 - k) * 30;
      fig.line(lx, ly, lx + 18, ly, { stroke: t.ramp[k], sw: 2.5, cap: 'round' });
      fig.text(lx + 26, ly, String(result.pressure_GPa), { size: TYPE.small, middle: true });
      fig.text(vx, ly, result.barrier_eV.toFixed(3), { size: TYPE.small, weight: 600, anchor: 'end', middle: true });
    });

    const vy = scale(206, 226, 332, 268);
    fig.text(left - 52, 246, 'Cell volume (Å³)', { size: TYPE.small, weight: 700, middle: true });
    for (const tick of [210, 220]) {
      fig.line(left, vy(tick), right, vy(tick), { stroke: t.grid });
      fig.text(left - 10, vy(tick), String(tick), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
    }
    results.forEach((result, k) => {
      fig.path(Figure.polyline(result.volumes_A3.map((v, i) => [x(i), vy(v)])), { stroke: t.ramp[k], sw: 2.25, join: 'round' });
    });
    fig.line(left, 340, right, 340, { stroke: t.axis });
    ['R', '1', '2', '3', '4', '5', 'P'].forEach((label, i) =>
      fig.text(x(i), 366, label, { size: TYPE.small, anchor: 'middle', fill: t.ink2, weight: i % 6 ? 400 : 700 }),
    );
    fig.text(right + 20, 366, 'image', { size: TYPE.small, fill: t.ink2 });
  },
};
