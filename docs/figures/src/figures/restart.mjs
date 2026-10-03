// Exact restart: what a checkpoint holds, and a real stop-and-restore compared with a run that
// never stopped (data/restart.json).

import { readFileSync } from 'node:fs';

import { Figure, scale, TYPE } from '../lib.mjs';

const data = JSON.parse(readFileSync(new URL('../../data/restart.json', import.meta.url)));

export default {
  name: 'restart',
  title: 'Exact restart',
  alt:
    'A checkpoint holds the atoms, the integrator, thermostat and barostat state, the random-number generator, the ' +
    'calculator contract, and the bond monitor. Two charts compare a run that was stopped at step 150 and restored ' +
    'with a run that never stopped: after an exact restart the cell volume follows the same curve and the position ' +
    'difference is zero at every step, while a restart from atoms, momenta, and cell alone drifts away.',
  caption:
    'A real comparison with the generic ASE adapter: 32 copper atoms, EMT, Langevin BAOAB NPT at 600 K and 1 GPa. ' +
    'The exact checkpoint was written to disk and read back at step 150; positions afterwards are identical to the ' +
    'uninterrupted run. Restarting from atoms, momenta, and cell alone gives a different trajectory.',
  width: 1200,
  height: 436,
  draw(fig) {
    const t = fig.t;

    // Left: contents of a checkpoint.
    const rows = [
      ['Atoms', t.md],
      ['Integrator', t.md],
      ['Random numbers', t.md],
      ['Calculator contract', t.md],
      ['Bond monitor', t.event],
    ];
    const card = { x: 36, y: 16, w: 310, h: 404 };
    fig.card(card.x, card.y, card.w, card.h);
    fig.text(card.x + 20, card.y + 40, 'In every checkpoint', { size: TYPE.body, weight: 700 });
    rows.forEach(([name, color], i) => {
      const ry = card.y + 66 + i * 66;
      fig.rect(card.x + 14, ry, card.w - 28, 54, { rx: 8, fill: t.bg, stroke: t.border });
      fig.circle(card.x + 36, ry + 27, 6, { fill: color });
      fig.text(card.x + 54, ry + 27, name, { size: TYPE.body, weight: 600, middle: true });
    });

    // Right: two charts on a shared step axis.
    const [left, right] = [458, 1150];
    const steps = data.steps;
    const x = scale(0, steps.at(-1), left, right);
    const split = data.split_step;
    const colors = { reference: t.ink3, exact: t.md, structural: t.event };
    const chart = (top, bottom, range, ticks, format, title) => {
      const y = scale(range[0], range[1], bottom, top);
      for (const tick of ticks) {
        fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
        fig.text(left - 12, y(tick), format(tick), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
      }
      fig.line(left, bottom, right, bottom, { stroke: t.axis });
      fig.text(left, top - 20, title, { size: TYPE.small, weight: 700, middle: true });
      fig.line(x(split), top - 4, x(split), bottom, { stroke: t.ink3, sw: 1.25, dash: '4 4' });
      return y;
    };
    const line = (y, values, color, sw, from = 0) =>
      fig.path(Figure.polyline(values.map((v, i) => [x(steps[i]), y(v)]).slice(from)), { stroke: color, sw, join: 'round', cap: 'round' });
    const after = steps.indexOf(split);

    const vy = chart(56, 170, [368, 392], [370, 380, 390], (v) => String(v), 'Cell volume (Å³)');
    line(vy, data.reference_volume_A3, colors.reference, 6);
    line(vy, data.structural_volume_A3, colors.structural, 2.5, after);
    line(vy, data.exact_volume_A3, colors.exact, 2.5);
    fig.text(x(split) + 10, 36, 'restart', { size: TYPE.small, fill: t.ink2, middle: true });

    const dy = chart(270, 360, [0, 0.45], [0, 0.2, 0.4], (v) => (v ? v.toFixed(1) : '0'), 'RMS position difference (Å)');
    line(dy, data.structural_rms_deviation_A, colors.structural, 2.5, after);
    line(dy, data.exact_rms_deviation_A, colors.exact, 3, after);
    for (const tick of [0, 100, 200, 300, 400]) {
      fig.line(x(tick), 360, x(tick), 366, { stroke: t.axis });
      fig.text(x(tick), 388, String(tick), { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    }
    fig.text(right, 420, 'MD step', { size: TYPE.small, anchor: 'end', fill: t.ink2 });

    // Direct labels and legend.
    const last = steps.length - 1;
    fig.text(right - 4, dy(data.structural_rms_deviation_A[last]) - 16, 'atoms, momenta, cell only', { size: TYPE.small, weight: 600, anchor: 'end', halo: true });
    fig.text(right - 4, dy(0) - 14, `exact: ${data.exact_max_abs_position_difference_A} Å at every step`, { size: TYPE.small, weight: 600, anchor: 'end', halo: true });
    const legend = [['uninterrupted', colors.reference, 6], ['exact restart', colors.exact, 2.5], ['structural restart', colors.structural, 2.5]];
    let lx = right - legend.reduce((sum, [label]) => sum + 34 + fig.measure(label, { size: TYPE.small }) + 26, -26);
    for (const [label, color, sw] of legend) {
      fig.line(lx, 208, lx + 24, 208, { stroke: color, sw, cap: 'round' });
      fig.text(lx + 34, 208, label, { size: TYPE.small, fill: t.ink2, middle: true });
      lx += 34 + fig.measure(label, { size: TYPE.small }) + 26;
    }
  },
};
