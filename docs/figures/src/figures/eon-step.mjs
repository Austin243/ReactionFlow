// One kinetic Monte Carlo step of the model run in data/eon.json: the rates of the processes in
// the thermal window, the random pick in proportion to rate, and the exponential waiting time.

import { Figure, scale, TYPE } from '../lib.mjs';
import { duration, events, rate, stateNumber } from '../landscape.mjs';

// The first step leaves the starting state.
const step = events().find((e) => e.kind === 'step' && e.index === 0);

/** Number as a power of ten for axis ticks, e.g. 10⁶. */
const SUPERSCRIPT = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const power = (exponent) => `10${[...String(exponent)].map((c) => SUPERSCRIPT[c]).join('')}`;

export default {
  name: 'eon-step',
  title: 'A kinetic Monte Carlo step',
  alt:
    'The first kinetic Monte Carlo step of the model run. On the left is the rate of each process out of the ' +
    'starting state, prefactor times exp(-barrier/kT) at 300 K. In the middle a random number picks one process ' +
    'in proportion to its rate. On the right the clock advances by a random time drawn from the exponential ' +
    'distribution with mean 1 over the summed rate.',
  caption:
    'The first step of the model run, leaving the starting state at 300 K. Each process in the thermal window is ' +
    'picked with probability proportional to its rate, and the clock advances by an exponentially distributed ' +
    'time with mean 1 / (sum of rates).',
  width: 1200,
  height: 330,
  draw(fig) {
    const t = fig.t;
    const { window } = step;
    const chosen = window.find((p) => p.id === step.step.process);
    const total = step.step.total_rate_s;
    const [pick] = step.step.random;
    const heading = (x, n, label) => {
      fig.badge(x + 13, 30, n, { r: 13 });
      fig.text(x + 36, 30, label, { size: TYPE.body, weight: 700, middle: true });
    };
    const colors = window.map((_, i) => t.ramp[[1, 2, 0][i % 3]]);
    // Dark text on a light bar, white on a dark one.
    const on = (color) => {
      const [r, g, b] = [1, 3, 5].map((k) => parseInt(color.slice(k, k + 2), 16) / 255);
      return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 ? '#1f2328' : '#ffffff';
    };

    // 1. Rates on a log axis.
    heading(36, 1, 'Rates');
    const [left, right] = [124, 400];
    const exponents = window.map((p) => Math.log10(rate(p)));
    const [lo, hi] = [Math.floor(Math.min(...exponents)) - 1, Math.ceil(Math.max(...exponents))];
    const x = scale(lo, hi, left, right);
    const rowY = (i) => 96 + i * 66;
    for (let e = lo; e <= hi; e += Math.max(1, Math.round((hi - lo) / 3))) {
      fig.line(x(e), 70, x(e), rowY(window.length - 1) + 30, { stroke: t.grid });
      fig.text(x(e), rowY(window.length - 1) + 58, power(e), { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    }
    fig.text(right, rowY(window.length - 1) + 92, 'rate (1/s)', { size: TYPE.small, anchor: 'end', fill: t.ink2 });
    window.forEach((p, i) => {
      const y = rowY(i);
      fig.text(36, y, `→ ${stateNumber(p.product)}`, { size: TYPE.body, weight: 700, middle: true });
      fig.rect(left, y - 13, x(Math.log10(rate(p))) - left, 26, { rx: 5, fill: colors[i] });
      fig.text(x(Math.log10(rate(p))) + 12, y, `${p.barrier_eV.toFixed(2)} eV`, { size: TYPE.small, fill: t.ink2, middle: true });
    });

    // 2. The pick: a random number on the summed rates.
    heading(452, 2, 'Pick');
    const [pl, pr, py] = [452, 760, 130];
    const share = scale(0, total, pl, pr);
    let acc = 0;
    window.forEach((p, i) => {
      const [a, b] = [share(acc), share(acc + rate(p))];
      fig.rect(a + (i ? 1.5 : 0), py - 22, b - a - (i ? 1.5 : 0), 44, { rx: 6, fill: colors[i] });
      if (b - a > 60) fig.text((a + b) / 2, py, `→ ${stateNumber(p.product)}`, { size: TYPE.body, weight: 700, anchor: 'middle', middle: true, fill: on(colors[i]) });
      acc += rate(p);
    });
    const ux = share(pick * total);
    fig.line(ux, py - 46, ux, py - 26, { stroke: t.ink, sw: 2 });
    fig.head(ux, py - 24, Math.PI / 2, { color: t.ink, size: 10 });
    fig.text(ux, py - 56, `u = ${pick.toFixed(2)}`, { size: TYPE.small, weight: 600, anchor: 'middle' });
    fig.line(pl, py + 40, pr, py + 40, { stroke: t.axis });
    for (const [v, label] of [[0, '0'], [total, 'Σk']]) {
      fig.line(share(v), py + 36, share(v), py + 44, { stroke: t.axis });
      fig.text(share(v), py + 68, label, { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    }
    fig.text((pl + pr) / 2, py + 120, ['moves to state ', { t: String(stateNumber(chosen.product)), weight: 700 }], { size: TYPE.body, anchor: 'middle', middle: true });

    // 3. The clock: an exponential waiting time with mean 1/Σk.
    heading(812, 3, 'Clock');
    const [cl, cr, ct, cb] = [828, 1150, 76, 244];
    const mean = 1 / total;
    const tx = scale(0, 4 * mean, cl, cr);
    const ty = scale(0, 1, cb, ct);
    const density = Array.from({ length: 81 }, (_, i) => [tx((4 * mean * i) / 80), ty(Math.exp(-i / 20))]);
    fig.path(Figure.polyline([[cl, cb], ...density, [cr, cb]]) + ' Z', { fill: fig.tint(t.md, 0.14) });
    fig.path(Figure.polyline(density), { stroke: t.md, sw: 2.25, join: 'round' });
    fig.line(cl, cb, cr, cb, { stroke: t.axis });
    const dt = step.step.dt_s;
    fig.line(tx(dt), cb, tx(dt), ty(Math.exp(-dt / mean)) - 2, { stroke: t.ink, sw: 2 });
    fig.circle(tx(dt), ty(Math.exp(-dt / mean)), 6, { fill: t.ink, stroke: t.bg, sw: 2 });
    fig.text(tx(dt) + 12, ty(Math.exp(-dt / mean)) - 16, `Δt = ${duration(dt)}`, { size: TYPE.body, weight: 700, halo: true });
    for (const k of [0, 1, 2, 3]) {
      fig.line(tx(k * mean), cb, tx(k * mean), cb + 6, { stroke: t.axis });
      fig.text(tx(k * mean), cb + 30, k === 0 ? '0' : k === 1 ? '1/Σk' : `${k}/Σk`, { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    }
    fig.text(cr, cb + 64, 'waiting time', { size: TYPE.small, anchor: 'end', fill: t.ink2 });
  },
};
