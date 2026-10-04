// The model run in data/eon.json on its landscape: the states and saddles EON found, the kinetic
// Monte Carlo trajectory between them, and the energy of the current state against simulated time.

import { Figure, scale, TYPE } from '../lib.mjs';
import { DOMAIN, energy, eon, LEVELS, stateNumber } from '../landscape.mjs';
import { drawSurface } from '../surface.mjs';

const forward = Object.entries(eon.processes).filter(([id]) => !id.endsWith('r'));

export default {
  name: 'eon-landscape',
  title: 'Adaptive kinetic Monte Carlo',
  alt:
    'A two-dimensional model energy landscape with several wells. EON searches found the minima and the saddles ' +
    'between them, and the kinetic Monte Carlo trajectory starts in the shallow well on the left and moves from ' +
    'well to well over the saddles. A chart of the current energy against simulated time on a logarithmic axis ' +
    'shows the descent in microseconds and the long waits in the deepest well.',
  caption:
    'ReactionFlow in EON mode with default settings and 12 steps on a two-dimensional model landscape at 300 K. ' +
    'EON process searches found every minimum and saddle shown, and the arrows are the kinetic Monte Carlo steps. ' +
    'The run reaches the deepest well within 20 µs and then leaves it only every 10 to 45 s. The well at the top ' +
    'is never entered because its saddle, 1.22 eV above state 4, lies outside the thermal window.',
  width: 1200,
  height: 470,
  draw(fig) {
    const t = fig.t;
    const rect = { x: 36, y: 16, w: 708, h: 438 };
    const { sx, sy } = drawSurface(fig, 'eon', rect, { domain: DOMAIN, field: energy, levels: LEVELS, columns: 170, rx: 10 });
    const at = ([x, y]) => [sx(x), sy(y)];
    const halo = fig.tint(t.contour, 0.18);
    const triangle = ([cx, cy], r, fill) =>
      fig.polygon([[cx, cy - r * 1.15], [cx + r, cy + r * 0.75], [cx - r, cy + r * 0.75]], { fill, stroke: t.bg, sw: 1.5, join: 'round' });

    // Every process found: minimum, saddle, minimum.
    for (const [, p] of forward) {
      fig.path(Figure.polyline([eon.states[p.state].position, p.saddle, eon.states[p.product].position].map(at)), {
        stroke: t.ink3,
        sw: 1.5,
        join: 'round',
      });
    }

    // The trajectory: one strand per step through its saddle; repeats of a hop sit side by side.
    const seen = new Map();
    eon.steps.forEach((step) => {
      const p = eon.processes[step.process];
      const [a, s, b] = [eon.states[step.from].position, p.saddle, eon.states[step.to].position].map(at);
      const key = [step.from, step.to].sort().join();
      const n = seen.get(key) ?? 0;
      seen.set(key, n + 1);
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const [nx, ny] = [-(b[1] - a[1]) / len, (b[0] - a[0]) / len];
      const side = (step.from < step.to ? 1 : -1) * (5 + 5 * Math.floor(n / 2));
      const mid = [s[0] + nx * side, s[1] + ny * side];
      // Stop short of the destination dot.
      const [px, py] = mid;
      const reach = Math.hypot(b[0] - px, b[1] - py);
      const end = [b[0] - ((b[0] - px) / reach) * 11, b[1] - ((b[1] - py) / reach) * 11];
      fig.path(Figure.smooth([a, mid, end]), { stroke: t.md, sw: 2.25, cap: 'round' });
      fig.head(end[0], end[1], Math.atan2(end[1] - py, end[0] - px), { color: t.md, size: 9 });
    });

    for (const [, p] of forward) triangle(at(p.saddle), 6.5, t.refine);

    // States, with their numbers; the starting structure relaxed into state 0.
    const start = at(eon.start);
    const first = eon.states['state-000000'].position;
    fig.arrow([start, at(first)], { color: t.ink, sw: 1.5, size: 7 });
    fig.polygon([[start[0], start[1] - 6], [start[0] + 6, start[1]], [start[0], start[1] + 6], [start[0] - 6, start[1]]], { fill: t.bg, stroke: t.ink, sw: 1.75, join: 'round' });
    fig.text(start[0] - 10, start[1] - 14, 'start', { size: TYPE.small, weight: 600, anchor: 'end', halo });
    for (const [id, state] of Object.entries(eon.states)) {
      const [cx, cy] = at(state.position);
      fig.circle(cx, cy, 7.5, { fill: t.bg });
      fig.circle(cx, cy, 5.5, { fill: t.ink });
      fig.text(cx, cy + 26, String(stateNumber(id)), { size: TYPE.small, weight: 700, anchor: 'middle', halo });
    }

    // Legend in the high ground at the bottom left.
    [
      [(kx, ky) => fig.circle(kx, ky, 5.5, { fill: t.ink }), 'state'],
      [(kx, ky) => triangle([kx, ky], 6.5, t.refine), 'saddle'],
      [(kx, ky) => fig.arrow([[kx - 10, ky], [kx + 10, ky]], { color: t.md, sw: 2.25, size: 8 }), 'KMC step'],
    ].forEach(([draw, label], i) => {
      const [kx, ky] = [rect.x + 34, rect.y + rect.h - 92 + i * 30];
      draw(kx, ky);
      fig.text(kx + 20, ky, label, { size: TYPE.small, middle: true, halo });
    });

    // Energy of the current state against simulated time.
    const [left, right, top, bottom] = [868, 1160, 36, 380];
    const times = eon.steps.map((s) => s.time_s);
    const [d0, d1] = [Math.floor(Math.log10(times[0])) - 1, Math.ceil(Math.log10(times.at(-1)))];
    const x = scale(d0, d1, left, right);
    const e0 = eon.states['state-000000'].energy_eV;
    const energies = [e0, ...eon.steps.map((s) => eon.states[s.to].energy_eV)].map((e) => e - e0);
    const [lo, hi] = [Math.floor(Math.min(...energies) * 4) / 4, 0.2];
    const y = scale(lo, hi, bottom, top);
    for (let v = 0; v >= lo; v -= 0.5) {
      fig.line(left, y(v), right, y(v), { stroke: t.grid });
      fig.text(left - 10, y(v), v === 0 ? '0' : v.toFixed(1), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
    }
    fig.line(left, bottom, right, bottom, { stroke: t.axis });
    const labels = { '-6': '1 µs', '-3': '1 ms', 0: '1 s', 3: '1000 s' };
    for (let d = d0; d <= d1; d++) {
      fig.line(x(d), bottom, x(d), bottom + 6, { stroke: t.axis });
      if (labels[d]) fig.text(x(d), bottom + 30, labels[d], { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    }
    fig.text(right, bottom + 64, 'simulated time', { size: TYPE.small, anchor: 'end', fill: t.ink2 });
    fig.text(left - 58, (top + bottom) / 2, 'E − E(0) (eV)', { size: TYPE.small, anchor: 'middle', fill: t.ink2, rotate: -90 });
    const stairs = [[x(d0), y(energies[0])]];
    times.forEach((time, i) => {
      stairs.push([x(Math.log10(time)), y(energies[i])], [x(Math.log10(time)), y(energies[i + 1])]);
    });
    stairs.push([right, y(energies.at(-1))]);
    fig.path(Figure.polyline(stairs), { stroke: t.md, sw: 2.25, join: 'round' });
  },
};
