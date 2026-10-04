// One EON process search from the starting state of the model run (data/eon.json), in the stages
// EON runs it, and every search from that state.

import { Figure, scale, TYPE } from '../lib.mjs';
import { energy, eon, LEVELS, stateNumber } from '../landscape.mjs';
import { drawSurface } from '../surface.mjs';

const START = 'state-000000';
const searches = eon.attempts.filter((a) => a.state === START);
// The search that found the process to state 2.
const shown = searches.find((a) => a.status === 'new' && a.product === 'state-000002');
const process = eon.processes[shown.process];
const [reactant, product] = [eon.states[START], eon.states[shown.product]];

export default {
  name: 'eon-search',
  title: 'One EON search',
  alt:
    'Four views of the model landscape around the starting state. A random push moves the atom away from the ' +
    'minimum, the dimer method climbs from there to a saddle, and relaxations down both sides of the saddle reach ' +
    'the starting state and a new state, so the process is kept. The last view shows all searches from the ' +
    'starting state. Most converge on the same two saddles and some climb into the outer wall and fail. A chart ' +
    'shows the energy along the process with its barrier of 0.44 eV.',
  caption:
    "An actual EON search from the model run, with ReactionFlow's default settings. EON then computes the " +
    'harmonic prefactor from the vibrations at the minimum and the saddle. Searches that find a known saddle ' +
    'again are repeats, and they raise the confidence that every low exit of the state is known.',
  width: 1200,
  height: 556,
  draw(fig) {
    const t = fig.t;
    const [PW, PH] = [384, 253];
    const panels = [
      { x: 36, y: 16 },
      { x: 436, y: 16 },
      { x: 36, y: 285 },
      { x: 436, y: 285 },
    ];
    const titles = ['Push', 'Dimer climb', 'Relax both sides', `All ${searches.length} searches`];
    // Close-up of the search, and a wider view for every search from the state.
    const near = { x0: -2.8, x1: -0.85, y0: -0.3, y1: 0.985 };
    const wide = { x0: -3.62, x1: -0.62, y0: -1.0, y1: 0.976 };
    const halo = fig.tint(t.contour, 0.18);

    panels.forEach((panel, index) => {
      const domain = index === 3 ? wide : near;
      const rect = { x: panel.x, y: panel.y, w: PW, h: PH };
      const { sx, sy } = drawSurface(fig, `eon-search-${index === 3 ? 'wide' : 'near'}`, rect, { domain, field: energy, levels: LEVELS });
      const at = ([px, py]) => [sx(px), sy(py)];
      const clip = fig.clipRect(`eon-search-${index}`, rect.x, rect.y, rect.w, rect.h, 8);
      const dot = (p, r, fill) => {
        const [cx, cy] = at(p);
        fig.circle(cx, cy, r + 2, { fill: t.bg });
        fig.circle(cx, cy, r, { fill });
      };
      const triangle = (p, r, fill) => {
        const [cx, cy] = at(p);
        fig.polygon([[cx, cy - r * 1.15], [cx + r, cy + r * 0.75], [cx - r, cy + r * 0.75]], { fill, stroke: t.bg, sw: 2, join: 'round' });
      };
      const trail = (points, color, o = {}) => {
        // Stop at the ring of the marker the path ends on.
        const drawn = points.map(at);
        const end = drawn.at(-1);
        const shown = drawn.slice(0, drawn.findLastIndex(([px, py]) => Math.hypot(px - end[0], py - end[1]) > (o.stop ?? 12)) + 1);
        if (!shown.length) return;
        const last = shown.at(-1);
        const reach = Math.hypot(end[0] - last[0], end[1] - last[1]);
        shown.push([end[0] - ((end[0] - last[0]) / reach) * (o.stop ?? 12) * 0.75, end[1] - ((end[1] - last[1]) / reach) * (o.stop ?? 12) * 0.75]);
        fig.arrow(shown, { color, sw: o.sw ?? 2, size: o.head ?? 8, dash: o.dash, opacity: o.opacity });
      };
      const label = (p, text, dx, dy) => {
        const [cx, cy] = at(p);
        fig.text(cx + dx, cy + dy, text, { size: TYPE.body, weight: 700, anchor: 'middle', halo });
      };
      // Legend in the high ground at the bottom right, or at the bottom left in the wide view.
      const key = (rows) => {
        const width = Math.max(...rows.map(([, text]) => fig.measure(text, { size: TYPE.small })));
        const kx = index === 3 ? rect.x + 30 : rect.x + PW - 40 - width;
        rows.forEach(([draw, text], i) => {
          const ky = rect.y + PH - 50 + i * 28;
          draw(kx, ky);
          fig.text(kx + 16, ky, text, { size: TYPE.small, middle: true, halo });
        });
      };
      const push = shown.climb[0];

      if (index === 0) {
        const [cx, cy] = at(reactant.position);
        fig.group({ clip }, () => fig.circle(cx, cy, 0.5 * (sx(1) - sx(0)), { stroke: t.ink3, sw: 1.25, dash: '4 4' }));
        const [qx, qy] = at(push);
        const reach = Math.hypot(qx - cx, qy - cy);
        const [ux, uy] = [(qx - cx) / reach, (qy - cy) / reach];
        fig.arrow([[cx + ux * 10, cy + uy * 10], [qx - ux * 9, qy - uy * 9]], { color: t.event, sw: 2.25, size: 9 });
        dot(reactant.position, 5.5, t.ink);
        fig.circle(qx, qy, 6, { fill: t.bg, stroke: t.ink, sw: 1.75 });
        label(reactant.position, '0', -2, 26);
        key([
          [(kx, ky) => fig.line(kx - 8, ky, kx + 8, ky, { stroke: t.ink3, sw: 1.25, dash: '4 4' }), '0.5 Å'],
          [(kx, ky) => fig.circle(kx, ky, 5, { fill: t.bg, stroke: t.ink, sw: 1.75 }), 'pushed'],
        ]);
      }
      if (index === 1) {
        trail(shown.climb, t.refine, { sw: 2.5, head: 9 });
        dot(reactant.position, 5.5, t.ink);
        const [qx, qy] = at(push);
        fig.circle(qx, qy, 6, { fill: t.bg, stroke: t.ink, sw: 1.75 });
        triangle(process.saddle, 8.5, t.refine);
        label(reactant.position, '0', -2, 26);
        key([
          [(kx, ky) => fig.line(kx - 8, ky, kx + 8, ky, { stroke: t.refine, sw: 2.5, cap: 'round' }), 'dimer path'],
          [(kx, ky) => fig.polygon([[kx, ky - 8], [kx + 7, ky + 5], [kx - 7, ky + 5]], { fill: t.refine }), 'saddle'],
        ]);
      }
      if (index === 2) {
        for (const side of shown.relax) trail(side, t.event, { dash: '2 5', sw: 2 });
        dot(reactant.position, 5.5, t.ink);
        dot(product.position, 5.5, t.ink);
        triangle(process.saddle, 8.5, t.refine);
        label(reactant.position, '0', -2, 26);
        label(product.position, String(stateNumber(shown.product)), 0, 28);
        const [gx, gy] = at(reactant.position);
        fig.glyph('check', gx - 24, gy - 2, 15, t.good);
        key([
          [(kx, ky) => fig.line(kx - 8, ky, kx + 8, ky, { stroke: t.event, sw: 2, dash: '2 5', cap: 'round' }), 'relaxation'],
          [(kx, ky) => fig.glyph('check', kx, ky, 14, t.good), 'back to the start'],
        ]);
      }
      if (index === 3) {
        fig.group({ clip }, () => {
          for (const search of searches) {
            const good = search.status !== 'failed';
            fig.path(Figure.polyline(search.climb.map(at)), { stroke: good ? t.refine : t.critical, sw: 1.5, join: 'round', opacity: good ? 0.8 : 0.55 });
          }
        });
        dot(reactant.position, 5.5, t.ink);
        const counts = new Map();
        for (const search of searches) if (search.process) counts.set(search.process, (counts.get(search.process) ?? 0) + 1);
        for (const [id, count] of counts) {
          const p = eon.processes[id];
          triangle(p.saddle, 7.5, t.refine);
          dot(eon.states[p.product].position, 5, t.ink);
          const [cx, cy] = at(p.saddle);
          fig.text(cx + 16, cy + (p.saddle[1] > 0 ? -8 : 20), `×${count}`, { size: TYPE.small, weight: 700, halo });
        }
        label(reactant.position, '0', -2, 26);
        key([
          [(kx, ky) => fig.line(kx - 8, ky, kx + 8, ky, { stroke: t.refine, sw: 2, cap: 'round' }), 'reached a saddle'],
          [(kx, ky) => fig.line(kx - 8, ky, kx + 8, ky, { stroke: t.critical, sw: 2, cap: 'round' }), 'failed'],
        ]);
      }

      fig.badge(panel.x + 26, panel.y + 26, index + 1, { r: 13 });
      fig.text(panel.x + 48, panel.y + 26, titles[index], { size: TYPE.body, weight: 700, middle: true, halo });
    });

    // Energy along the process: the two relaxations from the saddle, joined there.
    const path = [...shown.relax[0].slice().reverse(), ...shown.relax[1].slice(1)];
    const lengths = path.reduce((acc, p, i) => [...acc, i ? acc.at(-1) + Math.hypot(p[0] - path[i - 1][0], p[1] - path[i - 1][1]) : 0], []);
    const e0 = reactant.energy_eV;
    const [left, right, top, bottom] = [922, 1160, 34, 276];
    const x = scale(0, lengths.at(-1), left, right);
    const y = scale(-0.4, 0.5, bottom, top);
    for (const tick of [-0.4, 0, 0.4]) {
      fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
      fig.text(left - 10, y(tick), tick === 0 ? '0' : tick.toFixed(1), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
    }
    fig.line(left, bottom, right, bottom, { stroke: t.axis });
    for (let tick = 0; tick <= lengths.at(-1); tick += 0.5) {
      fig.line(x(tick), bottom, x(tick), bottom + 6, { stroke: t.axis });
      fig.text(x(tick), bottom + 30, tick ? tick.toFixed(1) : '0', { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    }
    fig.text(right, bottom + 62, 'path (Å)', { size: TYPE.small, anchor: 'end', fill: t.ink2 });
    fig.text(left - 58, (top + bottom) / 2, 'E − E(0) (eV)', { size: TYPE.small, anchor: 'middle', fill: t.ink2, rotate: -90 });
    const points = path.map((p, i) => [x(lengths[i]), y(energy(p[0], p[1]) - e0)]);
    fig.path(Figure.smooth(points, 0.35), { stroke: t.refine, sw: 2.25 });
    const peak = shown.relax[0].length - 1;
    for (const [i, kind] of [[0, 'min'], [peak, 'saddle'], [points.length - 1, 'min']]) {
      const [px, py] = points[i];
      fig.circle(px, py, 7.5, { fill: t.bg });
      if (kind === 'saddle') fig.polygon([[px, py - 10], [px + 9, py + 7], [px - 9, py + 7]], { fill: t.refine, stroke: t.bg, sw: 2, join: 'round' });
      else fig.circle(px, py, 5.5, { fill: t.ink });
    }
    fig.text(points[0][0], points[0][1] + 28, '0', { size: TYPE.small, weight: 700, anchor: 'middle' });
    fig.text(points.at(-1)[0], points.at(-1)[1] + 28, String(stateNumber(shown.product)), { size: TYPE.small, weight: 700, anchor: 'middle' });
    const bx = points[peak][0] + 30;
    fig.line(points[0][0] + 10, y(0), bx + 8, y(0), { stroke: t.ink3, sw: 1, dash: '4 4' });
    fig.line(points[peak][0] + 10, points[peak][1], bx + 8, points[peak][1], { stroke: t.ink3, sw: 1, dash: '4 4' });
    fig.arrow([[bx, y(0)], [bx, points[peak][1] + 1]], { color: t.ink, sw: 1.5, size: 8, both: true });
    fig.text(bx + 10, y(process.barrier_eV * 0.75), 'ΔE‡', { size: TYPE.body, weight: 700, middle: true, halo: true });

    // What the search returned.
    const card = { x: 852, y: 400, w: 312, h: 138 };
    fig.card(card.x, card.y, card.w, card.h);
    const mantissa = (process.prefactor_s / 10 ** Math.floor(Math.log10(process.prefactor_s))).toFixed(1);
    const rows = [
      ['check', [{ t: shown.status, mono: true, weight: 600 }]],
      [null, [{ t: 'ΔE‡ ', weight: 700 }, `${process.barrier_eV.toFixed(3)} eV`]],
      [null, [{ t: 'ν ', weight: 700 }, `${mantissa} × 10`, { t: String(Math.floor(Math.log10(process.prefactor_s))), sup: true }, ' s', { t: '−1', sup: true }]],
    ];
    rows.forEach(([glyph, text], i) => {
      const ry = card.y + 32 + i * 37;
      if (glyph) fig.glyph(glyph, card.x + 26, ry, 14, t.good);
      fig.text(card.x + 46, ry, text, { size: TYPE.small, middle: true });
    });
  },
};
