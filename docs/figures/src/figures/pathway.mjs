// Pathway refinement in four stages, drawn from a real refine_pathway() run on a model potential.

import { Figure, scale, TYPE } from '../lib.mjs';
import { DOMAIN, drawSurface, pathway } from '../surface.mjs';

const [relaxR, relaxP, neb, climb, sideA, sideB] = pathway.runs.map((run) => run.history);
const saddleIndex = pathway.frequency.saddle_index;
const mode = pathway.frequency.primary_mode[1];
const imaginary = Math.abs(Math.min(...pathway.frequency.frequencies_cm1));

export default {
  name: 'pathway',
  title: 'Pathway refinement',
  alt:
    'Four views of the same model energy surface: two molecular-dynamics snapshots relax to the reactant and product ' +
    'minima, an interpolated band of seven images relaxes onto the minimum-energy path, the highest image climbs to the ' +
    'saddle, and the saddle displaced both ways along its imaginary mode relaxes to both endpoints. A chart shows the ' +
    'energy of each image, with a barrier of 0.62 eV.',
  caption:
    'The stages of <code>refine_pathway()</code>, drawn from an actual run on a two-dimensional model potential with the ' +
    'default settings. The recorded outcome is on the right.',
  width: 1200,
  height: 556,
  draw(fig) {
    const t = fig.t;
    // The default view, extended downward so each legend has room under the band.
    const domain = { ...DOMAIN, y0: DOMAIN.y0 - 0.24 };
    const [PW, PH] = [384, 253];
    const panels = [
      { x: 36, y: 16 },
      { x: 436, y: 16 },
      { x: 36, y: 285 },
      { x: 436, y: 285 },
    ];
    const titles = ['Relax endpoints', 'Nudged elastic band', 'Climbing image', 'Saddle checks'];

    panels.forEach((panel, index) => {
      const { sx, sy } = drawSurface(fig, 'model', { x: panel.x, y: panel.y, w: PW, h: PH }, { domain });
      const at = ([px, py]) => [sx(px), sy(py)];
      const dot = (p, r, fill, o = {}) => {
        const [cx, cy] = at(p);
        fig.circle(cx, cy, r + 2, { fill: t.bg });
        fig.circle(cx, cy, r, { fill, ...o });
      };
      const hollow = (p, r, stroke) => {
        const [cx, cy] = at(p);
        fig.circle(cx, cy, r, { fill: t.bg, stroke, sw: 1.75 });
      };
      const diamond = (p, r) => {
        const [cx, cy] = at(p);
        fig.polygon([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], { fill: t.bg, stroke: t.ink, sw: 1.75, join: 'round' });
      };
      const triangle = (p, r, fill) => {
        const [cx, cy] = at(p);
        fig.polygon([[cx, cy - r * 1.15], [cx + r, cy + r * 0.75], [cx - r, cy + r * 0.75]], { fill, stroke: t.bg, sw: 2, join: 'round' });
      };
      const trail = (history, color, o = {}) => {
        // Follow the optimizer's steps, stopping at the ring of the marker the path ends on.
        const points = history.map((step) => at(step[0]));
        const end = points.at(-1);
        const shown = points.slice(0, points.findLastIndex(([px, py]) => Math.hypot(px - end[0], py - end[1]) > 16) + 1);
        const last = shown.at(-1);
        const reach = Math.hypot(end[0] - last[0], end[1] - last[1]);
        shown.push([end[0] - ((end[0] - last[0]) / reach) * 9, end[1] - ((end[1] - last[1]) / reach) * 9]);
        fig.arrow(shown, { color, sw: 1.75, size: 8, dash: o.dash });
      };
      const band = (images, color, o = {}) =>
        fig.path(Figure.polyline(images.map(at)), { stroke: color, sw: o.sw ?? 2, join: 'round', cap: 'round', dash: o.dash, opacity: o.opacity });
      const endpoints = () => {
        for (const [p, label, dx] of [[pathway.images[0], 'R', -17], [pathway.images.at(-1), 'P', 17]]) {
          dot(p, 5, t.ink);
          const [cx, cy] = at(p);
          fig.text(cx + dx * 1.2, cy + 18, label, { size: TYPE.body, weight: 700, anchor: 'middle', halo: fig.tint(t.contour, 0.18) });
        }
      };
      const key = (rows) => {
        // Legend in the flat high-energy region under the arch.
        rows.forEach(([draw, label], i) => {
          const [kx, ky] = [sx(-0.42), sy(-0.36) + i * 28];
          draw(kx, ky);
          fig.text(kx + 16, ky, label, { size: TYPE.small, middle: true, fill: t.ink });
        });
      };

      if (index === 0) {
        trail(relaxR, t.ink);
        trail(relaxP, t.ink);
        diamond(pathway.snapshots.reactant, 6);
        diamond(pathway.snapshots.product, 6);
        endpoints();
        key([
          [(kx, ky) => fig.polygon([[kx, ky - 6], [kx + 6, ky], [kx, ky + 6], [kx - 6, ky]], { fill: t.bg, stroke: t.ink, sw: 1.75, join: 'round' }), 'MD snapshot'],
          [(kx, ky) => fig.circle(kx, ky, 5, { fill: t.ink }), 'relaxed endpoint'],
        ]);
      }
      if (index === 1) {
        for (const step of [5, 10, 16]) band(neb[step], t.refine, { sw: 1.25, opacity: 0.45 });
        band(neb[0], t.ink2, { sw: 1.5, dash: '5 4' });
        neb[0].slice(1, -1).forEach((p) => hollow(p, 3.5, t.ink2));
        band(neb.at(-1), t.refine);
        neb.at(-1).slice(1, -1).forEach((p) => dot(p, 4.5, t.refine));
        endpoints();
        key([
          [(kx, ky) => fig.circle(kx, ky, 3.5, { fill: t.bg, stroke: t.ink2, sw: 1.75 }), 'interpolated'],
          [(kx, ky) => fig.circle(kx, ky, 4.5, { fill: t.refine }), 'converged band'],
        ]);
      }
      if (index === 2) {
        band(climb.at(-1), t.refine);
        climb.at(-1).slice(1, -1).forEach((p, i) => i + 1 !== saddleIndex && dot(p, 4.5, t.refine));
        hollow(climb[0][saddleIndex], 4.5, t.refine);
        triangle(climb.at(-1)[saddleIndex], 8, t.refine);
        const [from, to] = [at(climb[0][saddleIndex]), at(climb.at(-1)[saddleIndex])];
        fig.curve([from[0], from[1] - 9], [from[0] - 2, from[1] - 30], [to[0] + 6, to[1] - 34], [to[0] + 1, to[1] - 14], { color: t.ink, sw: 1.5, size: 7 });
        endpoints();
        key([
          [(kx, ky) => fig.circle(kx, ky, 4.5, { fill: t.bg, stroke: t.refine, sw: 1.75 }), 'before climbing'],
          [(kx, ky) => fig.polygon([[kx, ky - 8], [kx + 7, ky + 5], [kx - 7, ky + 5]], { fill: t.refine }), 'climbing image'],
        ]);
      }
      if (index === 3) {
        band(climb.at(-1), t.refine, { sw: 1.5, opacity: 0.4 });
        trail(sideA, t.event, { dash: '2 5' });
        trail(sideB, t.event, { dash: '2 5' });
        const saddle = at(pathway.images[saddleIndex]);
        const reach = 0.3 * (PW / 3.64);
        fig.arrow(
          [[saddle[0] - mode[0] * reach, saddle[1] + mode[1] * reach], [saddle[0] + mode[0] * reach, saddle[1] - mode[1] * reach]],
          { color: t.event, sw: 2.25, size: 9, both: true },
        );
        triangle(pathway.images[saddleIndex], 8, t.refine);
        endpoints();
        fig.text(saddle[0], saddle[1] + 44, `${imaginary.toFixed(0)}i cm⁻¹`, { size: TYPE.small, weight: 600, anchor: 'middle', halo: fig.tint(t.contour, 0.18) });
        key([
          [(kx, ky) => fig.arrow([[kx - 8, ky], [kx + 8, ky]], { color: t.event, sw: 2, size: 6, both: true }), 'imaginary mode'],
          [(kx, ky) => fig.line(kx - 8, ky, kx + 8, ky, { stroke: t.event, sw: 1.75, dash: '2 5', cap: 'round' }), 'relaxation'],
        ]);
      }

      fig.badge(panel.x + 26, panel.y + 26, index + 1, { r: 13 });
      fig.text(panel.x + 48, panel.y + 26, titles[index], { size: TYPE.body, weight: 700, middle: true, halo: fig.tint(t.contour, 0.18) });
    });

    // Energy of each image after climbing.
    const cx0 = 852;
    const [left, right, top, bottom] = [cx0 + 70, 1160, 34, 276];
    const x = scale(-0.35, 6.35, left, right);
    const y = scale(-1.2, 0.8, bottom, top);
    for (const tick of [-1, -0.5, 0, 0.5]) {
      fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
      fig.text(left - 10, y(tick), tick === 0 ? '0' : tick.toFixed(1), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
    }
    fig.line(left, bottom, right, bottom, { stroke: t.axis });
    ['R', '1', '2', '3', '4', '5', 'P'].forEach((label, i) =>
      fig.text(x(i), bottom + 28, label, { size: TYPE.small, anchor: 'middle', fill: t.ink2, weight: i % 6 ? 400 : 700 }),
    );
    fig.text((left + right) / 2, bottom + 58, 'image', { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    fig.text(cx0 + 4, (top + bottom) / 2, 'E − E(R) (eV)', { size: TYPE.small, anchor: 'middle', fill: t.ink2, rotate: -90 });
    const zero = pathway.energies_eV[0];
    const after = pathway.energies_eV.map((e, i) => [x(i), y(e - zero)]);
    fig.path(Figure.polyline(after), { stroke: t.refine, sw: 2.25, join: 'round' });
    after.forEach(([px, py], i) => {
      fig.circle(px, py, 7.5, { fill: t.bg });
      if (i === saddleIndex) fig.polygon([[px, py - 10], [px + 9, py + 7], [px - 9, py + 7]], { fill: t.refine, stroke: t.bg, sw: 2, join: 'round' });
      else fig.circle(px, py, 5.5, { fill: i % 6 ? t.refine : t.ink });
    });
    const bx = x(saddleIndex) + 30;
    fig.line(x(0) + 10, y(0), bx + 8, y(0), { stroke: t.ink3, sw: 1, dash: '4 4' });
    fig.arrow([[bx, y(0)], [bx, after[saddleIndex][1] + 1]], { color: t.ink, sw: 1.5, size: 8, both: true });
    fig.text(bx + 10, y(pathway.barrier_eV / 2) - 12, 'ΔE‡', { size: TYPE.body, weight: 700, middle: true });
    fig.text(bx + 10, y(pathway.barrier_eV / 2) + 14, `${pathway.barrier_eV.toFixed(2)} eV`, { size: TYPE.body, weight: 700, middle: true });

    // What the run recorded.
    const card = { x: cx0, y: 400, w: 312, h: 138 };
    fig.card(card.x, card.y, card.w, card.h);
    [pathway.status, pathway.frequency.status, pathway.connectivity.status].forEach((value, i) => {
      const ry = card.y + 32 + i * 36;
      fig.glyph('check', card.x + 26, ry, 14, t.good);
      fig.text(card.x + 46, ry, value, { size: TYPE.small, mono: true, weight: 600, middle: true });
    });
  },
};
