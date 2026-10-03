// Banner: an MD trajectory that crosses between basins, and the refined path for that event.

import { Figure, TYPE } from '../lib.mjs';
import { drawSurface, pathway } from '../surface.mjs';
import { firstCrossing, langevin } from '../toy.mjs';

const DOMAIN = { x0: -1.72, x1: 1.92, y0: -0.66, y1: 1.413 };
const OBSERVATION_FS = 10;
const PERSISTENCE = 3;

// Illustrative trajectory: Langevin dynamics on the same model surface, cut around its crossing.
const all = langevin({ seed: 19, steps: 40000 });
const hop = firstCrossing(all);
const start = hop - 520;
const trajectory = all.slice(start, hop + 430);

// The frame where the detector's persistence rule would confirm the broken bond.
const beyond = ([x, y]) => Math.hypot(x - pathway.anchor[0], y - pathway.anchor[1]) >= pathway.thresholds_A[1];
let confirmedIndex = null;
for (let i = Math.ceil(start / OBSERVATION_FS) * OBSERVATION_FS, run = 0; i < hop + 430 && confirmedIndex === null; i += OBSERVATION_FS) {
  run = beyond(all[i]) ? run + 1 : 0;
  if (run === PERSISTENCE) confirmedIndex = i - start;
}
const confirmed = confirmedIndex === null ? null : trajectory[confirmedIndex];

// Animated variant: the trajectory draws itself, the ring appears when it reaches the confirmed
// frame, then the refined path fades in. One loop lasts LOOP seconds.
const [LOOP, DRAW, FADE] = [12, 6, 1];

/** Small mark: a path over a barrier between two states. */
export function mark(fig, x, y, size) {
  const t = fig.t;
  const s = size / 56;
  fig.rect(x, y, size, size, { rx: 13 * s, fill: fig.tint(t.refine, 0.14), stroke: fig.tint(t.refine, 0.5), sw: 1 });
  const [ax, ay, bx, by, px, py] = [x + 13 * s, y + 39 * s, x + 43 * s, y + 39 * s, x + 28 * s, y + 17 * s];
  fig.path(`M${ax} ${ay} C${ax + 6 * s} ${ay - 4 * s} ${px - 11 * s} ${py} ${px} ${py} C${px + 11 * s} ${py} ${bx - 6 * s} ${by - 4 * s} ${bx} ${by}`, {
    stroke: t.refine,
    sw: 3 * s,
    cap: 'round',
  });
  fig.circle(ax, ay, 4.6 * s, { fill: t.ink });
  fig.circle(bx, by, 4.6 * s, { fill: t.ink });
  fig.polygon([[px, py - 7.5 * s], [px + 6.8 * s, py + 4.2 * s], [px - 6.8 * s, py + 4.2 * s]], { fill: t.refine, stroke: fig.tint(t.refine, 0.14), sw: 1.5 * s, join: 'round' });
}

const ALT =
  'ReactionFlow: automated reaction discovery and pathway refinement for atomistic trajectories. A molecular-dynamics ' +
  'trajectory on an energy surface leaves the reactant basin and settles in the product basin; the refined ' +
  'minimum-energy path between the two basins passes over the saddle point.';

function draw(fig, animated) {
    const t = fig.t;

    // Name, description, and three facts.
    mark(fig, 48, 48, 60);
    fig.text(44, 180, 'ReactionFlow', { size: 66, weight: 800, spacing: -1.2 });
    fig.lines(48, 226, ['Automated reaction discovery', 'and pathway refinement for', 'atomistic trajectories'], { size: TYPE.head, fill: t.ink2, lh: 34 });
    let px = 48;
    for (const [label, color] of [['any MLIP', t.md], ['NVT or NPT', t.refine], ['exact restart', t.event]]) {
      px += fig.pill(px, 346, label, { size: TYPE.small, dot: color, fill: t.panel, stroke: t.border }) + 10;
    }

    // Surface, trajectory, and refined path; the plot keeps the aspect ratio of DOMAIN.
    const rect = { x: 606, y: 41, w: 558, h: 318 };
    const { sx, sy } = drawSurface(fig, 'hero', rect, { domain: DOMAIN, columns: 150, rx: 10 });
    const at = ([x, y]) => [sx(x), sy(y)];
    const clip = fig.clipRect('hero-plot', rect.x, rect.y, rect.w, rect.h, 10);
    const points = trajectory.map(at);
    const lengths = points.map((p, i) => (i ? Math.hypot(p[0] - points[i - 1][0], p[1] - points[i - 1][1]) : 0));
    const total = lengths.reduce((a, b) => a + b, 0);
    const smil = (attribute, values, keyTimes, extra = '') =>
      animated ? `<animate attributeName="${attribute}" dur="${LOOP}s" repeatCount="indefinite" values="${values}" keyTimes="${keyTimes}" ${extra}/>` : '';
    const drawn = (DRAW / LOOP).toFixed(4);
    fig.add(
      `<path d="${Figure.polyline(points)}" fill="none" stroke="${t.md}" stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round" ` +
        `opacity="0.8" clip-path="url(#${clip})"${animated ? ` stroke-dasharray="${total.toFixed(1)}" stroke-dashoffset="${total.toFixed(1)}"` : ''}>` +
        `${smil('stroke-dashoffset', `${total.toFixed(1)};0;0`, `0;${drawn};1`)}</path>`,
    );
    const appear = (from, to) => smil('opacity', '0;0;1;1', `0;${from.toFixed(4)};${to.toFixed(4)};1`);

    fig.add(`<g${animated ? ' opacity="0"' : ''}>${appear(DRAW / LOOP, (DRAW + FADE) / LOOP)}`);
    const band = pathway.images.map(at);
    fig.path(Figure.polyline(band), { stroke: t.bg, sw: 6, join: 'round', cap: 'round', opacity: 0.55 });
    fig.path(Figure.polyline(band), { stroke: t.refine, sw: 2.75, join: 'round', cap: 'round' });
    band.forEach(([cx, cy], i) => {
      const end = i === 0 || i === band.length - 1;
      fig.circle(cx, cy, end ? 8.5 : 7.5, { fill: t.bg });
      if (i === pathway.frequency.saddle_index) {
        fig.polygon([[cx, cy - 11], [cx + 9.5, cy + 7], [cx - 9.5, cy + 7]], { fill: t.refine, stroke: t.bg, sw: 2, join: 'round' });
      } else {
        fig.circle(cx, cy, end ? 6.5 : 5.5, { fill: end ? t.ink : t.refine });
      }
    });
    fig.add('</g>');
    if (confirmed) {
      const [cx, cy] = at(confirmed);
      const reached = (lengths.slice(0, confirmedIndex + 1).reduce((a, b) => a + b, 0) / total) * (DRAW / LOOP);
      fig.add(`<g${animated ? ' opacity="0"' : ''}>${appear(reached, reached + 0.025)}`);
      fig.glow(cx, cy, 26, t.event);
      fig.circle(cx, cy, 8, { stroke: t.event, sw: 3 });
      fig.add('</g>');
    }

    // Labels sit on the flat, high-energy parts of the surface.
    const halo = fig.tint(t.contour, 0.16);
    const label = (x, y, s, o = {}) => fig.text(x, y, s, { size: TYPE.small, weight: 600, halo, ...o });
    const saddle = band[pathway.frequency.saddle_index];
    fig.add(`<g${animated ? ' opacity="0"' : ''}>${appear(DRAW / LOOP, (DRAW + FADE) / LOOP)}`);
    label(saddle[0] - 4, saddle[1] + 34, 'saddle', { anchor: 'middle' });
    fig.add('</g>');
    label(sx(-1.66), sy(0.33), 'reactant');
    label(sx(1.86), sy(0.33), 'product', { anchor: 'end' });
    const keyX = sx(-0.46);
    [
      [(y) => fig.path(`M${keyX - 11} ${y + 3} l5 -6 l5 6 l5 -6 l5 6`, { stroke: t.md, sw: 1.4, join: 'round', cap: 'round' }), 'MD trajectory'],
      [(y) => fig.circle(keyX, y, 6, { stroke: t.event, sw: 2.5 }), 'bond change confirmed'],
      [(y) => { fig.line(keyX - 11, y, keyX + 11, y, { stroke: t.refine, sw: 2.75, cap: 'round' }); fig.circle(keyX, y, 4.5, { fill: t.refine }); }, 'refined path'],
    ].forEach(([draw, text], i) => {
      const y = sy(-0.06) + i * 30;
      draw(y);
      fig.text(keyX + 22, y, text, { size: TYPE.small, middle: true });
    });
}

export default {
  name: 'hero',
  title: 'ReactionFlow',
  label: 'Banner',
  alt: ALT,
  caption:
    'Banner for the top of the README or the landing page. The trajectory is a short Langevin run on the same model ' +
    'surface as the pathway figure, and the green band is the refined path from that figure.',
  width: 1200,
  height: 400,
  draw: (fig) => draw(fig, false),
};

export const heroAnimated = {
  name: 'hero-animated',
  title: 'ReactionFlow',
  label: 'Banner (animated)',
  animated: true,
  alt: ALT,
  caption:
    'The same banner as an animated SVG: the trajectory draws itself, the ring appears where the bond change is ' +
    'confirmed, and the refined path fades in. It loops every 12 s.',
  width: 1200,
  height: 400,
  draw: (fig) => draw(fig, true),
};
