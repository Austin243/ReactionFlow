// README banner: the two ways ReactionFlow finds reactions, each on a model landscape, and the loop
// each one runs.

import { Figure, HERO_TYPE as TYPE } from '../lib.mjs';
import { DOMAIN, energy, eon, LEVELS, stateNumber } from '../landscape.mjs';
import { drawSurface, pathway } from '../surface.mjs';
import { firstCrossing, langevin } from '../toy.mjs';
import { mark } from './hero.mjs';

// The MD view has the aspect ratio of the EON view, so the two panels are the same size.
const MD_DOMAIN = { x0: -1.72, x1: 1.92, y0: -0.754, y1: 1.5102 };
const [PW, PH] = [545, 339];
const COLUMNS = [36, 619];

// Illustrative trajectory, cut around its crossing, and the observation where the detector's
// persistence rule would confirm the change.
const all = langevin({ seed: 19, steps: 40000 });
const hop = firstCrossing(all);
const trajectory = all.slice(hop - 520, hop + 430);
let confirmed = null;
for (let i = Math.ceil((hop - 520) / 10) * 10, run = 0; i < hop + 430 && !confirmed; i += 10) {
  run = Math.hypot(all[i][0] - pathway.anchor[0], all[i][1] - pathway.anchor[1]) >= pathway.thresholds_A[1] ? run + 1 : 0;
  if (run === 3) confirmed = all[i];
}

function molecularDynamics(fig, rect) {
  const t = fig.t;
  const { sx, sy } = drawSurface(fig, 'overview-md', rect, { domain: MD_DOMAIN, columns: 150, rx: 8 });
  const at = ([x, y]) => [sx(x), sy(y)];
  const clip = fig.clipRect('overview-md-trace', rect.x, rect.y, rect.w, rect.h, 8);
  fig.path(Figure.polyline(trajectory.map(at)), { stroke: t.md, sw: 1.2, opacity: 0.82, join: 'round', cap: 'round', clip });
  const band = pathway.images.map(at);
  fig.path(Figure.polyline(band), { stroke: t.bg, sw: 7, opacity: 0.7, join: 'round' });
  fig.path(Figure.polyline(band), { stroke: t.refine, sw: 3.4, join: 'round' });
  band.forEach(([x, y], i) => {
    const end = i === 0 || i === band.length - 1;
    if (i === pathway.frequency.saddle_index) fig.saddle(x, y, 10);
    else fig.circle(x, y, end ? 7 : 6, { fill: end ? t.ink : t.refine, stroke: t.bg, sw: 2.3 });
  });
  if (confirmed) {
    const [x, y] = at(confirmed);
    fig.glow(x, y, 24, t.event);
    fig.circle(x, y, 8, { stroke: t.event, sw: 3 });
  }
  const label = (x, y, s, o = {}) => fig.text(x, y, s, { size: TYPE.small, weight: 650, halo: fig.tint(t.contour, 0.16), ...o });
  label(sx(-1.64), sy(0.56), 'Reactant');
  label(sx(1.84), sy(0.56), 'Product', { anchor: 'end' });
  const saddle = band[pathway.frequency.saddle_index];
  label(saddle[0], saddle[1] + 40, 'Saddle', { anchor: 'middle' });
}

function kineticMonteCarlo(fig, rect) {
  const t = fig.t;
  const { sx, sy } = drawSurface(fig, 'overview-eon', rect, { domain: DOMAIN, field: energy, levels: LEVELS, columns: 150, rx: 8 });
  const at = ([x, y]) => [sx(x), sy(y)];
  const halo = fig.tint(t.contour, 0.18);
  const forward = Object.entries(eon.processes).filter(([id]) => !id.endsWith('r'));

  // Every process found, then the steps taken: one strand per step, repeats of a hop side by side.
  for (const [, p] of forward) {
    fig.path(Figure.polyline([eon.states[p.state].position, p.saddle, eon.states[p.product].position].map(at)), { stroke: t.ink3, sw: 1.5, join: 'round' });
  }
  const seen = new Map();
  for (const step of eon.steps) {
    const p = eon.processes[step.process];
    const [a, s, b] = [eon.states[step.from].position, p.saddle, eon.states[step.to].position].map(at);
    const key = [step.from, step.to].sort().join();
    const n = seen.get(key) ?? 0;
    seen.set(key, n + 1);
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const [nx, ny] = [-(b[1] - a[1]) / len, (b[0] - a[0]) / len];
    const side = (step.from < step.to ? 1 : -1) * (4 + 4 * Math.floor(n / 2));
    const mid = [s[0] + nx * side, s[1] + ny * side];
    const reach = Math.hypot(b[0] - mid[0], b[1] - mid[1]);
    const end = [b[0] - ((b[0] - mid[0]) / reach) * 10, b[1] - ((b[1] - mid[1]) / reach) * 10];
    fig.path(Figure.smooth([a, mid, end]), { stroke: t.md, sw: 2.2, cap: 'round' });
    fig.head(end[0], end[1], Math.atan2(end[1] - mid[1], end[0] - mid[0]), { color: t.md, size: 8 });
  }
  for (const [, p] of forward) fig.saddle(...at(p.saddle), 6.5, { sw: 1.5 });

  // States with their numbers; the starting structure relaxes into state 0.
  const start = at(eon.start);
  fig.arrow([start, at(eon.states['state-000000'].position)], { color: t.ink, sw: 1.5, size: 7 });
  fig.polygon([[start[0], start[1] - 6], [start[0] + 6, start[1]], [start[0], start[1] + 6], [start[0] - 6, start[1]]], { fill: t.bg, stroke: t.ink, sw: 1.75, join: 'round' });
  fig.text(start[0] - 8, start[1] - 18, 'Start', { size: TYPE.small, weight: 650, anchor: 'end', halo });
  for (const [id, state] of Object.entries(eon.states)) {
    const [x, y] = at(state.position);
    fig.circle(x, y, 6.5, { fill: t.ink, stroke: t.bg, sw: 2 });
    fig.text(x, y + 30, String(stateNumber(id)), { size: TYPE.small, weight: 700, anchor: 'middle', halo });
  }
}

/** Boxes joined by arrows across [x, x + w], each as wide as its label needs plus an equal share of the rest. */
function flow(fig, x, y, w, items) {
  const gap = 26;
  const natural = items.map(([label]) => fig.measure(label, { size: TYPE.body, weight: 650 }));
  const pad = (w - gap * (items.length - 1) - natural.reduce((a, b) => a + b, 0)) / items.length;
  items.forEach(([label, color], i) => {
    const bw = natural[i] + pad;
    fig.rect(x, y, bw, 48, { rx: 7, fill: fig.tint(color, 0.12), stroke: fig.tint(color, 0.6) });
    fig.text(x + bw / 2, y + 24, label, { size: TYPE.body, weight: 650, anchor: 'middle', middle: true });
    if (i < items.length - 1) fig.arrow([[x + bw + 5, y + 24], [x + bw + gap - 5, y + 24]], { color: fig.t.ink3, sw: 1.75, size: 7 });
    x += bw + gap;
  });
}

export default {
  name: 'overview',
  title: 'ReactionFlow',
  label: 'Overview',
  alt:
    'ReactionFlow finds reactions in two ways, both shown on model energy landscapes. In molecular dynamics a ' +
    'trajectory crosses from the reactant basin to the product basin, the bond change is confirmed near the ' +
    'saddle, and the refined path runs over the saddle. The loop is run MD, detect, refine, and resume. With EON, ' +
    'saddle searches find the states and saddles of the landscape and kinetic Monte Carlo steps move from the ' +
    'starting well to the deepest one. The loop is search saddles, compute their rates, and take a kinetic Monte Carlo step.',
  caption:
    'Banner for the top of the README. Left, an illustrative Langevin trajectory on the model surface and the path ' +
    'that <code>refine_pathway()</code> refined there. Right, the states, saddles, and kinetic Monte Carlo steps ' +
    'of the EON model run.',
  width: 1200,
  height: 728,
  draw(fig) {
    const t = fig.t;

    // Name and the three things every run has.
    mark(fig, 36, 30, 72);
    fig.text(126, 89, 'ReactionFlow', { size: 60, weight: 800, spacing: -1 });
    fig.text(38, 150, 'Automated reaction discovery and pathway refinement', { size: TYPE.head, fill: t.ink2 });
    const pills = [['Any MLIP', t.md], ['NVT or NPT', t.refine], ['Exact restart', t.event]];
    const widths = pills.map(([label]) => fig.measure(label, { size: TYPE.small, weight: 500 }) + 2 * TYPE.small * 0.72 + TYPE.small * 0.62 + 6);
    let px = 1164 - widths.reduce((a, b) => a + b, 0) - 10 * (pills.length - 1);
    pills.forEach(([label, color], i) => {
      fig.pill(px, 66, label, { size: TYPE.small, dot: color, fill: t.panel, stroke: t.border });
      px += widths[i] + 10;
    });
    fig.line(36, 180, 1164, 180, { stroke: t.border });

    // Two panels of the same size, their legends, and the loop each one runs.
    const top = 250;
    fig.heading(COLUMNS[0], 226, 'a', 'Molecular dynamics');
    fig.heading(COLUMNS[1], 226, 'b', 'Rare events with EON');
    molecularDynamics(fig, { x: COLUMNS[0], y: top, w: PW, h: PH });
    kineticMonteCarlo(fig, { x: COLUMNS[1], y: top, w: PW, h: PH });

    const ky = top + PH + 34;
    fig.keys(COLUMNS[0] + 4, ky, [
      [(x, y) => fig.path(`M${x - 11} ${y + 3} l5 -6 l5 6 l5 -6 l6 6`, { stroke: t.md, sw: 1.8, join: 'round', cap: 'round' }), 'MD'],
      [(x, y) => fig.circle(x, y, 6, { stroke: t.event, sw: 2.5 }), 'Bond change'],
      [(x, y) => { fig.line(x - 11, y, x + 11, y, { stroke: t.refine, sw: 2.5 }); fig.circle(x, y, 4, { fill: t.refine }); }, 'Refined path'],
    ]);
    fig.keys(COLUMNS[1] + 4, ky, [
      [(x, y) => fig.circle(x, y, 5.5, { fill: t.ink }), 'State'],
      [(x, y) => fig.saddle(x, y, 6.5, { sw: 0 }), 'Saddle'],
      [(x, y) => fig.arrow([[x - 11, y], [x + 11, y]], { color: t.md, sw: 2.5, size: 8 }), 'KMC step'],
    ]);

    const fy = ky + 32;
    flow(fig, COLUMNS[0], fy, PW, [['Run MD', t.md], ['Detect', t.event], ['Refine', t.refine], ['Resume', t.md]]);
    flow(fig, COLUMNS[1], fy, PW, [['Search saddles', t.refine], ['Rates', t.event], ['KMC step', t.md]]);
  },
};
