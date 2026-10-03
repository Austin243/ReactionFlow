// Animated replay of the band optimization in data/pathway.json: every frame is one optimizer
// step of the real run. Loops in any browser, including inside a GitHub README.

import { Figure, TYPE } from '../lib.mjs';
import { drawSurface, pathway } from '../surface.mjs';

const neb = pathway.runs[2].history;
const climb = pathway.runs[3].history;
const saddleIndex = pathway.frequency.saddle_index;

// Timeline in seconds: hold the first guess, relax, hold, climb, hold the result.
const [HOLD, RELAX, PAUSE, CLIMB, REST] = [1.2, 4.2, 0.9, 1.6, 2.6];
const TOTAL = HOLD + RELAX + PAUSE + CLIMB + REST;
const frames = [];
frames.push([0, neb[0]]);
neb.forEach((band, i) => frames.push([HOLD + (RELAX * i) / (neb.length - 1), band]));
climb.forEach((band, i) => frames.push([HOLD + RELAX + PAUSE + (CLIMB * i) / (climb.length - 1), band]));
frames.push([TOTAL, climb.at(-1)]);
const keyTimes = frames.map(([time]) => (time / TOTAL).toFixed(4)).join(';');

export default {
  name: 'band-animation',
  title: 'A band relaxing onto the path',
  label: 'Band relaxation (animated)',
  animated: true,
  alt:
    'Animation on a model energy surface: seven images start on a straight line between the reactant and product ' +
    'minima, relax onto the curved minimum-energy path, and the highest image then climbs to the saddle point.',
  caption:
    'An animated SVG that replays the optimizer steps of the run behind the pathway figure: nudged elastic band, then ' +
    'climbing image. It loops every 10.5 s and plays inside a GitHub README.',
  width: 760,
  height: 468,
  draw(fig) {
    const t = fig.t;
    const rect = { x: 20, y: 20, w: 720, h: 428 };
    const { sx, sy } = drawSurface(fig, 'anim', rect, { columns: 150, rx: 10 });
    const at = ([x, y]) => [sx(x), sy(y)];
    const animate = (attribute, values, extra = '') =>
      `<animate attributeName="${attribute}" dur="${TOTAL}s" repeatCount="indefinite" keyTimes="${keyTimes}" values="${values}" ${extra}/>`;
    const number = (v) => v.toFixed(1);

    // First guess, left in place as a reference.
    fig.path(Figure.polyline(neb[0].map(at)), { stroke: t.ink2, sw: 1.5, dash: '5 4' });

    // Band.
    const bands = frames.map(([, band]) => band.map(at).map(([x, y]) => `${number(x)},${number(y)}`).join(' '));
    fig.add(
      `<polyline points="${bands[0]}" fill="none" stroke="${t.refine}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round">` +
        `${animate('points', bands.join(';'))}</polyline>`,
    );

    // Images.
    const count = neb[0].length;
    for (let i = 0; i < count; i++) {
      const end = i === 0 || i === count - 1;
      const xs = frames.map(([, band]) => number(at(band[i])[0])).join(';');
      const ys = frames.map(([, band]) => number(at(band[i])[1])).join(';');
      const [x0, y0] = at(neb[0][i]);
      const moving = end ? '' : animate('cx', xs) + animate('cy', ys);
      fig.add(`<circle cx="${number(x0)}" cy="${number(y0)}" r="${end ? 9 : 8}" fill="${t.bg}">${moving}</circle>`);
      fig.add(`<circle cx="${number(x0)}" cy="${number(y0)}" r="${end ? 6.5 : 5.5}" fill="${end ? t.ink : t.refine}">${moving}</circle>`);
    }

    // Saddle marker and labels that follow the stages.
    const stageTimes = [0, HOLD, HOLD + RELAX + PAUSE, HOLD + RELAX + PAUSE + CLIMB, TOTAL].map((time) => (time / TOTAL).toFixed(4));
    const during = (index) =>
      `<animate attributeName="opacity" dur="${TOTAL}s" repeatCount="indefinite" calcMode="discrete" keyTimes="${stageTimes.join(';')}" ` +
      `values="${[0, 1, 2, 3, 3].map((stage) => (stage === index ? 1 : 0)).join(';')}"/>`;
    const [cx, cy] = at(climb.at(-1)[saddleIndex]);
    fig.add(
      `<polygon points="${number(cx)},${number(cy - 13)} ${number(cx + 11)},${number(cy + 8)} ${number(cx - 11)},${number(cy + 8)}" ` +
        `fill="${t.refine}" stroke="${t.bg}" stroke-width="2" stroke-linejoin="round" opacity="0">${during(3)}</polygon>`,
    );
    const halo = fig.tint(t.contour, 0.16);
    [
      'Interpolated guess',
      'Nudged elastic band',
      'Climbing image',
      `Saddle: ΔE‡ = ${pathway.barrier_eV.toFixed(2)} eV`,
    ].forEach((label, index) => {
      fig.add(`<g opacity="${index ? 0 : 1}">${during(index)}`);
      fig.text(rect.x + 24, rect.y + 40, label, { size: TYPE.body, weight: 700, halo });
      fig.add('</g>');
    });
    for (const [p, label, dx] of [[pathway.images[0], 'R', -20], [pathway.images.at(-1), 'P', 20]]) {
      const [px, py] = at(p);
      fig.text(px + dx, py + 18, label, { size: TYPE.body, weight: 700, anchor: 'middle', halo });
    }
  },
};
