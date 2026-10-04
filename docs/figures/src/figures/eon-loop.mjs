// EON mode on a timeline: searches from the current state until its confidence reaches the
// target, then a kinetic Monte Carlo step to the next state. The timeline replays the start of the
// model run in data/eon.json; the cards below name the parts of one search and of one step.

import { Figure, TYPE } from '../lib.mjs';
import { duration, eon, events, stateNumber } from '../landscape.mjs';

// The timeline ends after this many KMC steps.
const SHOWN = 3;
const run = events();
const shown = run.slice(0, run.findIndex((e) => e.kind === 'step' && e.index === SHOWN - 1) + 1);

export default {
  name: 'eon-loop',
  title: 'How EON mode works',
  alt:
    'Timeline of the start of an EON run. Searches from the current state find new processes, repeats of known ' +
    'ones, and failures. The confidence rises with each repeat in a row and drops back to zero at each new ' +
    'process. When it reaches 0.95, a kinetic Monte Carlo step moves to the next state and advances the clock. ' +
    'Five cards name the push, climb, relax, prefactor, and step.',
  caption:
    'One search pushes the state, climbs to a saddle with the dimer method, relaxes both sides, and keeps the ' +
    'process if one side is the starting state. EON then computes its harmonic prefactor. Searches repeat until ' +
    'the confidence target, then a kinetic Monte Carlo step picks a process by rate and advances the clock. The ' +
    'timeline is the start of the model run in the other EON figures.',
  width: 1200,
  height: 548,
  draw(fig) {
    const t = fig.t;

    // --- Timeline -------------------------------------------------------------------------
    // Searches and steps fill [x0, x1]; the run continues to the right edge.
    const [x0, x1, edge] = [228, 1084, 1164];
    const lanes = [
      { y: 92, label: 'Searches', color: t.event },
      { y: 152, label: 'Confidence', color: t.ink3 },
      { y: 210, label: 'State', color: t.md },
    ];
    for (const lane of lanes) {
      fig.circle(44, lane.y, 6, { fill: lane.color });
      fig.text(58, lane.y, lane.label, { size: TYPE.body, weight: 600, middle: true });
    }
    fig.line(x0, lanes[0].y, edge, lanes[0].y, { stroke: t.grid });

    // Searches advance by one pitch; a step takes a fixed gap.
    const searches = shown.filter((e) => e.kind === 'search').length;
    const gap = 26;
    const pitch = (x1 - x0 - SHOWN * gap) / searches;
    let cursor = x0;
    const xs = shown.map((e) => {
      const width = e.kind === 'search' ? pitch : gap;
      cursor += width;
      return cursor - width / 2;
    });

    // Legend for the marks and the step lines.
    const mark = (status, x, y) => {
      if (status === 'new') fig.circle(x, y, 6.5, { fill: t.refine });
      else if (status === 'repeat') fig.line(x, y - 8, x, y + 8, { stroke: t.ink3, sw: 1.75, cap: 'round' });
      else if (status === 'failed') fig.glyph('cross', x, y, 11, t.critical);
      else fig.circle(x, y, 5, { fill: t.warn });
    };
    let lx = x0;
    for (const status of ['new', 'repeat', 'failed']) {
      mark(status, lx + 6, 34);
      fig.text(lx + 20, 34, status, { size: TYPE.small, mono: true, weight: 600, middle: true });
      lx += 20 + fig.measure(status, { size: TYPE.small, mono: true, weight: 600 }) + 34;
    }
    fig.line(lx + 6, 22, lx + 6, 46, { stroke: t.ink3, sw: 1.25, dash: '4 4' });
    fig.text(lx + 20, 34, 'KMC step', { size: TYPE.small, middle: true });

    // Step boundaries, with the clock's advance under each.
    const ly = lanes[2].y;
    shown.forEach((e, i) => {
      if (e.kind !== 'step') return;
      fig.line(xs[i], 62, xs[i], ly + 22, { stroke: t.ink3, sw: 1.25, dash: '4 4' });
      fig.text(xs[i], ly + 46, `+${duration(e.step.dt_s)}`, { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    });

    // One mark per search, by outcome.
    shown.forEach((e, i) => e.kind === 'search' && mark(e.attempt.status, xs[i], lanes[0].y));

    // Confidence of the current state after each search, on a 0..1 band.
    const [cTop, cBottom] = [lanes[1].y - 22, lanes[1].y + 22];
    const cy = (v) => cBottom - v * (cBottom - cTop);
    const target = eon.akmc.confidence;
    fig.line(x0, cy(target), x1, cy(target), { stroke: t.md, sw: 1.25, dash: '5 4' });
    fig.text(x1 + 12, cy(target), String(target), { size: TYPE.small, weight: 700, middle: true, fill: t.md });
    fig.line(x0, cBottom, x1, cBottom, { stroke: t.grid });
    let segment = [];
    const flush = () => {
      if (segment.length > 1) fig.path(Figure.polyline(segment), { stroke: t.ink2, sw: 2, join: 'round' });
      segment = [];
    };
    shown.forEach((e, i) => {
      if (e.kind === 'step') return flush();
      segment.push([xs[i], cy(e.confidence)]);
    });
    flush();

    // The current state as a bar per visit, then the run continues.
    const bar = 28;
    let from = x0;
    shown.forEach((e, i) => {
      if (e.kind !== 'step') return;
      const to = xs[i] - gap / 2 + 2;
      fig.rect(from, ly - bar / 2, to - from, bar, { rx: 7, fill: t.md });
      fig.text((from + to) / 2, ly, `state ${stateNumber(e.step.from)}`, { size: TYPE.small, weight: 600, anchor: 'middle', middle: true, fill: '#ffffff' });
      from = xs[i] + gap / 2 - 2;
    });
    fig.path(`M${from + 7} ${ly - bar / 2} H${edge - 14} L${edge} ${ly} L${edge - 14} ${ly + bar / 2} H${from + 7} a7 7 0 0 1 -7 -7 v${-(bar - 14)} a7 7 0 0 1 7 -7 z`, { fill: t.md, opacity: 0.35 });

    // --- Cards ----------------------------------------------------------------------------
    const cards = [
      { title: 'Push', color: t.event, note: 'random displacement' },
      { title: 'Climb', color: t.refine, note: 'dimer to a saddle' },
      { title: 'Relax', color: t.event, note: 'one side is the start' },
      { title: 'Prefactor', color: t.refine, note: 'harmonic TST' },
      { title: 'Step', color: t.md, note: 'pick by rate' },
    ];
    const [cw, ch, cg, cy0] = [216, 176, 12, 280];
    cards.forEach((card, i) => {
      const cx = 36 + i * (cw + cg);
      fig.card(cx, cy0, cw, ch);
      fig.path(`M${cx + 10} ${cy0 + 0.5} h${cw - 20}`, { stroke: card.color, sw: 3, cap: 'round' });
      const [mx, my] = [cx + cw / 2, cy0 + 54];
      if (i === 0) {
        fig.glow(mx - 30, my + 10, 34, t.event);
        fig.circle(mx - 30, my + 10, 6.5, { fill: t.ink });
        fig.arrow([[mx - 22, my + 4], [mx + 20, my - 18]], { color: t.event, sw: 2, size: 8 });
        fig.circle(mx + 28, my - 22, 6, { fill: t.panel, stroke: t.ink, sw: 1.75 });
      }
      if (i === 1) {
        const path = [[mx - 70, my + 22], [mx - 40, my + 16], [mx - 10, my + 2], [mx + 18, my - 14], [mx + 44, my - 22]];
        fig.path(Figure.smooth(path), { stroke: t.refine, sw: 2.25, cap: 'round' });
        const [dx, dy] = [mx - 22, my + 9];
        fig.line(dx - 9, dy + 5, dx + 9, dy - 5, { stroke: t.ink, sw: 2 });
        fig.circle(dx - 9, dy + 5, 4.5, { fill: t.ink });
        fig.circle(dx + 9, dy - 5, 4.5, { fill: t.ink });
        fig.polygon([[mx + 54, my - 34], [mx + 63, my - 18], [mx + 45, my - 18]], { fill: t.refine });
      }
      if (i === 2) {
        const top = [mx, my - 22];
        fig.curve([top[0] - 8, top[1] + 4], [top[0] - 36, top[1] + 6], [mx - 54, my + 4], [mx - 58, my + 22], { color: t.event, sw: 1.75, dash: '2 5', size: 7 });
        fig.curve([top[0] + 8, top[1] + 4], [top[0] + 36, top[1] + 6], [mx + 54, my + 4], [mx + 58, my + 22], { color: t.event, sw: 1.75, dash: '2 5', size: 7 });
        fig.polygon([[top[0], top[1] - 9], [top[0] + 9, top[1] + 7], [top[0] - 9, top[1] + 7]], { fill: t.refine });
        fig.circle(mx - 58, my + 30, 6.5, { fill: t.ink });
        fig.circle(mx + 58, my + 30, 6.5, { fill: t.ink });
        fig.glyph('check', mx - 80, my + 30, 14, t.good);
      }
      if (i === 3) {
        // A well and a saddle, each with its vibrations.
        const curve = [[mx - 72, my - 12], [mx - 44, my + 26], [mx - 16, my + 4], [mx + 10, my - 22], [mx + 36, my + 2], [mx + 70, my + 30]];
        fig.path(Figure.smooth(curve), { stroke: t.ink3, sw: 2 });
        fig.circle(mx - 44, my + 26, 5.5, { fill: t.ink });
        fig.polygon([[mx + 10, my - 31], [mx + 18, my - 17], [mx + 2, my - 17]], { fill: t.refine });
        fig.path(Figure.polyline([[mx - 58, my + 40], [mx - 52, my + 34], [mx - 46, my + 40], [mx - 40, my + 34], [mx - 34, my + 40]]), { stroke: t.refine, sw: 1.75, join: 'round', cap: 'round' });
      }
      if (i === 4) {
        const widths = [92, 46, 18];
        let bx = mx - 78;
        widths.forEach((w, k) => {
          fig.rect(bx, my - 6, w - 2, 22, { rx: 4, fill: t.ramp[2 - k] });
          bx += w;
        });
        fig.head(mx - 78 + 118, my - 9, Math.PI / 2, { color: t.ink, size: 10 });
        fig.line(mx - 78 + 118, my - 28, mx - 78 + 118, my - 12, { stroke: t.ink, sw: 1.75 });
      }
      fig.badge(cx + 30, cy0 + 112, i + 1, { r: 13 });
      fig.text(cx + 52, cy0 + 112, card.title, { size: TYPE.body, weight: 700, middle: true });
      fig.text(cx + 18, cy0 + 148, card.note, { size: TYPE.small, fill: t.ink2, middle: true });
    });

    // Searches repeat until the state is confident; steps repeat for akmc.steps.
    const left = 36 + cw / 2;
    const inner = cy0 + ch + 24;
    const fourth = 36 + 3 * (cw + cg) + cw / 2;
    fig.arrow([[fourth, cy0 + ch + 2], [fourth, inner], [left + 16, inner], [left + 16, cy0 + ch + 3]], { color: t.event, sw: 1.75, size: 9, radius: 10 });
    fig.text((left + fourth) / 2, inner, 'until confident', { size: TYPE.small, fill: t.ink2, anchor: 'middle', middle: true, halo: true, haloWidth: 14 });
    const outer = cy0 + ch + 60;
    const fifth = 36 + 4 * (cw + cg) + cw / 2;
    fig.arrow([[fifth, cy0 + ch + 2], [fifth, outer], [left - 16, outer], [left - 16, cy0 + ch + 3]], { color: t.ink3, sw: 1.75, size: 9, radius: 10 });
    fig.text(600, outer, ['repeats for ', { t: 'steps', mono: true }], { size: TYPE.small, fill: t.ink2, anchor: 'middle', middle: true, halo: true, haloWidth: 14 });
  },
};
