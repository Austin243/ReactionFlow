// The loop on a timeline: MD runs, the monitor confirms a bond change, MD pauses at an exact
// checkpoint while the pathway is refined, and the same trajectory continues.

import { acetonitrile, disk } from '../chem.mjs';
import { Figure, TYPE } from '../lib.mjs';

export default {
  name: 'how-it-works',
  title: 'How ReactionFlow works',
  alt:
    'Timeline of one trajectory. Molecular dynamics runs while a bond monitor checks every observation interval. ' +
    'When a bond change persists, the dynamics pause at an exact checkpoint, the pathway is refined by endpoint ' +
    'relaxation, nudged elastic band, climbing image, and saddle checks, and then the checkpoint is restored and ' +
    'the same trajectory continues. Five cards summarize the steps: run MD, watch bonds, checkpoint, refine, resume.',
  caption:
    'The whole loop for one trajectory. Each trajectory is one process; the monitor runs on CPU cores of the same node, ' +
    'and pathway refinement uses the same model while the dynamics are paused.',
  width: 1200,
  height: 470,
  draw(fig) {
    const t = fig.t;

    // --- Timeline -------------------------------------------------------------------------
    const [x0, x1] = [228, 1164];
    const lanes = [
      { y: 74, label: 'MD', color: t.md },
      { y: 124, label: 'Bond monitor', color: t.event },
      { y: 174, label: 'Refinement', color: t.refine },
    ];
    const step = 24;
    const [pause, resume] = [x0 + 16 * step, x0 + 31 * step];
    const first = pause - 4 * step;

    for (const lane of lanes) {
      fig.circle(44, lane.y, 6, { fill: lane.color });
      fig.text(58, lane.y, lane.label, { size: TYPE.body, weight: 600, middle: true });
      fig.line(x0, lane.y, x1, lane.y, { stroke: t.grid, sw: 1 });
    }
    for (const [gx, label] of [[pause, 'checkpoint'], [resume, 'restore']]) {
      fig.line(gx, 46, gx, lanes[2].y + 24, { stroke: t.ink3, sw: 1.25, dash: '4 4' });
      fig.text(gx, 34, label, { size: TYPE.small, weight: 600, anchor: 'middle' });
    }

    // Dynamics: running, paused, running.
    const bar = 28;
    const md = lanes[0].y;
    fig.rect(x0, md - bar / 2, pause - x0, bar, { rx: 7, fill: t.md });
    fig.rect(pause + 4, md - bar / 2, resume - pause - 8, bar, { rx: 7, fill: t.panel, stroke: t.axis, dash: '5 4' });
    fig.text((pause + resume) / 2, md, 'paused', { size: TYPE.small, anchor: 'middle', middle: true, fill: t.ink2 });
    fig.path(
      `M${resume + 7} ${md - bar / 2} H${x1 - 14} L${x1} ${md} L${x1 - 14} ${md + bar / 2} H${resume + 7} a7 7 0 0 1 -7 -7 v${-(bar - 14)} a7 7 0 0 1 7 -7 z`,
      { fill: t.md },
    );

    // Monitor: one check per observation; the last few before the pause follow the bond change.
    const mon = lanes[1].y;
    for (let gx = x0 + step; gx < x1 - 6; gx += step) {
      if (gx > pause && gx <= resume) continue;
      if (gx >= first && gx <= pause) continue;
      fig.line(gx, mon - 7, gx, mon + 7, { stroke: t.ink3, sw: 1.75, cap: 'round' });
    }
    for (let gx = first; gx < pause; gx += step) fig.circle(gx, mon, 6, { fill: t.event });
    fig.polygon([[pause, mon - 11], [pause + 11, mon], [pause, mon + 11], [pause - 11, mon]], { fill: t.event, stroke: t.bg, sw: 2, join: 'round' });

    // Refinement stages inside the pause, each as wide as its label needs.
    const ref = lanes[2].y;
    const stages = ['relax', 'NEB', 'CI-NEB', 'checks'];
    const span = resume - pause - 8 - 3 * 4;
    const natural = stages.map((s) => fig.measure(s, { size: TYPE.small, weight: 500 }));
    const pad = (span - natural.reduce((a, b) => a + b, 0)) / stages.length;
    let sx = pause + 4;
    stages.forEach((label, i) => {
      const w = natural[i] + pad;
      fig.rect(sx, ref - bar / 2, w, bar, { rx: 7, fill: fig.tint(t.refine, 0.2), stroke: t.refine, sw: 1.25 });
      fig.text(sx + w / 2, ref, label, { size: TYPE.small, anchor: 'middle', middle: true, weight: 500 });
      sx += w + 4;
    });

    // --- Step cards -----------------------------------------------------------------------
    const cards = [
      { title: 'Run MD', color: t.md, note: 'NVT or NPT' },
      { title: 'Watch bonds', color: t.event, note: 'distance thresholds' },
      { title: 'Checkpoint', color: t.ink3, note: 'the exact state' },
      { title: 'Refine', color: t.refine, note: 'NEB and CI-NEB' },
      { title: 'Resume', color: t.md, note: 'from the same step' },
    ];
    const [cw, ch, gap, cy] = [216, 196, 12, 226];
    cards.forEach((card, i) => {
      const cx = 36 + i * (cw + gap);
      fig.card(cx, cy, cw, ch);
      fig.path(`M${cx + 10} ${cy + 0.5} h${cw - 20}`, { stroke: card.color, sw: 3, cap: 'round' });
      const [mx, my] = [cx + cw / 2, cy + 60];
      if (i === 0) {
        const a = acetonitrile(fig, mx - 40, my - 10, 22, 13);
        const b = acetonitrile(fig, mx + 44, my + 12, 198, 13);
        fig.arrow([[a.nitrogen[0] + 9, a.nitrogen[1] + 6], [a.nitrogen[0] + 25, a.nitrogen[1] + 16]], { color: t.md, sw: 1.75, size: 6 });
        fig.arrow([[b.nitrogen[0] - 9, b.nitrogen[1] - 6], [b.nitrogen[0] - 25, b.nitrogen[1] - 16]], { color: t.md, sw: 1.75, size: 6 });
      }
      if (i === 1) {
        const pair = (px, bonded) => {
          if (bonded) fig.glow(px, my, 30, t.event);
          fig.bond(px - 15, my, px + 15, my, bonded ? { w: 4.5 } : { kind: 'forming', w: 3.2 });
          fig.atom(px - (bonded ? 13 : 19), my, 9.5, 'C');
          fig.atom(px + (bonded ? 13 : 19), my, 9, 'N');
        };
        pair(mx - 54, false);
        fig.arrow([[mx - 14, my], [mx + 12, my]], { color: t.ink2, size: 7 });
        pair(mx + 52, true);
      }
      if (i === 2) disk(fig, mx - 22, my - 22, 44, t.ink2);
      if (i === 3) {
        const points = [-66, -44, -22, 0, 22, 44, 66].map((dx, k) => [mx + dx, my + 22 - 44 * Math.exp(-((k - 3) ** 2) / 3.2) - (k > 3 ? (k - 3) * -4 : 0)]);
        fig.path(Figure.smooth(points), { stroke: t.refine, sw: 2 });
        points.forEach(([px, py], k) => {
          fig.circle(px, py, 6.5, { fill: t.panel });
          if (k === 3) fig.polygon([[px, py - 8], [px + 7, py + 5], [px - 7, py + 5]], { fill: t.refine });
          else fig.circle(px, py, 4.5, { fill: k % 6 ? t.refine : t.ink });
        });
      }
      if (i === 4) {
        fig.circle(mx - 58, my, 17, { fill: t.md });
        fig.polygon([[mx - 63, my - 8], [mx - 49, my], [mx - 63, my + 8]], { fill: '#ffffff' });
        const wave = Array.from({ length: 25 }, (_, k) => [mx - 30 + k * 4.2, my + 9 * Math.sin(k * 0.8) * Math.exp(-k / 40)]);
        fig.path(Figure.smooth(wave), { stroke: t.md, sw: 2, cap: 'round' });
        fig.head(mx + 82, my + 1, 0, { color: t.md, size: 9 });
      }
      fig.badge(cx + 30, cy + 128, i + 1, { r: 13 });
      fig.text(cx + 52, cy + 128, card.title, { size: TYPE.body, weight: 700, middle: true });
      fig.text(cx + 18, cy + 166, card.note, { size: TYPE.small, fill: t.ink2, middle: true });
    });

    // The loop repeats.
    const ly = cy + ch + 28;
    fig.arrow([[36 + 4 * (cw + gap) + cw / 2, cy + ch + 2], [36 + 4 * (cw + gap) + cw / 2, ly], [36 + cw / 2, ly], [36 + cw / 2, cy + ch + 3]], {
      color: t.ink3,
      sw: 1.75,
      size: 9,
      radius: 10,
    });
    fig.text(600, ly, ['repeats until ', { t: 'total_steps', mono: true }], { size: TYPE.small, fill: t.ink2, anchor: 'middle', middle: true, halo: true, haloWidth: 14 });
  },
};
