// Every way a refinement can end, where it ends, and what happens next.

import { TYPE } from '../lib.mjs';

export default {
  name: 'outcomes',
  title: 'Refinement outcomes',
  alt:
    'Refinement runs through five stages: prepare endpoints, relax endpoints, check bonds, nudged elastic band, and ' +
    'climbing image. Each stage can end the attempt with a named status: unresolved, relaxation_failed, collapsed, ' +
    'neb_failed, or ci_neb_failed. A converged pathway is recorded as ci_neb_converged with frequency and connectivity ' +
    'diagnostics. Every outcome is saved, the exact checkpoint is restored, and the dynamics continue.',
  caption:
    'Each stage either passes or ends the attempt with a recorded status, saved with whatever images exist. Outcomes are ' +
    'never overwritten. After an unsuccessful attempt, the next resolved occurrence of the same class starts a new one; a ' +
    '<code>ci_neb_converged</code> result ends the attempts for that class.',
  width: 1200,
  height: 462,
  draw(fig) {
    const t = fig.t;
    const [bx, bw, bh, pitch, y0] = [36, 290, 44, 60, 38];
    const rows = [
      { title: 'Prepare endpoints', exits: ['unresolved'] },
      { title: 'Relax endpoints', exits: ['relaxation_failed'] },
      { title: 'Check bonds', exits: ['collapsed', 'unresolved'] },
      { title: 'NEB or SSNEB', exits: ['neb_failed'] },
      { title: 'Climbing image', exits: ['ci_neb_failed'] },
    ];
    const ys = [0, 1, 2, 3, 4, 5, 6.25].map((i) => y0 + i * pitch);
    const label = (x, y, text, color, glyph) => {
      if (glyph) fig.glyph(glyph, x + 7, y, 14, color);
      else fig.circle(x + 7, y, 6, { fill: color });
      fig.text(x + 22, y, text, { size: TYPE.small, mono: true, weight: 600, middle: true });
      return 22 + fig.measure(text, { size: TYPE.small, mono: true, weight: 600 });
    };
    const exitX = bx + bw + 48;

    rows.forEach((row, i) => {
      const y = ys[i];
      fig.card(bx, y - bh / 2, bw, bh, { fill: t.panel });
      fig.text(bx + 18, y, row.title, { size: TYPE.body, weight: 700, middle: true });
      fig.arrow([[bx + bw / 2, y + bh / 2 + 2], [bx + bw / 2, ys[i + 1] - bh / 2 - 2]], { color: t.ink2, size: 8 });
      fig.arrow([[bx + bw + 6, y], [exitX - 8, y]], { color: t.ink3, size: 8, sw: 1.5 });
      let x = exitX;
      for (const status of row.exits) x += label(x, y, status, t.warn) + 30;
    });

    // Converged, with the two diagnostics that never change the status.
    const yc = ys[5];
    fig.card(bx, yc - bh / 2, bw, bh, { fill: fig.tint(t.good, 0.1), stroke: t.good });
    label(bx + 14, yc, 'ci_neb_converged', t.good, 'check');
    fig.text(exitX, yc, 'diagnostics:', { size: TYPE.small, fill: t.ink3, middle: true });
    let dx = exitX + fig.measure('diagnostics:', { size: TYPE.small }) + 16;
    for (const name of ['frequencies', 'connectivity']) {
      fig.circle(dx + 6, yc, 5.5, { fill: t.refine });
      fig.text(dx + 20, yc, name, { size: TYPE.small, middle: true });
      dx += 20 + fig.measure(name, { size: TYPE.small }) + 26;
    }

    // An unexpected error in any stage is an outcome too.
    const ye = ys[6];
    fig.card(bx, ye - bh / 2, bw, bh, { fill: t.bg, stroke: t.axis, dash: '5 4' });
    fig.text(bx + 18, ye, 'Unexpected error', { size: TYPE.body, weight: 700, middle: true, fill: t.ink2 });
    fig.arrow([[bx + bw + 6, ye], [exitX - 8, ye]], { color: t.ink3, size: 8, sw: 1.5 });
    label(exitX, ye, 'failed', t.critical, 'cross');

    // Every outcome is saved, the checkpoint is restored, and the dynamics continue.
    const bus = 846;
    fig.line(bus, ys[0], bus, ys[6], { stroke: t.axis, sw: 1.5 });
    for (const y of ys) fig.line(bus - 14, y, bus, y, { stroke: t.axis, sw: 1.5 });
    const mid = (ys[0] + ys[6]) / 2;
    const card = { x: 892, w: 272, y: mid - 120, h: 240 };
    fig.arrow([[bus, mid], [card.x - 4, mid]], { color: t.ink2, size: 9 });
    fig.card(card.x, card.y, card.w, card.h, { fill: t.panel });
    const steps = [
      [['save ', { t: 'result.json', mono: true }], null],
      ['restore the checkpoint', null],
      ['MD continues', t.md],
    ];
    steps.forEach(([text, color], i) => {
      const y = card.y + 44 + i * 76;
      let x = card.x + 22;
      if (color) {
        fig.circle(x + 7, y, 7, { fill: color });
        x += 22;
      }
      fig.text(x, y, text, { size: TYPE.body, weight: 600, middle: true });
      if (i < 2) fig.arrow([[card.x + 34, y + 18], [card.x + 34, y + 58]], { color: t.ink2, size: 8 });
    });
  },
};
