// How each EON search ends, how the ending moves the count of repeats, and the confidence rule
// that decides when the current state has been searched enough.

import { scale, TYPE } from '../lib.mjs';

export default {
  name: 'eon-outcomes',
  title: 'Search outcomes',
  alt:
    'Each EON search passes four checks, the search itself, the thermal window, whether the saddle is already ' +
    'known, and whether the product is the starting state. They end the search as failed, outside_window, repeat, ' +
    'or same_state. A search that passes all four is a new process and resets the count of repeats, and a repeat ' +
    'adds one. A chart shows the confidence 1 - 1/N reaching 0.95 after 20 repeats in a row, when the kinetic ' +
    'Monte Carlo step is taken.',
  caption:
    'Each search either ends at a check with a recorded status or adds a process and its reverse. ' +
    '<code>repeat</code> counts toward the confidence only for a process inside the thermal window.',
  width: 1200,
  height: 382,
  draw(fig) {
    const t = fig.t;
    const [bx, bw, bh, pitch, y0] = [36, 290, 44, 66, 40];
    const rows = [
      { title: 'EON search', exit: 'failed', color: t.critical, glyph: 'cross' },
      { title: 'Thermal window', exit: 'outside_window', color: t.warn },
      { title: 'Known saddle', exit: 'repeat', color: t.ink2, count: 'N + 1' },
      { title: 'Same state', exit: 'same_state', color: t.warn },
    ];
    const ys = [0, 1, 2, 3, 4].map((i) => y0 + i * pitch);
    const exitX = bx + bw + 48;
    const countX = 604;
    const label = (x, y, text, color, glyph) => {
      if (glyph) fig.glyph(glyph, x + 7, y, 14, color);
      else fig.circle(x + 7, y, 6, { fill: color });
      fig.text(x + 22, y, text, { size: TYPE.small, mono: true, weight: 600, middle: true });
    };
    const count = (y, text) => fig.pill(countX, y, text, { size: TYPE.small, mono: true, weight: 700, fill: t.panel2 });

    rows.forEach((row, i) => {
      const y = ys[i];
      fig.card(bx, y - bh / 2, bw, bh, { fill: t.panel });
      fig.text(bx + 18, y, row.title, { size: TYPE.body, weight: 700, middle: true });
      fig.arrow([[bx + bw / 2, y + bh / 2 + 2], [bx + bw / 2, ys[i + 1] - bh / 2 - 2]], { color: t.ink2, size: 8 });
      fig.arrow([[bx + bw + 6, y], [exitX - 8, y]], { color: t.ink3, size: 8, sw: 1.5 });
      label(exitX, y, row.exit, row.color, row.glyph);
      if (row.count) count(y, row.count);
    });

    // A search that passes every check is a new process.
    const yn = ys[4];
    fig.card(bx, yn - bh / 2, bw, bh, { fill: fig.tint(t.good, 0.1), stroke: t.good });
    label(bx + 14, yn, 'new', t.good, 'check');
    count(yn, 'N = 0');

    // After every search: the confidence rule.
    const bus = 742;
    fig.line(bus, ys[0], bus, ys[4], { stroke: t.axis, sw: 1.5 });
    for (const y of ys) fig.line(bus - 14, y, bus, y, { stroke: t.axis, sw: 1.5 });
    const mid = (ys[0] + ys[4]) / 2;
    const card = { x: 786, w: 378, y: 16, h: 350 };
    fig.arrow([[bus, mid], [card.x - 4, mid]], { color: t.ink2, size: 9 });
    fig.card(card.x, card.y, card.w, card.h, { fill: t.panel });

    // Confidence 1 - 1/N after N repeats in a row; the default target is 0.95.
    const [left, right, top, bottom] = [card.x + 100, card.x + card.w - 28, card.y + 36, card.y + 196];
    const x = scale(0, 25, left, right);
    const y = scale(0, 1, bottom, top);
    for (const tick of [0, 0.5]) {
      fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
      fig.text(left - 12, y(tick), String(tick), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
    }
    fig.line(left, y(0.95), right, y(0.95), { stroke: t.md, sw: 1.5, dash: '5 4' });
    fig.text(left - 12, y(0.95), '0.95', { size: TYPE.small, weight: 700, anchor: 'end', middle: true, fill: t.md });
    fig.line(left, bottom, right, bottom, { stroke: t.axis });
    for (const tick of [0, 10, 20]) fig.text(x(tick), bottom + 26, String(tick), { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
    fig.text(right, bottom + 26, 'N', { size: TYPE.small, weight: 700, anchor: 'end', fill: t.ink2 });
    for (let n = 1; n <= 25; n++) {
      const reached = n >= 20;
      fig.circle(x(n), y(1 - 1 / n), reached ? 4.5 : 3.5, { fill: reached ? t.md : t.ink2 });
    }
    fig.text(card.x + 26, (top + bottom) / 2, '1 − 1/N', { size: TYPE.small, weight: 700, anchor: 'middle', rotate: -90, fill: t.ink2 });

    // What happens next.
    // Points at the target in blue, below it in gray.
    [
      ['KMC step', t.md],
      ['next search', t.ink2],
    ].forEach(([text, color], i) => {
      const ly = card.y + 262 + i * 40;
      fig.circle(card.x + 30, ly, 7, { fill: color });
      fig.text(card.x + 48, ly, text, { size: TYPE.body, weight: 600, middle: true });
    });
  },
};
