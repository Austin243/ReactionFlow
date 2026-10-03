// Reaction identity: a change graph, what it ignores, and how occurrences group into classes.

import { mix, TYPE } from '../lib.mjs';

/**
 * Change graph of acetonitrile <-> ketenimine (a hydrogen moves from the methyl carbon to nitrogen).
 * variant: 'forward' | 'reverse' | 'loss' (only the C-H bond breaks); bend and flip redraw the
 * same graph with a different geometry.
 */
function changeGraph(fig, x, y, s, { variant = 'forward', bend = 0, flip = false, ids = null } = {}) {
  const t = fig.t;
  const arm = (length, degrees) => [length * Math.cos((degrees * Math.PI) / 180), length * Math.sin((degrees * Math.PI) / 180)];
  const c2 = [58, 0];
  const n = [c2[0] + arm(58, bend)[0], c2[1] + arm(58, bend)[1]];
  const moving = [n[0] / 2, flip ? 52 : -54];
  const nodes = [
    ['C', [0, 0]], ['C', c2], ['N', n], ['H', [-36, -36]], ['H', [-36, 36]], ['H', moving],
  ].map(([el, [px, py]]) => [el, [x + px * s, y + py * s]]);
  const edges = [
    [0, 1, 'same'], [1, 2, 'same'], [0, 3, 'same'], [0, 4, 'same'],
    [0, 5, variant === 'reverse' ? 'formed' : 'broken'],
    ...(variant === 'loss' ? [] : [[5, 2, variant === 'reverse' ? 'broken' : 'formed']]),
  ];
  for (const [a, b, kind] of edges) {
    const [p, q] = [nodes[a][1], nodes[b][1]];
    const style = {
      same: { stroke: t.ink3, sw: 2 * s },
      formed: { stroke: t.refine, sw: 4.5 * s },
      broken: { stroke: t.event, sw: 4.5 * s, dash: `${6 * s} ${5 * s}` },
    }[kind];
    fig.line(...p, ...q, { ...style, cap: kind === 'broken' ? 'butt' : 'round' });
  }
  nodes.forEach(([el, [px, py]], i) => {
    const r = (el === 'H' ? 11 : 14) * s;
    fig.circle(px, py, r, { fill: t.bg, stroke: t.ink2, sw: 1.5 });
    fig.text(px, py, el, { size: (el === 'H' ? 14 : 17) * s, weight: 700, anchor: 'middle', middle: true });
    if (ids) {
      const [below, side] = [r + 14, r + 6];
      const [dx, dy, anchor] = [[0, below, 'middle'], [0, below, 'middle'], [0, below, 'middle'], [-side, 0, 'end'], [-side, 0, 'end'], [side, -4, 'start']][i];
      fig.text(px + dx, py + dy, String(ids[i]), { size: 17, mono: true, fill: t.ink3, anchor, middle: true });
    }
  });
}

export default {
  name: 'identity',
  title: 'Reaction classes',
  alt:
    'A reaction occurrence is described by a change graph: atoms are nodes labeled by element, and bonds are edges ' +
    'labeled unchanged, formed, or broken. Occurrences with different atom numbers, a different geometry, or the ' +
    'reverse direction have the same graph and fall into one class; a different pattern of changes is a new class. ' +
    'Every occurrence is stored, and a class is refined at each new occurrence until one attempt converges.',
  caption:
    'Identity is exact graph isomorphism over the connected region touched by the change. Element labels and the ' +
    'pattern of formed and broken bonds must match; atom numbering, geometry, and direction do not matter. Every ' +
    'occurrence is kept, and a class is refined again at later occurrences only until one attempt converges. ' +
    'The example is the hydrogen shift between acetonitrile and ketenimine.',
  width: 1200,
  height: 404,
  draw(fig) {
    const t = fig.t;
    const [top, h] = [60, 320];
    const heading = (x, n, label) => {
      fig.badge(x + 13, 30, n, { r: 13 });
      fig.text(x + 36, 30, label, { size: TYPE.body, weight: 700, middle: true });
    };

    // 1. One occurrence as a change graph.
    heading(36, 1, 'Change graph');
    fig.card(36, top, 280, h, { fill: t.panel });
    changeGraph(fig, 124, top + 128, 1.12, { ids: [17, 18, 19, 40, 41, 42] });
    [
      [{ stroke: t.ink3, sw: 2.5 }, 'unchanged'],
      [{ stroke: t.refine, sw: 5 }, 'formed'],
      [{ stroke: t.event, sw: 5, dash: '7 5' }, 'broken'],
    ].forEach(([style, label], i) => {
      const ly = top + 232 + i * 30;
      fig.line(58, ly, 94, ly, { ...style, cap: style.dash ? 'butt' : 'round' });
      fig.text(108, ly, label, { size: TYPE.small, fill: t.ink2, middle: true });
    });

    // 2. Comparisons.
    heading(340, 2, 'Same reaction?');
    const cells = [
      { label: 'renumbered', same: true, x: 86, options: { ids: [23, 9, 77, 31, 150, 64] } },
      { label: 'reversed', same: true, options: { variant: 'reverse' } },
      { label: 'new geometry', same: true, options: { bend: -30, flip: true } },
      { label: 'different change', same: false, options: { variant: 'loss' } },
    ];
    const [cw, ch] = [206, (h - 12) / 2];
    cells.forEach((cell, i) => {
      const [cx, cy] = [340 + (i % 2) * (cw + 12), top + Math.floor(i / 2) * (ch + 12)];
      const color = cell.same ? t.good : t.critical;
      fig.card(cx, cy, cw, ch, { fill: t.bg });
      changeGraph(fig, cx + (cell.x ?? 70), cy + 62, 0.82, cell.options);
      fig.text(cx + 14, cy + ch - 16, cell.label, { size: TYPE.small, fill: t.ink2 });
      fig.circle(cx + cw - 26, cy + ch - 23, 14, { fill: fig.tint(color, 0.16), stroke: color, sw: 1.25 });
      fig.text(cx + cw - 26, cy + ch - 23, cell.same ? '=' : '≠', { size: TYPE.body, weight: 700, anchor: 'middle', middle: true });
    });

    // 3. Classes: every occurrence is kept; a class is refined until one attempt converges.
    heading(788, 3, 'Reaction classes');
    const card = { x: 788, y: top, w: 376, h };
    fig.card(card.x, card.y, card.w, card.h, { fill: t.panel });
    fig.text(card.x + 18, card.y + 34, 'reactions.sqlite3', { size: TYPE.small, mono: true, weight: 700 });
    const dot = (x, y, kind) => {
      if (kind === 'converged') fig.circle(x, y, 7.5, { fill: t.refine });
      else if (kind === 'attempt') fig.circle(x, y, 7, { fill: t.warn, stroke: mix(t.warn, t.ink, 0.3), sw: 1 });
      else fig.circle(x, y, 6.5, { fill: t.bg, stroke: t.ink3, sw: 1.75 });
    };
    [
      ['broken C-H; formed H-N', ['converged', 'recorded', 'recorded']],
      ['broken C-H', ['attempt', 'converged']],
    ].forEach(([label, kinds], i) => {
      const ry = card.y + 54 + i * 82;
      fig.rect(card.x + 14, ry, card.w - 28, 72, { rx: 8, fill: t.bg, stroke: t.border });
      fig.text(card.x + 30, ry + 24, label, { size: TYPE.small, mono: true, weight: 600, middle: true });
      kinds.forEach((kind, k) => dot(card.x + 38 + k * 24, ry + 52, kind));
    });
    [['converged', 'converged'], ['attempt', 'not converged'], ['recorded', 'recorded only']].forEach(([kind, label], i) => {
      const ly = card.y + 242 + i * 28;
      dot(card.x + 38, ly, kind);
      fig.text(card.x + 56, ly, label, { size: TYPE.small, fill: t.ink2, middle: true });
    });
  },
};
