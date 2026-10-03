// The model boundary: the built-in adapters from `reactionflow models` (data/models.json), the two
// ways to bring another model, the interface they all satisfy, and the commands that prepare one.

import { readFileSync } from 'node:fs';

import { TYPE } from '../lib.mjs';

const catalog = JSON.parse(readFileSync(new URL('../../data/models.json', import.meta.url)));
const listed = new Set(catalog.backends.map((backend) => backend.backend));

// One tile per adapter; `key` ties a tile to the catalog.
const TILES = [
  { key: 'mace', name: 'MACE' },
  { key: 'mace_field', name: 'MACE-FIELD' },
  { key: 'uma', name: 'UMA' },
  { key: 'aimnet2', name: 'AIMNet2' },
  { key: 'orb', name: 'ORB' },
  { key: 'mattersim', name: 'MatterSim' },
  { key: 'chgnet', name: 'CHGNet' },
  { key: 'sevennet', name: 'SevenNet' },
  { key: 'nep', name: 'NEP89' },
  { key: 'ani1xnr', name: 'ANI-1xnr' },
  { name: ['Any ASE', 'calculator'], open: true },
  { name: ['Custom', 'adapter'], open: true },
];
const missing = [...listed].filter((key) => !TILES.some((tile) => tile.key === key));
if (missing.length) throw new Error(`models figure has no tile for: ${missing.join(', ')}`);

export default {
  name: 'models',
  title: 'Models',
  alt:
    'Ten built-in adapters: MACE, MACE-FIELD, UMA, AIMNet2, ORB, MatterSim, CHGNet, SevenNet, NEP89, and ANI-1xnr, ' +
    'plus a generic adapter for any ASE calculator and a custom adapter interface. Every adapter provides three ' +
    'methods: start, restore, and calculator. Three commands list the models, prepare the selected one, and run.',
  caption:
    'Model selection lives in the campaign file. Every adapter, built in or not, gives ReactionFlow the same three ' +
    'things: a new MD runtime, that runtime rebuilt from an exact checkpoint, and a calculator for each refinement stage.',
  width: 1200,
  height: 414,
  draw(fig) {
    const t = fig.t;

    // Adapter tiles.
    const [tw, th, gap] = [178, 76, 12];
    TILES.forEach((tile, i) => {
      const [x, y] = [36 + (i % 6) * (tw + gap), 16 + Math.floor(i / 6) * (th + gap)];
      fig.card(x, y, tw, th, tile.open ? { fill: t.bg, stroke: t.ink3, dash: '5 4' } : { fill: t.panel });
      if (tile.open) {
        fig.lines(x + 16, y + 31, tile.name, { size: TYPE.body, weight: 700, lh: 25 });
        return;
      }
      const count = catalog.backends.find((backend) => backend.backend === tile.key).models.length;
      fig.text(x + 16, y + 30, tile.name, { size: TYPE.body, weight: 700, middle: true });
      fig.text(x + 16, y + 56, `${count} ${count > 1 ? 'models' : 'model'}`, { size: TYPE.small, fill: t.ink2, middle: true });
    });

    // The interface every adapter satisfies.
    const by = 16 + 2 * th + gap + 32;
    fig.arrow([[600, by - 26], [600, by - 3]], { color: t.ink2, size: 9 });
    fig.card(36, by, 1128, 84, { fill: t.bg, stroke: t.ink2, sw: 1.5 });
    fig.lines(56, by + 34, ['Every adapter', 'provides'], { size: TYPE.body, weight: 700, lh: 26 });
    [
      ['start(atoms)', 'a new MD run', t.md],
      ['restore(snapshot)', 'that run, from a checkpoint', t.md],
      ['calculator(stage)', 'one refinement stage', t.refine],
    ].forEach(([name, note, color], i) => {
      const x = 222 + i * 316;
      fig.rect(x, by + 12, 302, 60, { rx: 8, fill: fig.tint(color, 0.1), stroke: fig.tint(color, 0.55) });
      fig.circle(x + 18, by + 31, 5.5, { fill: color });
      fig.text(x + 32, by + 31, name, { size: TYPE.small, mono: true, weight: 700, middle: true });
      fig.text(x + 32, by + 56, note, { size: TYPE.small, fill: t.ink2, middle: true });
    });

    // Commands.
    const cy = by + 84 + 20;
    const cw = (1128 - 2 * 36) / 3;
    [
      ['reactionflow models', 'lists every model'],
      ['reactionflow prepare', 'downloads the weights'],
      ['reactionflow run', 'runs one trajectory'],
    ].forEach(([command, note], i) => {
      const x = 36 + i * (cw + 36);
      fig.card(x, cy, cw, 76);
      fig.badge(x + 28, cy + 27, i + 1, { r: 13 });
      fig.text(x + 52, cy + 27, command, { size: TYPE.small, mono: true, weight: 700, middle: true });
      fig.text(x + 52, cy + 55, note, { size: TYPE.small, fill: t.ink2, middle: true });
      if (i) fig.arrow([[x - 32, cy + 38], [x - 4, cy + 38]], { color: t.ink2, size: 9 });
    });
  },
};
