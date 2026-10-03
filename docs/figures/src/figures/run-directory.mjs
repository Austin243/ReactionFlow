// What one trajectory writes, and which file is the authority for what.

import { TYPE } from '../lib.mjs';

export default {
  name: 'run-directory',
  title: 'What a run writes',
  alt:
    'Directory tree of one trajectory: a contract file, state.json, the reactions.sqlite3 registry, one runtime ' +
    'checkpoint, numbered trajectory segments, one candidates folder per detected occurrence with reactant and product ' +
    'structures, and one pathways folder per refined occurrence with result.json and the band images.',
  caption:
    'Every trajectory owns one directory of versioned JSON, SQLite, and ASE trajectory files. Files are flushed to disk ' +
    'and then renamed into place, so an interrupted job never leaves a half-written record.',
  width: 1200,
  height: 518,
  draw(fig) {
    const t = fig.t;
    const entries = [
      { depth: 0, name: 'outputs/<campaign>/<trajectory-id>/', kind: 'dir' },
      { depth: 1, name: 'trajectory-contract.json', note: 'hashes of the inputs', color: t.ink3 },
      { depth: 1, name: 'state.json', note: 'phase and counters', color: t.ink3 },
      { depth: 1, name: 'reactions.sqlite3', note: 'occurrences and classes', color: t.event },
      { depth: 1, name: 'runtime-checkpoints/', note: 'the exact checkpoint', color: t.md, kind: 'dir' },
      { depth: 1, name: 'segments/0000/trajectory.traj', note: 'MD frames, one segment per pause', color: t.md },
      { depth: 1, name: 'candidates/<occurrence-id>/', note: 'one per detected occurrence', color: t.event, kind: 'dir' },
      { depth: 2, name: 'candidate.json', note: 'atoms, bonds, frames', color: t.event },
      { depth: 2, name: 'reactant.traj  product.traj', note: 'the two bracketing frames', color: t.event },
      { depth: 1, name: 'pathways/<occurrence-id>/', note: 'one per refinement attempt', color: t.refine, kind: 'dir' },
      { depth: 2, name: 'result.json', note: 'status, barrier, diagnostics', color: t.refine },
      { depth: 2, name: 'images.traj', note: 'the band', color: t.refine },
      { depth: 1, name: 'last-error.json', note: 'only after an error', color: t.critical },
    ];
    const [x0, top, pitch, indent] = [58, 46, 32, 30];
    const noteX = 500;
    fig.card(36, 16, 1128, entries.length * pitch + 22, { fill: t.panel });
    entries.forEach((entry, i) => {
      const y = top + i * pitch;
      const x = x0 + entry.depth * indent;
      if (entry.depth) {
        // Elbow from the parent's spine.
        const spine = x - indent + 8;
        const next = entries.slice(i + 1).findIndex((e) => e.depth <= entry.depth);
        const more = next !== -1 && entries[i + 1 + next].depth === entry.depth;
        fig.path(`M${spine} ${y - pitch / 2} V${y} H${x - 6}`, { stroke: t.axis, sw: 1.25 });
        if (more) fig.line(spine, y, spine, y + pitch * (next + 1) - pitch / 2, { stroke: t.axis, sw: 1.25 });
      }
      fig.text(x, y, entry.name, { size: TYPE.small, mono: true, weight: entry.kind === 'dir' ? 700 : 500, middle: true });
      if (entry.note) {
        fig.circle(noteX, y, 5.5, { fill: entry.color });
        fig.text(noteX + 18, y, entry.note, { size: TYPE.small, fill: t.ink2, middle: true });
      }
    });

    // Legend.
    const ly = 16 + entries.length * pitch + 22 + 34;
    let x = 40;
    for (const [label, color] of [['run state', t.ink3], ['molecular dynamics', t.md], ['detection', t.event], ['pathway refinement', t.refine]]) {
      fig.circle(x + 6, ly, 5.5, { fill: color });
      fig.text(x + 20, ly, label, { size: TYPE.small, fill: t.ink2, middle: true });
      x += 20 + fig.measure(label, { size: TYPE.small }) + 30;
    }
  },
};
