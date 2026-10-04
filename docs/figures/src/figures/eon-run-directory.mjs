// What an EON campaign writes: one progress file and immutable records per state, search,
// process, and step.

import { TYPE } from '../lib.mjs';

export default {
  name: 'eon-run-directory',
  title: 'What an EON run writes',
  alt:
    'Directory tree of one EON campaign with search-contract.json, search-state.json, and eon.log, then one ' +
    'record per state, per search attempt, per process with its reverse, and per kinetic Monte Carlo step.',
  caption:
    'Every record is an immutable JSON file with a SHA-256 checksum, written and then renamed into place. ' +
    'Progress lives in <code>search-state.json</code>, and rerunning the same command resumes from it.',
  width: 1200,
  height: 422,
  draw(fig) {
    const t = fig.t;
    const entries = [
      { depth: 0, name: 'outputs/<campaign>/', kind: 'dir' },
      { depth: 1, name: 'search-contract.json', note: 'structure, model, settings, code', color: t.ink3 },
      { depth: 1, name: 'search-state.json', note: 'current state and clock', color: t.ink3 },
      { depth: 1, name: 'eon.log', note: "EON's own log", color: t.ink3 },
      { depth: 1, name: 'states/state-000000.json', note: 'minimum and energy', color: t.ink },
      { depth: 1, name: 'attempts/attempt-000000.json', note: 'one search and its outcome', color: t.event },
      { depth: 1, name: 'processes/', note: 'one per new saddle', color: t.refine, kind: 'dir' },
      { depth: 2, name: 'process-000000.json', note: 'saddle, product, barrier, prefactors', color: t.refine },
      { depth: 2, name: 'process-000000r.json', note: 'the same process in reverse', color: t.refine },
      { depth: 1, name: 'steps/step-000000.json', note: 'from, to, time step', color: t.md },
    ];
    const [x0, top, pitch, indent] = [58, 46, 32, 30];
    const noteX = 486;
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
    for (const [label, color] of [['run', t.ink3], ['states', t.ink], ['searches', t.event], ['processes', t.refine], ['kinetic Monte Carlo', t.md]]) {
      fig.circle(x + 6, ly, 5.5, { fill: color });
      fig.text(x + 20, ly, label, { size: TYPE.small, fill: t.ink2, middle: true });
      x += 20 + fig.measure(label, { size: TYPE.small }) + 30;
    }
  },
};
