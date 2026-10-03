// A campaign as swimlanes: independent trajectories, one per GPU, across a resubmitted job.

import { TYPE } from '../lib.mjs';

export default {
  name: 'campaign',
  title: 'Campaigns',
  alt:
    'Eight trajectories run as eight independent tasks on two four-GPU nodes. Each lane shows molecular dynamics with ' +
    'occasional pauses for pathway refinement in that lane only. One trajectory stops with an error while the others ' +
    'continue. After the job ends, resubmitting the same command resumes the incomplete trajectories from their exact ' +
    'checkpoints and leaves the completed one unchanged.',
  caption:
    'A campaign is one JSON file: a starting structure and any number of independently parameterized trajectories. ' +
    'Slurm starts one task per trajectory, each with its own GPU for GPU models; there is no central scheduler or monitor service. The timings are schematic.',
  width: 1200,
  height: 452,
  draw(fig) {
    const t = fig.t;

    // Lanes. Segment ends are in units of the two job spans (job 1 is 366 long, job 2 is 352).
    const [L, R] = [214, 1150];
    const k = (R - 10 - L - 44) / (366 + 352);
    const [end1, start2] = [L + 366 * k, L + 366 * k + 44];
    const top = 84;
    const [bar, pitch] = [24, 38];
    const y = (i) => top + i * pitch + (i > 3 ? 16 : 0);
    const lanes = [
      { model: 'A', job1: [['md', 0, 116], ['ref', 116, 162], ['md', 162, 366]], job2: [['md', 0, 204]], done2: 204 },
      { model: 'A', job1: [['md', 0, 366]], job2: [['md', 0, 94], ['ref', 94, 146], ['md', 146, 352]], more: true },
      { model: 'B', job1: [['md', 0, 66], ['ref', 66, 104], ['md', 104, 246], ['ref', 246, 296], ['md', 296, 366]], job2: [['md', 0, 352]], more: true },
      { model: 'B', job1: [['md', 0, 296]], done1: 296, job2: [] },
      { model: 'A', job1: [['md', 0, 206]], error: 206, job2: [['md', 0, 352]], more: true },
      { model: 'A', job1: [['md', 0, 176], ['ref', 176, 236], ['md', 236, 366]], job2: [['md', 0, 174], ['ref', 174, 224], ['md', 224, 352]], more: true },
      { model: 'B', job1: [['md', 0, 366]], job2: [['md', 0, 352]], more: true },
      { model: 'B', job1: [['md', 0, 286], ['ref', 286, 340], ['md', 340, 366]], job2: [['md', 0, 284]], done2: 284 },
    ];

    // Job spans and the gap between them.
    fig.rect(end1 + 4, top - 22, start2 - end1 - 8, y(7) + bar / 2 + 22 - (top - 22), { fill: t.panel2, rx: 4 });
    for (const [a, b, label] of [[L, end1, 'job 1'], [start2, R, 'job 2, resubmitted']]) {
      fig.line(a, top - 38, b, top - 38, { stroke: t.ink3, sw: 1.25 });
      for (const edge of [a, b]) fig.line(edge, top - 43, edge, top - 33, { stroke: t.ink3, sw: 1.25 });
      fig.text((a + b) / 2, top - 52, label, { size: TYPE.body, weight: 600, anchor: 'middle' });
    }
    fig.text((end1 + start2) / 2, (top + y(7)) / 2, 'job ends', { size: TYPE.small, fill: t.ink2, anchor: 'middle', middle: true, rotate: -90 });

    // Node brackets and lane labels.
    [0, 4].forEach((firstLane, node) => {
      const [a, b] = [y(firstLane) - bar / 2 - 2, y(firstLane + 3) + bar / 2 + 2];
      fig.path(`M${L - 140} ${a} h-8 V${b} h8`, { stroke: t.ink3, sw: 1.5 });
      fig.text(L - 162, (a + b) / 2, `node ${node + 1}`, { size: TYPE.small, weight: 600, anchor: 'middle', middle: true, rotate: -90 });
    });
    const model = (cx, cy, letter) => {
      fig.circle(cx, cy, 12.5, { fill: t.bg, stroke: t.axis, sw: 1.25 });
      fig.text(cx, cy, letter, { size: 16, weight: 700, anchor: 'middle', middle: true, fill: t.ink2 });
    };
    const mark = (kind, cx, cy) => {
      const color = kind === 'check' ? t.good : t.critical;
      fig.circle(cx, cy, 12, { fill: fig.tint(color, 0.16), stroke: color, sw: 1.5 });
      fig.glyph(kind, cx, cy, kind === 'check' ? 11 : 9, color);
    };
    lanes.forEach((lane, i) => {
      const cy = y(i);
      fig.text(L - 126, cy, `GPU ${i % 4}`, { size: TYPE.small, fill: t.ink2, middle: true });
      model(L - 28, cy, lane.model);
      const segment = (origin, [kind, a, b]) =>
        fig.rect(origin + a * k + (a ? 1 : 0), cy - bar / 2, (b - a) * k - (a ? 1 : 0), bar, { rx: 5, fill: kind === 'md' ? t.md : t.refine });
      lane.job1.forEach((s) => segment(L, s));
      lane.job2.forEach((s) => segment(start2, s));
      if (lane.more) fig.polygon([[R - 10, cy - bar / 2], [R + 2, cy], [R - 10, cy + bar / 2]], { fill: t.md });
      for (const [origin, at] of [[L, lane.done1], [start2, lane.done2]]) {
        if (at !== undefined) mark('check', origin + at * k + 20, cy);
      }
      if (lane.error !== undefined) mark('cross', L + lane.error * k + 20, cy);
      if (lane.done1 !== undefined) {
        fig.rect(start2, cy - bar / 2, 130, bar, { rx: 5, stroke: t.axis, dash: '4 4' });
        fig.text(start2 + 144, cy, 'already complete', { size: TYPE.small, fill: t.ink2, middle: true });
      }
    });

    // Legend.
    const ly = y(7) + 56;
    const items = [
      [(x) => fig.rect(x, ly - 9, 28, 18, { rx: 4, fill: t.md }), 28, 'MD'],
      [(x) => fig.rect(x, ly - 9, 28, 18, { rx: 4, fill: t.refine }), 28, 'refinement'],
      [(x) => mark('check', x + 12, ly), 24, 'complete'],
      [(x) => mark('cross', x + 12, ly), 24, 'error'],
      [(x) => { model(x + 12, ly, 'A'); model(x + 42, ly, 'B'); }, 54, 'model'],
    ];
    let x = 36;
    for (const [draw, w, label] of items) {
      draw(x);
      fig.text(x + w + 10, ly, label, { size: TYPE.small, fill: t.ink2, middle: true });
      x += w + 10 + fig.measure(label, { size: TYPE.small }) + 36;
    }
  },
};
