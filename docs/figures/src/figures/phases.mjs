// The phases a trajectory moves through, as recorded in state.json and shown by `reactionflow status`.

import { TYPE } from '../lib.mjs';

export default {
  name: 'phases',
  title: 'Run phases',
  alt:
    'A trajectory starts as new, then alternates between running and refining: when a pathway is queued the run ' +
    'checkpoints and refines, and when every queued pathway has an outcome it restores the checkpoint and runs again. ' +
    'It ends as completed when it reaches its step target, or as failed if its run state cannot be saved.',
  caption:
    'The phase is stored in <code>state.json</code> and listed by <code>reactionflow status</code>, so a resubmitted ' +
    'job continues each trajectory from the right place.',
  width: 1200,
  height: 260,
  draw(fig) {
    const t = fig.t;
    const [cy, ty, h] = [62, 196, 56];
    const size = TYPE.body;
    const node = (id, cx, y, w, color, glyph) => {
      fig.rect(cx - w / 2, y - h / 2, w, h, { rx: 10, fill: fig.tint(color, 0.14), stroke: color, sw: 1.75 });
      fig.text(cx + (glyph ? 12 : 0), y, id, { size, mono: true, weight: 700, anchor: 'middle', middle: true });
      if (glyph) fig.glyph(glyph, cx - fig.measure(id, { size, mono: true, weight: 700 }) / 2 - 6, y, 13, color);
      return { cx, y, left: cx - w / 2, right: cx + w / 2 };
    };
    const created = node('new', 130, cy, 120, t.ink3);
    const running = node('running', 450, cy, 170, t.md);
    const refining = node('refining', 870, cy, 170, t.refine);
    const completed = node('completed', 450, ty, 190, t.good, 'check');
    const failed = node('failed', 870, ty, 150, t.critical, 'cross');

    fig.arrow([[created.right + 4, cy], [running.left - 4, cy]], { color: t.ink2, size: 9 });
    fig.text((created.right + running.left) / 2, cy - 14, 'start', { size: TYPE.small, fill: t.ink2, anchor: 'middle' });

    // The loop between dynamics and refinement.
    const mid = (running.right + refining.left) / 2;
    fig.arrow([[running.right + 4, cy - 12], [refining.left - 4, cy - 12]], { color: t.ink2, size: 9 });
    fig.arrow([[refining.left - 4, cy + 12], [running.right + 4, cy + 12]], { color: t.ink2, size: 9 });
    fig.text(mid, cy - 26, 'checkpoint', { size: TYPE.small, fill: t.ink2, anchor: 'middle' });
    fig.text(mid, cy + 42, 'restore', { size: TYPE.small, fill: t.ink2, anchor: 'middle' });

    // Terminal phases.
    fig.arrow([[running.cx, cy + h / 2 + 4], [completed.cx, ty - h / 2 - 4]], { color: t.ink2, size: 9 });
    fig.text(running.cx - 14, (cy + ty) / 2, ['reached ', { t: 'total_steps', mono: true }], { size: TYPE.small, fill: t.ink2, anchor: 'end', middle: true });
    fig.text(failed.left - 16, ty, 'from any phase', { size: TYPE.small, fill: t.ink2, anchor: 'end', middle: true });
    fig.text(failed.right + 16, ty, 'run state not saved', { size: TYPE.small, fill: t.ink2, middle: true });
  },
};
