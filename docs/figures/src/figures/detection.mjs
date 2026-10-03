// Bond-change detection: one C-N pair run through the real detector and tracker (data/detection.json).

import { readFileSync } from 'node:fs';

import { Figure, scale, TYPE } from '../lib.mjs';

const data = JSON.parse(readFileSync(new URL('../../data/detection.json', import.meta.url)));

export default {
  name: 'detection',
  title: 'Detecting a bond change',
  alt:
    'Distance between one atom pair over 22 observation frames, in units of the summed covalent radii. ' +
    'A single frame below the formation threshold is ignored. Three consecutive frames below it confirm ' +
    'a formed bond, and three stable frames after that produce a reaction candidate.',
  caption:
    'One atom pair run through <code>BondChangeDetector</code> and <code>ReactionTracker</code> with the default settings. ' +
    'Formation needs the distance at or below 1.15 times the summed covalent radii for three consecutive observations; ' +
    'breaking needs 1.30 or more. The reactant and product snapshots are the two observations that bracket the crossing.',
  width: 1200,
  height: 512,
  draw(fig) {
    const t = fig.t;
    const [left, right, top, bottom] = [132, 1018, 28, 322];
    const frames = data.frames;
    const x = scale(-0.7, frames.length - 0.3, left, right);
    const y = scale(0.85, 1.75, bottom, top);
    const [form, brk] = [data.form_scale, data.break_scale];
    const candidate = data.candidates[0];
    const event = data.events[0];

    // Grid and axes.
    for (const tick of [1.0, 1.2, 1.4, 1.6]) {
      fig.line(left, y(tick), right, y(tick), { stroke: t.grid });
      fig.text(left - 12, y(tick), tick.toFixed(1), { size: TYPE.small, anchor: 'end', middle: true, fill: t.ink2 });
    }
    fig.line(left, bottom, right, bottom, { stroke: t.axis });
    for (const f of frames) {
      fig.line(x(f.frame), bottom, x(f.frame), bottom + 6, { stroke: t.axis });
      if (f.frame % 2 === 0) {
        fig.text(x(f.frame), bottom + 28, String(f.frame), { size: TYPE.small, anchor: 'middle', fill: t.ink2 });
      }
    }
    fig.text((left + right) / 2, bottom + 60, 'observation frame', { size: TYPE.body, anchor: 'middle', fill: t.ink2 });
    fig.text(left - 76, (top + bottom) / 2, ['distance / (r', { t: 'i', sub: true }, ' + r', { t: 'j', sub: true }, ')'], {
      size: TYPE.body,
      anchor: 'middle',
      fill: t.ink2,
      rotate: -90,
    });

    // Hysteresis gap and thresholds.
    fig.rect(left, y(brk), right - left, y(form) - y(brk), { fill: t.panel2 });
    for (const [value, label] of [[brk, `break ≥ ${brk.toFixed(2)}`], [form, `form ≤ ${form.toFixed(2)}`]]) {
      fig.line(left, y(value), right, y(value), { stroke: t.ink3, sw: 1.25, dash: '6 5' });
      fig.text(right + 12, y(value), label, { size: TYPE.body, weight: 600, middle: true });
    }
    fig.text(left + 14, y((form + brk) / 2), 'hysteresis gap', { size: TYPE.small, fill: t.ink2, middle: true });

    // Guides from the frames that matter down to the axis; the counters below line up with them.
    const laneA = bottom + 106;
    const laneB = bottom + 148;
    for (const frame of [event.first_seen_frame, event.confirmed_frame, candidate.observed_frame]) {
      fig.line(x(frame), y(frames[frame].ratio) + 11, x(frame), bottom, { stroke: t.axis, sw: 1 });
    }

    // Trace and markers.
    fig.path(Figure.polyline(frames.map((f) => [x(f.frame), y(f.ratio)])), { stroke: t.md, sw: 2.25, join: 'round', cap: 'round' });
    for (const f of frames) {
      const [cx, cy] = [x(f.frame), y(f.ratio)];
      fig.circle(cx, cy, 8.5, { fill: t.bg });
      if (f.persistence_count) fig.circle(cx, cy, 6.5, { fill: t.event });
      else if (f.bonded) fig.circle(cx, cy, 6.5, { fill: t.md });
      else fig.circle(cx, cy, 5.25, { fill: t.bg, stroke: t.md, sw: 2.25 });
    }

    // Legend.
    const legendX = x(15.6);
    [
      ['hollow', 'not bonded'],
      ['bonded', 'bonded'],
      ['pending', 'crossed, pending'],
    ].forEach(([kind, label], i) => {
      const ly = top + 14 + i * 32;
      if (kind === 'hollow') fig.circle(legendX, ly, 5.25, { fill: t.bg, stroke: t.md, sw: 2.25 });
      else fig.circle(legendX, ly, 6.5, { fill: kind === 'bonded' ? t.md : t.event });
      fig.text(legendX + 18, ly, label, { size: TYPE.small, middle: true, fill: t.ink2 });
    });

    // Annotations in the plot: [marker frame, leader end offset, text offset from that end, anchor, text].
    const note = (frame, [lx, ly], [dx, dy], anchor, row) => {
      const [px, py] = [x(frame), y(frames[frame].ratio)];
      const length = Math.hypot(lx, ly);
      if (length) {
        fig.line(px + (lx / length) * 11, py + (ly / length) * 11, px + lx, py + ly, { stroke: t.ink3, sw: 1 });
      }
      fig.text(px + lx + dx, py + ly + dy, row, { size: TYPE.small, anchor, fill: t.ink2, halo: true });
    };
    note(5, [-18, 18], [-4, 16], 'end', 'no event');
    note(candidate.reactant_frame, [30, -56], [5, 2], 'start', [{ t: 'reactant_frame', mono: true }]);
    note(candidate.product_frame, [-26, 36], [-5, 16], 'end', [{ t: 'product_frame', mono: true }]);
    note(17, [0, 0], [0, -20], 'middle', 'still bonded');

    // Counters under the plot.
    const chip = (frame, cy, label, kind) => {
      const [w, h] = [36, 30];
      const styles = {
        pending: { fill: fig.tint(t.event, 0.16), stroke: t.event, color: t.ink, weight: 500 },
        done: { fill: t.event, stroke: t.event, color: '#ffffff', weight: 700 },
        reset: { fill: t.bg, stroke: t.axis, color: t.ink2, weight: 500 },
      }[kind];
      fig.rect(x(frame) - w / 2, cy - h / 2, w, h, { rx: 6, fill: styles.fill, stroke: styles.stroke, sw: 1.25 });
      fig.text(x(frame), cy, label, { size: TYPE.small * 0.9, anchor: 'middle', middle: true, fill: styles.color, weight: styles.weight });
    };
    const total = data.persistence_frames;
    fig.text(36, laneA, 'persistence', { size: TYPE.body, weight: 600, middle: true });
    fig.text(36, laneB, 'stability', { size: TYPE.body, weight: 600, middle: true });
    for (const f of frames) {
      if (f.persistence_count) chip(f.frame, laneA, `${f.persistence_count}/${total}`, 'pending');
    }
    chip(6, laneA, `0/${total}`, 'reset');
    chip(event.confirmed_frame, laneA, `${total}/${total}`, 'done');
    fig.text(x(event.confirmed_frame) + 30, laneA, 'bond formed', { size: TYPE.small, middle: true, fill: t.ink2 });
    for (const f of frames) {
      if (f.stability_count) chip(f.frame, laneB, `${f.stability_count}/${data.stability_frames}`, 'pending');
    }
    chip(candidate.observed_frame, laneB, `${data.stability_frames}/${data.stability_frames}`, 'done');
    fig.text(x(candidate.observed_frame) + 30, laneB, 'reaction candidate', { size: TYPE.small, middle: true, fill: t.ink2 });
  },
};
