// The two-dimensional landscape behind data/eon.json: Gaussian wells on a walled bowl, searched by
// ReactionFlow's EON mode.

import { readFileSync } from 'node:fs';

export const eon = JSON.parse(readFileSync(new URL('../data/eon.json', import.meta.url)));
const L = eon.landscape;

/** Energy in eV; the same expression as landscape_energy() in make_data.py. */
export function energy(x, y) {
  let e = 0;
  for (const [wx, wy, depth, width] of L.wells) e -= depth * Math.exp(-((x - wx) ** 2 + (y - wy) ** 2) / (2 * width ** 2));
  const r = Math.hypot(x - L.center[0], y - L.center[1]);
  const outside = Math.max(0, r - L.wall_radius_A);
  return e + 0.5 * L.bowl * r * r + 0.5 * L.wall * outside * outside;
}

export const DOMAIN = { x0: -3.25, x1: 3.5, y0: -1.75, y1: 2.45 };
export const LEVELS = Array.from({ length: 15 }, (_, i) => -2.8 + 0.2 * i);

/** The number in a state ID, e.g. 3 for state-000003. */
export const stateNumber = (id) => Number(id.split('-')[1]);

/** A time in seconds with two significant figures and a readable unit. */
export function duration(seconds) {
  const units = [[3600, 'h'], [60, 'min'], [1, 's'], [1e-3, 'ms'], [1e-6, 'µs'], [1e-9, 'ns'], [1e-12, 'ps']];
  const [size, unit] = units.find(([size]) => seconds >= size) ?? units.at(-1);
  const value = seconds / size;
  return `${value >= 10 ? value.toFixed(0) : value.toPrecision(2)} ${unit}`;
}

/** EON's confidence after `repeats` consecutive repeats of known processes. */
export const confidence = (repeats) => (repeats < 1 ? 0 : 1 - 1 / repeats);

/** Harmonic transition-state rate in 1/s. */
export const rate = (process) => process.prefactor_s * Math.exp(-process.barrier_eV / eon.akmc.kT_eV);

/**
 * The run in the order it happened, replayed with the runner's rules: every search with the
 * confidence of its state afterwards, and every KMC step, taken once a state reaches the target,
 * with the processes it chose from. The replayed rates must add up to the recorded total.
 */
export function events() {
  const repeats = {};
  const known = {};
  const out = [];
  let current = eon.attempts[0].state;
  let [a, s] = [0, 0];
  while (s < eon.steps.length) {
    repeats[current] ??= 0;
    known[current] ??= [];
    if (confidence(repeats[current]) >= eon.akmc.confidence) {
      const step = eon.steps[s++];
      const processes = known[current].map((id) => ({ id, ...eon.processes[id] }));
      const lowest = Math.min(...processes.map((p) => p.barrier_eV));
      const window = processes.filter((p) => p.barrier_eV <= lowest + eon.akmc.thermal_window_kT * eon.akmc.kT_eV);
      const total = window.reduce((sum, p) => sum + rate(p), 0);
      if (step.from !== current || Math.abs(total / step.total_rate_s - 1) > 1e-9) throw new Error('step does not replay');
      out.push({ kind: 'step', step, index: s - 1, processes, window });
      current = step.to;
      continue;
    }
    const attempt = eon.attempts[a++];
    if (attempt.state !== current) throw new Error('search out of order');
    if (attempt.status === 'new') {
      repeats[current] = 0;
      known[current].push(attempt.process);
      if (attempt.reverse) (known[attempt.product] ??= []).push(attempt.reverse);
    } else if (attempt.status === 'repeat' && attempt.relevant) repeats[current] += 1;
    out.push({ kind: 'search', attempt, confidence: confidence(repeats[current]) });
  }
  return out;
}
