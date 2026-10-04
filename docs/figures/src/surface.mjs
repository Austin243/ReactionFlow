// The two-dimensional model potential behind data/pathway.json, and a vector contour renderer.

import { readFileSync } from 'node:fs';

import { mix, scale } from './lib.mjs';

export const pathway = JSON.parse(readFileSync(new URL('../data/pathway.json', import.meta.url)));
const S = pathway.surface;

/** Potential energy in eV; the same expression as surface_energy() in make_data.py. */
export function energy(x, y) {
  const floor = S.g * (x ** 4 / 4 - (0.1 * x ** 3) / 3 - 0.605 * x ** 2 - 0.12 * x + 0.2016666667);
  const valley = S.c * (1 - ((x - S.center) / S.half_width) ** 2);
  return floor + 0.5 * S.k * (y - valley) ** 2;
}

/** Gradient [dE/dx, dE/dy]. */
export function gradient(x, y) {
  const dfloor = S.g * (x + 1) * (x + 0.1) * (x - 1.2);
  const offset = y - S.c * (1 - ((x - S.center) / S.half_width) ** 2);
  const dvalley = (-2 * S.c * (x - S.center)) / S.half_width ** 2;
  return [dfloor - S.k * offset * dvalley, S.k * offset];
}

export const DOMAIN = { x0: -1.72, x1: 1.92, y0: -0.72, y1: 1.44 };
export const LEVELS = Array.from({ length: 15 }, (_, i) => -1 + 0.25 * i);

/**
 * Closed iso-lines of a gridded field at one level (marching squares). The outer ring of the grid
 * is treated as above every level, so each line closes on itself and an even-odd fill of the
 * loops covers exactly the region below the level.
 */
function isoLoops(values, xs, ys, level) {
  const [nx, ny] = [xs.length, ys.length];
  const high = (i, j) => values[j][i] >= level;
  const point = (edge) => {
    const [kind, i, j] = edge.split(':').map((v, n) => (n ? Number(v) : v));
    const [i2, j2] = kind === 'H' ? [i + 1, j] : [i, j + 1];
    const [a, b] = [values[j][i], values[j2][i2]];
    const f = Math.min(1, Math.max(0, (level - a) / (b - a)));
    return [xs[i] + f * (xs[i2] - xs[i]), ys[j] + f * (ys[j2] - ys[j])];
  };
  const links = new Map();
  const link = (a, b) => {
    for (const [from, to] of [[a, b], [b, a]]) {
      if (!links.has(from)) links.set(from, []);
      links.get(from).push(to);
    }
  };
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const [top, right, bottom, left] = [`H:${i}:${j}`, `V:${i + 1}:${j}`, `H:${i}:${j + 1}`, `V:${i}:${j}`];
      const code = (high(i, j) ? 8 : 0) | (high(i + 1, j) ? 4 : 0) | (high(i + 1, j + 1) ? 2 : 0) | (high(i, j + 1) ? 1 : 0);
      const center = (values[j][i] + values[j][i + 1] + values[j + 1][i + 1] + values[j + 1][i]) / 4 >= level;
      const cases = {
        1: [[left, bottom]], 2: [[bottom, right]], 3: [[left, right]], 4: [[top, right]],
        5: center ? [[top, left], [bottom, right]] : [[top, right], [left, bottom]],
        6: [[top, bottom]], 7: [[top, left]], 8: [[top, left]], 9: [[top, bottom]],
        10: center ? [[top, right], [left, bottom]] : [[top, left], [bottom, right]],
        11: [[top, right]], 12: [[left, right]], 13: [[bottom, right]], 14: [[left, bottom]],
      }[code];
      for (const [a, b] of cases ?? []) link(a, b);
    }
  }
  const loops = [];
  const seen = new Set();
  for (const start of links.keys()) {
    if (seen.has(start)) continue;
    const loop = [];
    let [previous, current] = [null, start];
    while (current && !seen.has(current)) {
      seen.add(current);
      loop.push(point(current));
      const next = links.get(current).find((edge) => edge !== previous && !seen.has(edge));
      [previous, current] = [current, next];
    }
    if (loop.length > 2) loops.push(loop);
  }
  return loops;
}

/**
 * Draw the filled, contoured surface into a rectangle. Low energy recedes toward the page color.
 * `field` and `levels` default to this model potential; pass another surface's with its own key.
 * Returns the maps from model coordinates (Å) to pixels.
 */
export function drawSurface(fig, key, rect, { domain = DOMAIN, columns = 132, rx = 8, field = energy, levels = LEVELS } = {}) {
  const t = fig.t;
  const group = fig.def(`surface-${key}`, (id) => {
    const rows = Math.round((columns * rect.h) / rect.w);
    // One extra ring of nodes outside the visible rectangle carries the "above every level" value.
    const xs = Array.from({ length: columns + 3 }, (_, i) => ((i - 1) / columns) * rect.w);
    const ys = Array.from({ length: rows + 3 }, (_, j) => ((j - 1) / rows) * rect.h);
    const dataX = scale(0, rect.w, domain.x0, domain.x1);
    const dataY = scale(0, rect.h, domain.y1, domain.y0);
    const values = ys.map((py, j) =>
      xs.map((px, i) => (i === 0 || j === 0 || i === xs.length - 1 || j === ys.length - 1 ? 1e9 : field(dataX(px), dataY(py)))),
    );
    const color = (k) => mix(t.surfaceLow, t.surfaceHigh, k / levels.length);
    let body = `<rect width="${rect.w}" height="${rect.h}" fill="${color(levels.length)}"/>`;
    for (let k = levels.length - 1; k >= 0; k--) {
      const d = isoLoops(values, xs, ys, levels[k])
        .map((loop) => loop.map(([px, py], n) => `${n ? 'L' : 'M'}${px.toFixed(1)} ${py.toFixed(1)}`).join('') + 'Z')
        .join('');
      body += `<path d="${d}" fill="${color(k)}" fill-rule="evenodd" stroke="${t.contour}" stroke-width="0.6" stroke-opacity="0.55"/>`;
    }
    return `<g id="${id}">${body}</g>`;
  });
  const clip = fig.clipRect(`surface-clip-${key}-${Math.round(rect.x)}-${Math.round(rect.y)}`, rect.x, rect.y, rect.w, rect.h, rx);
  fig.add(`<g clip-path="url(#${clip})"><use xlink:href="#${group}" x="${rect.x}" y="${rect.y}"/></g>`);
  fig.rect(rect.x, rect.y, rect.w, rect.h, { rx, stroke: t.border });
  return {
    sx: scale(domain.x0, domain.x1, rect.x, rect.x + rect.w),
    sy: scale(domain.y0, domain.y1, rect.y + rect.h, rect.y),
    perAngstrom: rect.w / (domain.x1 - domain.x0),
  };
}
