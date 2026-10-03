// A short Langevin trajectory on the model surface, used only as an illustration of MD that
// crosses from the reactant basin to the product basin.

import { gradient, pathway } from './surface.mjs';

const TIME_UNIT_FS = 10.1805; // sqrt(amu Å² / eV)

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * BAOAB Langevin dynamics of one particle (4 amu) started at the reactant minimum.
 * Returns positions every `stride` femtoseconds.
 */
export function langevin({ seed, steps, kT = 0.13, frictionFs = 100, stride = 1 }) {
  const random = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  const [mass, dt] = [4, 1 / TIME_UNIT_FS];
  const decay = Math.exp(-dt / (frictionFs / TIME_UNIT_FS));
  const kick = Math.sqrt((kT / mass) * (1 - decay * decay));
  let [x, y] = pathway.images[0];
  let [vx, vy] = [gauss() * Math.sqrt(kT / mass), gauss() * Math.sqrt(kT / mass)];
  let [gx, gy] = gradient(x, y);
  const points = [[x, y]];
  for (let step = 1; step <= steps; step++) {
    vx -= (0.5 * dt * gx) / mass;
    vy -= (0.5 * dt * gy) / mass;
    x += 0.5 * dt * vx;
    y += 0.5 * dt * vy;
    vx = decay * vx + kick * gauss();
    vy = decay * vy + kick * gauss();
    x += 0.5 * dt * vx;
    y += 0.5 * dt * vy;
    [gx, gy] = gradient(x, y);
    vx -= (0.5 * dt * gx) / mass;
    vy -= (0.5 * dt * gy) / mass;
    if (step % stride === 0) points.push([x, y]);
  }
  return points;
}

/**
 * First clean reactant-to-product crossing in a trajectory: the index where the particle last
 * left the reactant side before settling in the product basin, or null.
 */
export function firstCrossing(points, settle = 600) {
  const saddleX = pathway.images[pathway.frequency.saddle_index][0];
  for (let i = 1; i < points.length - settle; i++) {
    if (points[i - 1][0] < saddleX && points[i][0] >= saddleX) {
      if (points.slice(i, i + settle).every(([x]) => x > saddleX)) return i;
    }
  }
  return null;
}
