// Small ball-and-stick drawings shared by the diagrams.

/** Acetonitrile (CH3-C≡N) in projection, centered on its nitrile carbon; s is pixels per Å. */
export function acetonitrile(fig, x, y, degrees, s = 13, o = {}) {
  const [c, n] = [Math.cos((degrees * Math.PI) / 180), Math.sin((degrees * Math.PI) / 180)];
  const at = ([ax, ay]) => [x + (ax * c - ay * n) * s, y + (ax * n + ay * c) * s];
  const [nitrogen, carbon, methyl] = [at([1.16, 0]), at([0, 0]), at([-1.46, 0])];
  const hydrogens = [[-1.84, 1.0], [-1.84, -1.0], [-2.5, 0]].map(at);
  for (const h of hydrogens) fig.bond(...methyl, ...h, { w: s * 0.2, opacity: o.opacity });
  fig.bond(...methyl, ...carbon, { w: s * 0.26, opacity: o.opacity });
  fig.bond(...carbon, ...nitrogen, { kind: 'triple', w: s * 0.2, opacity: o.opacity });
  for (const h of hydrogens) fig.atom(...h, s * 0.3, 'H', o);
  fig.atom(...methyl, s * 0.5, 'C', o);
  fig.atom(...carbon, s * 0.5, 'C', o);
  fig.atom(...nitrogen, s * 0.47, 'N', o);
  return { nitrogen, carbon, methyl };
}

/** Disk icon for a durable checkpoint. */
export function disk(fig, x, y, size, color) {
  const s = size / 40;
  const t = fig.t;
  fig.path(
    `M${x + 4 * s} ${y} h${26 * s} l${10 * s} ${10 * s} v${26 * s} a${4 * s} ${4 * s} 0 0 1 ${-4 * s} ${4 * s} h${-32 * s} ` +
      `a${4 * s} ${4 * s} 0 0 1 ${-4 * s} ${-4 * s} v${-32 * s} a${4 * s} ${4 * s} 0 0 1 ${4 * s} ${-4 * s} z`,
    { fill: color },
  );
  fig.rect(x + 9 * s, y + 3 * s, 17 * s, 11 * s, { rx: 1.5 * s, fill: t.bg, opacity: 0.9 });
  fig.rect(x + 19 * s, y + 5 * s, 4 * s, 7 * s, { rx: 1 * s, fill: color });
  fig.rect(x + 7 * s, y + 21 * s, 26 * s, 16 * s, { rx: 2 * s, fill: t.bg, opacity: 0.9 });
}
