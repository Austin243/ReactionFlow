// Small SVG toolkit shared by every figure: themes, text metrics, shapes, arrows, and atoms.
// No dependencies. Every color is written as a plain attribute so the files stay editable in
// Illustrator, Inkscape, or Figma.

export const SANS =
  "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";
export const MONO =
  "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace";

// Surfaces and inks follow GitHub's light and dark pages. The three role colors are the first
// three slots of a palette validated for color-vision deficiency on both surfaces:
// blue = molecular dynamics, orange = detection, aqua = pathway refinement.
export const THEMES = {
  light: {
    name: 'light',
    bg: '#ffffff',
    panel: '#f6f8fa',
    panel2: '#eaeef2',
    border: '#d1d9e0',
    ink: '#1f2328',
    ink2: '#59636e',
    ink3: '#818b98',
    grid: '#e7ebef',
    axis: '#c0c8d1',
    md: '#2a78d6',
    event: '#eb6834',
    refine: '#1baf7a',
    good: '#0ca30c',
    warn: '#fab219',
    serious: '#ec835a',
    critical: '#d03b3b',
    ramp: ['#86b6ef', '#2a78d6', '#104281'],
    surfaceLow: '#f8fafc',
    surfaceHigh: '#a8b8cd',
    contour: '#8598b3',
    atomStroke: '#000000',
    atomStrokeOpacity: 0.3,
    bond: '#6e7781',
  },
  dark: {
    name: 'dark',
    bg: '#0d1117',
    panel: '#151b23',
    panel2: '#1e2630',
    border: '#3d444d',
    ink: '#f0f6fc',
    ink2: '#a3adb8',
    ink3: '#768390',
    grid: '#1f262e',
    axis: '#3d444d',
    md: '#3987e5',
    event: '#d95926',
    refine: '#199e70',
    good: '#0ca30c',
    warn: '#fab219',
    serious: '#ec835a',
    critical: '#d03b3b',
    ramp: ['#184f95', '#3987e5', '#9ec5f4'],
    surfaceLow: '#0e131a',
    surfaceHigh: '#3c4a60',
    contour: '#5d6c82',
    atomStroke: '#ffffff',
    atomStrokeOpacity: 0.35,
    bond: '#8b949e',
  },
};

const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));

/** Blend two hex colors; t = 0 gives a, t = 1 gives b. */
export function mix(a, b, t) {
  const [x, y] = [hex(a), hex(b)];
  return '#' + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, '0')).join('');
}

const esc = (s) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const r2 = (v) => (Number.isInteger(v) ? String(v) : String(+v.toFixed(2)));

// Advance widths in thousandths of an em for a Helvetica-like sans. System fonts differ by a few
// percent, so layouts leave slack rather than fitting text exactly.
const W = {};
const widths = (chars, w) => [...chars].forEach((c) => (W[c] = w));
widths(" !,.:;/I[]\\'|", 278);
widths('fijlt', 250);
widths('r()-`', 333);
widths('"*', 370);
widths('ckvxyzsJ', 500);
widths('abdeghnopqu0123456789$#?_L', 556);
widths('+<=>~≤≥±×÷≈', 584);
widths('FTZ', 611);
widths('ABEKPSVXYΔÅ&', 667);
widths('CDHNRUw', 722);
widths('GOQ', 778);
widths('mM', 833);
widths('%', 889);
widths('W', 944);
widths('→←↑↓—…', 1000);
widths('·', 278);
widths('–‡', 556);
widths('⁻¹²³', 350);
widths('✓✕', 800);

/** Approximate rendered width of a string. */
export function textWidth(s, size, { mono = false, weight = 400 } = {}) {
  if (mono) return String(s).length * size * 0.602;
  let total = 0;
  for (const c of String(s)) total += W[c] ?? 600;
  // Calibrated against the macOS system font, which tracks small sizes wider and large sizes tighter.
  const tracking = size <= 14.5 ? 1.045 : size <= 17 ? 1.02 : size >= 20 ? 0.985 : 1;
  return (total / 1000) * size * tracking * (weight >= 600 ? 1.06 : weight >= 500 ? 1.03 : 1);
}

/** Break a sentence into lines no wider than maxWidth. */
export function wrap(s, maxWidth, o = {}) {
  const rows = [];
  let row = '';
  for (const word of String(s).split(' ')) {
    const next = row ? `${row} ${word}` : word;
    if (row && textWidth(next, o.size ?? 16, o) > maxWidth) {
      rows.push(row);
      row = word;
    } else row = next;
  }
  if (row) rows.push(row);
  return rows;
}

// Type sizes for every figure, in SVG units. Figures are 1200 units wide and a GitHub README shows
// them about 840 px wide, so `small` renders near 14 px and `head` near 18 px.
export const TYPE = { small: 20, body: 22, head: 26 };

// The multi-panel figures fit more into the same width, so their text is a step larger: `small`
// renders near 17 px at README width, the size of the README's own text.
export const HERO_TYPE = { small: 24, body: 26, head: 30 };

const ELEMENTS = {
  C: ['#d4d4d4', '#737373', '#2e2e2e'],
  N: ['#c3d0ff', '#4264d9', '#1c2e7c'],
  H: ['#ffffff', '#ececec', '#9c9c9c'],
  O: ['#ffc7bf', '#e0452f', '#8c1c10'],
};

export class Figure {
  constructor({ id, width, height, theme, title, desc }) {
    this.id = id;
    this.w = width;
    this.h = height;
    this.t = THEMES[theme];
    this.title = title;
    this.desc = desc;
    this.out = [];
    this.defs = new Map();
  }

  add(s) {
    this.out.push(s);
    return this;
  }

  def(key, body) {
    const id = `${this.id}-${key}`;
    if (!this.defs.has(id)) this.defs.set(id, body(id));
    return id;
  }

  /** Blend a role color toward the surface for fills that sit behind text. */
  tint(color, amount = 0.12) {
    return mix(this.t.bg, color, amount);
  }

  attrs(o) {
    return Object.entries(o)
      .filter(([, v]) => v !== undefined && v !== null && v !== false)
      .map(([k, v]) => `${k}="${typeof v === 'number' ? r2(v) : esc(v)}"`)
      .join(' ');
  }

  shape(o) {
    return {
      fill: o.fill ?? 'none',
      'fill-opacity': o.fillOpacity,
      stroke: o.stroke,
      'stroke-width': o.stroke ? o.sw ?? 1 : undefined,
      'stroke-dasharray': o.dash,
      'stroke-linecap': o.cap,
      'stroke-linejoin': o.join,
      'stroke-opacity': o.strokeOpacity,
      opacity: o.opacity,
      'clip-path': o.clip ? `url(#${o.clip})` : undefined,
      transform: o.transform,
    };
  }

  rect(x, y, w, h, o = {}) {
    return this.add(`<rect ${this.attrs({ x, y, width: w, height: h, rx: o.rx, ...this.shape(o) })}/>`);
  }

  circle(cx, cy, r, o = {}) {
    return this.add(`<circle ${this.attrs({ cx, cy, r, ...this.shape(o) })}/>`);
  }

  line(x1, y1, x2, y2, o = {}) {
    return this.add(
      `<line ${this.attrs({ x1, y1, x2, y2, ...this.shape({ stroke: this.t.ink2, ...o, fill: undefined }) })}/>`,
    );
  }

  path(d, o = {}) {
    return this.add(`<path ${this.attrs({ d, ...this.shape(o) })}/>`);
  }

  polygon(points, o = {}) {
    const p = points.map(([x, y]) => `${r2(x)},${r2(y)}`).join(' ');
    return this.add(`<polygon ${this.attrs({ points: p, ...this.shape(o) })}/>`);
  }

  /** Open polyline through points, as a path string. */
  static polyline(points) {
    return points.map(([x, y], i) => `${i ? 'L' : 'M'}${r2(x)} ${r2(y)}`).join(' ');
  }

  /** Smooth curve through points (Catmull-Rom converted to cubic Béziers). */
  static smooth(points, tension = 0.5) {
    if (points.length < 3) return Figure.polyline(points);
    let d = `M${r2(points[0][0])} ${r2(points[0][1])}`;
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i - 1] ?? points[i];
      const p1 = points[i];
      const p2 = points[i + 1];
      const p3 = points[i + 2] ?? p2;
      const c1 = [p1[0] + ((p2[0] - p0[0]) * tension) / 3, p1[1] + ((p2[1] - p0[1]) * tension) / 3];
      const c2 = [p2[0] - ((p3[0] - p1[0]) * tension) / 3, p2[1] - ((p3[1] - p1[1]) * tension) / 3];
      d += ` C${r2(c1[0])} ${r2(c1[1])} ${r2(c2[0])} ${r2(c2[1])} ${r2(p2[0])} ${r2(p2[1])}`;
    }
    return d;
  }

  group(o, draw) {
    this.add(`<g ${this.attrs({ transform: o.transform, opacity: o.opacity, 'clip-path': o.clip ? `url(#${o.clip})` : undefined })}>`);
    draw();
    return this.add('</g>');
  }

  clipRect(key, x, y, w, h, rx = 0) {
    return this.def(key, (id) => `<clipPath id="${id}"><rect ${this.attrs({ x, y, width: w, height: h, rx })}/></clipPath>`);
  }

  /**
   * One line of text. `s` is a string or an array of runs: strings or
   * { t, mono, weight, fill, italic, sub, sup }.
   * y is the baseline, or the vertical center when o.middle is set.
   */
  text(x, y, s, o = {}) {
    const size = o.size ?? 16;
    const runs = (Array.isArray(s) ? s : [s]).map((run) => (typeof run === 'string' ? { t: run } : run));
    const baseline = o.middle ? y + size * 0.355 : y;
    let shift = 0;
    const body = runs
      .map((run) => {
        const script = run.sub ? 1 : run.sup ? -1 : 0;
        const target = script * size * (run.sub ? 0.22 : 0.36);
        const dy = target - shift;
        shift = target;
        const a = this.attrs({
          'font-family': (run.mono ?? o.mono) ? MONO : undefined,
          'font-weight': run.weight,
          'font-style': run.italic ? 'italic' : undefined,
          'font-size': script ? size * 0.72 : (run.mono ?? o.mono) ? size * 0.94 : undefined,
          fill: run.fill,
          dy: dy ? r2(dy) : undefined,
        });
        return a ? `<tspan ${a}>${esc(run.t)}</tspan>` : esc(run.t);
      })
      .join('');
    if (o.halo) {
      // Same text underneath with a surface-colored outline, so it stays readable over lines and fills.
      this.add(
        `<text ${this.attrs({
          x,
          y: baseline,
          'font-size': size,
          'font-weight': o.weight,
          'text-anchor': o.anchor === 'middle' || o.anchor === 'end' ? o.anchor : undefined,
          fill: 'none',
          stroke: o.halo === true ? this.t.bg : o.halo,
          'stroke-width': o.haloWidth ?? 4,
          'stroke-linejoin': 'round',
        })}>${body}</text>`,
      );
    }
    return this.add(
      `<text ${this.attrs({
        x,
        y: baseline,
        'font-size': size,
        'font-weight': o.weight,
        'font-style': o.italic ? 'italic' : undefined,
        'text-anchor': o.anchor === 'middle' || o.anchor === 'end' ? o.anchor : undefined,
        'letter-spacing': o.spacing,
        fill: o.fill ?? this.t.ink,
        opacity: o.opacity,
        transform: o.rotate ? `rotate(${o.rotate} ${r2(x)} ${r2(baseline)})` : undefined,
      })}>${body}</text>`,
    );
  }

  /** Width of what text() would draw for the same arguments. */
  measure(s, o = {}) {
    const size = o.size ?? 16;
    return (Array.isArray(s) ? s : [s])
      .map((run) => (typeof run === 'string' ? { t: run } : run))
      .reduce((total, run) => {
        const mono = run.mono ?? o.mono;
        const scale = run.sub || run.sup ? 0.72 : mono ? 0.94 : 1;
        return total + textWidth(run.t, size * scale, { mono, weight: run.weight ?? o.weight ?? 400 });
      }, 0);
  }

  /** Several lines; y is the first baseline. */
  lines(x, y, rows, o = {}) {
    const lh = o.lh ?? (o.size ?? 16) * 1.38;
    rows.forEach((row, i) => this.text(x, y + i * lh, row, o));
    return this;
  }

  /** Rounded panel. */
  card(x, y, w, h, o = {}) {
    return this.rect(x, y, w, h, {
      rx: o.rx ?? 10,
      fill: o.fill ?? this.t.panel,
      stroke: o.stroke ?? this.t.border,
      sw: o.sw ?? 1,
      dash: o.dash,
      opacity: o.opacity,
    });
  }

  /** Rounded tag sized to its label. Returns its width. x is the left edge unless o.anchor says otherwise. */
  pill(x, y, label, o = {}) {
    const size = o.size ?? 14;
    const padX = o.padX ?? size * 0.72;
    const h = o.h ?? size * 1.75;
    const dot = o.dot ? size * 0.62 + 6 : 0;
    const w = this.measure(label, { size, mono: o.mono, weight: o.weight ?? 500 }) + 2 * padX + dot;
    const left = o.anchor === 'middle' ? x - w / 2 : o.anchor === 'end' ? x - w : x;
    this.rect(left, y - h / 2, w, h, {
      rx: o.rx ?? h / 2,
      fill: o.fill ?? this.t.panel2,
      stroke: o.stroke,
      sw: o.sw ?? 1,
    });
    if (o.dot) this.circle(left + padX + size * 0.31, y, size * 0.31, { fill: o.dot });
    this.text(left + padX + dot, y, label, {
      size,
      mono: o.mono,
      weight: o.weight ?? 500,
      fill: o.color ?? this.t.ink,
      middle: true,
    });
    return w;
  }

  /** Filled arrowhead with its tip at (x, y), pointing along angle (radians). */
  head(x, y, angle, o = {}) {
    const size = o.size ?? 9;
    const half = (o.spread ?? 0.62) * size;
    const [c, s] = [Math.cos(angle), Math.sin(angle)];
    const [bx, by] = [x - size * c, y - size * s];
    return this.polygon(
      [
        [x, y],
        [bx - half * s, by + half * c],
        [bx + half * s, by - half * c],
      ],
      { fill: o.color ?? this.t.ink2, opacity: o.opacity },
    );
  }

  /** Polyline arrow; corners are rounded by o.radius. The line stops inside the head. */
  arrow(points, o = {}) {
    const color = o.color ?? this.t.ink2;
    const size = o.head === false ? 0 : o.size ?? 9;
    const pts = points.map((p) => [...p]);
    const [a, b] = [pts[pts.length - 2], pts[pts.length - 1]];
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const tip = [...b];
    pts[pts.length - 1] = [b[0] - 0.75 * size * Math.cos(angle), b[1] - 0.75 * size * Math.sin(angle)];
    let startAngle;
    if (o.both) {
      startAngle = Math.atan2(pts[0][1] - pts[1][1], pts[0][0] - pts[1][0]);
      const start = [...pts[0]];
      pts[0] = [start[0] - 0.75 * size * Math.cos(startAngle), start[1] - 0.75 * size * Math.sin(startAngle)];
      this.head(start[0], start[1], startAngle, { color, size, opacity: o.opacity });
    }
    const radius = o.radius ?? 0;
    let d = `M${r2(pts[0][0])} ${r2(pts[0][1])}`;
    for (let i = 1; i < pts.length; i++) {
      const p = pts[i];
      const next = pts[i + 1];
      if (!next || !radius) {
        d += ` L${r2(p[0])} ${r2(p[1])}`;
        continue;
      }
      const prev = pts[i - 1];
      const inLen = Math.hypot(p[0] - prev[0], p[1] - prev[1]);
      const outLen = Math.hypot(next[0] - p[0], next[1] - p[1]);
      const rr = Math.min(radius, inLen / 2, outLen / 2);
      const p1 = [p[0] - ((p[0] - prev[0]) / inLen) * rr, p[1] - ((p[1] - prev[1]) / inLen) * rr];
      const p2 = [p[0] + ((next[0] - p[0]) / outLen) * rr, p[1] + ((next[1] - p[1]) / outLen) * rr];
      d += ` L${r2(p1[0])} ${r2(p1[1])} Q${r2(p[0])} ${r2(p[1])} ${r2(p2[0])} ${r2(p2[1])}`;
    }
    this.path(d, { stroke: color, sw: o.sw ?? 1.75, dash: o.dash, cap: 'round', join: 'round', opacity: o.opacity });
    if (size) this.head(tip[0], tip[1], angle, { color, size, opacity: o.opacity });
    return this;
  }

  /** Cubic Bézier arrow from p0 to p1 with control points c1 and c2. */
  curve(p0, c1, c2, p1, o = {}) {
    const color = o.color ?? this.t.ink2;
    const size = o.head === false ? 0 : o.size ?? 9;
    const angle = Math.atan2(p1[1] - c2[1], p1[0] - c2[0]);
    const end = [p1[0] - 0.75 * size * Math.cos(angle), p1[1] - 0.75 * size * Math.sin(angle)];
    this.path(
      `M${r2(p0[0])} ${r2(p0[1])} C${r2(c1[0])} ${r2(c1[1])} ${r2(c2[0])} ${r2(c2[1])} ${r2(end[0])} ${r2(end[1])}`,
      { stroke: color, sw: o.sw ?? 1.75, dash: o.dash, cap: 'round', opacity: o.opacity },
    );
    if (size) this.head(p1[0], p1[1], angle, { color, size, opacity: o.opacity });
    return this;
  }

  /** Shaded sphere for an element (C, N, H, O). */
  atom(x, y, r, el, o = {}) {
    const [light, mid, dark] = ELEMENTS[el];
    const id = this.def(`atom-${el}`, (gid) =>
      `<radialGradient id="${gid}" cx="0.36" cy="0.32" r="0.72"><stop offset="0" stop-color="${light}"/>` +
      `<stop offset="0.55" stop-color="${mid}"/><stop offset="1" stop-color="${dark}"/></radialGradient>`,
    );
    return this.circle(x, y, r, {
      fill: `url(#${id})`,
      stroke: this.t.atomStroke,
      strokeOpacity: this.t.atomStrokeOpacity,
      sw: 0.75,
      opacity: o.opacity,
    });
  }

  /** Bond between two atom centers. kind: 'single' | 'double' | 'triple' | 'forming' | 'breaking'. */
  bond(x1, y1, x2, y2, o = {}) {
    const kind = o.kind ?? 'single';
    const w = o.w ?? 4;
    const color = o.color ?? this.t.bond;
    const count = kind === 'double' ? 2 : kind === 'triple' ? 3 : 1;
    const len = Math.hypot(x2 - x1, y2 - y1);
    const [nx, ny] = [-(y2 - y1) / len, (x2 - x1) / len];
    for (let i = 0; i < count; i++) {
      const off = (i - (count - 1) / 2) * w * 1.25;
      this.line(x1 + nx * off, y1 + ny * off, x2 + nx * off, y2 + ny * off, {
        stroke: color,
        sw: count > 1 ? w * 0.62 : w,
        cap: 'round',
        dash: kind === 'forming' || kind === 'breaking' ? `${r2(w * 0.4)} ${r2(w * 1.5)}` : undefined,
        opacity: o.opacity,
      });
    }
    return this;
  }

  /** Soft halo that marks a detected bond change. */
  glow(x, y, r, color) {
    const id = this.def(`glow-${color.slice(1)}`, (gid) =>
      `<radialGradient id="${gid}"><stop offset="0" stop-color="${color}" stop-opacity="0.55"/>` +
      `<stop offset="0.55" stop-color="${color}" stop-opacity="0.2"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></radialGradient>`,
    );
    return this.circle(x, y, r, { fill: `url(#${id})` });
  }

  /** Panel letter and title, as in a multi-panel figure; y is the baseline. */
  heading(x, y, letter, title) {
    this.text(x, y, letter, { size: HERO_TYPE.head, weight: 750 });
    return this.text(x + 34, y, title, { size: HERO_TYPE.head, weight: 650 });
  }

  /** Legend items in a row, each [draw(x, y), label] with the mark centered 11 units in. Returns the end x. */
  keys(x, y, items, o = {}) {
    for (const [draw, label] of items) {
      draw(x + 11, y);
      this.text(x + 30, y, label, { size: HERO_TYPE.small, middle: true, fill: o.fill ?? this.t.ink2, halo: o.halo });
      x += 30 + this.measure(label, { size: HERO_TYPE.small }) + (o.gap ?? 32);
    }
    return x;
  }

  /** Saddle point marker, a triangle centered on (x, y). */
  saddle(x, y, r, o = {}) {
    return this.polygon([[x, y - r * 1.15], [x + r, y + r * 0.75], [x - r, y + r * 0.75]], {
      fill: o.fill ?? this.t.refine,
      stroke: o.stroke ?? this.t.bg,
      sw: o.sw ?? 2,
      join: 'round',
    });
  }

  /** Numbered step badge. */
  badge(x, y, label, o = {}) {
    const r = o.r ?? 13;
    this.circle(x, y, r, { fill: o.fill ?? this.t.ink });
    return this.text(x, y, String(label), {
      size: r * 1.08,
      weight: 700,
      anchor: 'middle',
      middle: true,
      fill: o.color ?? this.t.bg,
    });
  }

  /** Check mark, cross, or pause glyph drawn as strokes, centered on (x, y). */
  glyph(kind, x, y, size, color) {
    const s = size / 2;
    const stroke = { stroke: color, sw: Math.max(1.6, size * 0.16), cap: 'round', join: 'round' };
    if (kind === 'check') return this.path(`M${r2(x - s)} ${r2(y + s * 0.05)} L${r2(x - s * 0.3)} ${r2(y + s * 0.7)} L${r2(x + s)} ${r2(y - s * 0.65)}`, stroke);
    if (kind === 'cross') return this.path(`M${r2(x - s * 0.8)} ${r2(y - s * 0.8)} L${r2(x + s * 0.8)} ${r2(y + s * 0.8)} M${r2(x + s * 0.8)} ${r2(y - s * 0.8)} L${r2(x - s * 0.8)} ${r2(y + s * 0.8)}`, stroke);
    if (kind === 'pause') return this.path(`M${r2(x - s * 0.45)} ${r2(y - s * 0.8)} V${r2(y + s * 0.8)} M${r2(x + s * 0.45)} ${r2(y - s * 0.8)} V${r2(y + s * 0.8)}`, stroke);
    throw new Error(`unknown glyph ${kind}`);
  }

  toString({ background = true } = {}) {
    const t = this.t;
    const head =
      `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 ${this.w} ${this.h}" width="${this.w}" height="${this.h}" ` +
      `role="img" aria-labelledby="${this.id}-title ${this.id}-desc" font-family="${SANS}">`;
    const meta = `<title id="${this.id}-title">${esc(this.title)}</title><desc id="${this.id}-desc">${esc(this.desc)}</desc>`;
    const defs = this.defs.size ? `<defs>${[...this.defs.values()].join('')}</defs>` : '';
    const bg = background
      ? `<rect x="0.5" y="0.5" width="${this.w - 1}" height="${this.h - 1}" rx="14" fill="${t.bg}" stroke="${t.border}"/>`
      : '';
    return [head, meta, defs, bg, ...this.out, '</svg>', ''].join('\n');
  }
}

/** Linear map from a data interval to a pixel interval. */
export const scale = (d0, d1, p0, p1) => (v) => p0 + ((v - d0) / (d1 - d0)) * (p1 - p0);
