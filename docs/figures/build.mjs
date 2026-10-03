// Build every figure: <name>-{light,dark}.svg next to this script, png/<name>-{light,dark}.png, and
// index.html. Only the SVGs are committed; the README embeds them.
//
//   node build.mjs                 all figures
//   node build.mjs detection       export PNGs for the named figures only
//   node build.mjs --no-png        skip PNG export (no Chrome needed)
//   node build.mjs --transparent   no background card
//   node build.mjs --scale=1       PNG pixel ratio (default 2)

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { FIGURES } from './src/figures.mjs';
import { Figure } from './src/lib.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const flags = process.argv.slice(2).filter((a) => a.startsWith('--'));
const names = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const option = (name) => flags.find((f) => f === `--${name}` || f.startsWith(`--${name}=`));
const pixelRatio = Number(option('scale')?.split('=')[1] ?? 2);
const out = option('out')?.split('=')[1] ?? ROOT;

const unknown = names.filter((name) => !FIGURES.some((f) => f.name === name));
if (unknown.length) throw new Error(`unknown figure: ${unknown.join(', ')}`);
const selected = new Set(FIGURES.filter((f) => !names.length || names.includes(f.name)));

mkdirSync(out, { recursive: true });
mkdirSync(join(out, 'png'), { recursive: true });

const jobs = [];
for (const figure of FIGURES) {
  for (const theme of ['light', 'dark']) {
    const fig = new Figure({
      id: `rf-${figure.name}`,
      width: figure.width,
      height: figure.height,
      theme,
      title: figure.title,
      desc: figure.alt,
    });
    figure.draw(fig);
    const svg = join(out, `${figure.name}-${theme}.svg`);
    writeFileSync(svg, fig.toString({ background: !option('transparent') }));
    // An animated SVG has no single frame worth exporting.
    if (selected.has(figure) && !figure.animated) {
      jobs.push({ svg, png: join(out, 'png', `${figure.name}-${theme}.png`), width: fig.w, height: fig.h });
    }
  }
}
console.log(`wrote ${FIGURES.length * 2} SVG files`);

// Chrome writes the screenshot and then stays open, so stop it once the PNG is complete.
const IEND = Buffer.from([0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82]);
function screenshot(job, slot) {
  return new Promise((resolve, reject) => {
    rmSync(job.png, { force: true });
    const profile = join(tmpdir(), `reactionflow-figures-chrome-${process.pid}-${slot}`);
    const child = spawn(
      CHROME,
      [
        '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
        '--no-default-browser-check', '--disable-extensions', '--disable-background-networking',
        '--disable-component-update', '--disable-sync', '--default-background-color=00000000',
        `--user-data-dir=${profile}`, `--window-size=${job.width},${job.height}`,
        `--force-device-scale-factor=${pixelRatio}`, `--screenshot=${job.png}`,
        pathToFileURL(job.svg).href,
      ],
      { stdio: 'ignore' },
    );
    const started = Date.now();
    const finish = (error) => {
      clearInterval(timer);
      child.kill();
      error ? reject(error) : resolve();
    };
    const timer = setInterval(() => {
      if (existsSync(job.png) && readFileSync(job.png).subarray(-8).equals(IEND)) finish();
      else if (Date.now() - started > 60_000) finish(new Error(`Chrome did not render ${job.png}`));
    }, 200);
    child.on('error', finish);
  });
}

if (!option('no-png')) {
  const queue = [...jobs];
  await Promise.all(
    [0, 1, 2, 3].map(async (slot) => {
      for (let job = queue.shift(); job; job = queue.shift()) await screenshot(job, slot);
    }),
  );
  console.log(`wrote ${jobs.length} PNG files at ${pixelRatio}x`);
}

// Gallery of every figure, both themes. The SVGs are embedded, so the page is a single file.
const embed = (name, theme) =>
  `data:image/svg+xml;base64,${readFileSync(join(out, `${name}-${theme}.svg`)).toString('base64')}`;
const row = (f) => `
  <section id="${f.name}">
    <h2>${f.label ?? f.title}</h2>
    <p>${f.caption}</p>
    <figure class="light"><img src="${embed(f.name, 'light')}" alt="${f.alt}"></figure>
    <figure class="dark"><img src="${embed(f.name, 'dark')}" alt="${f.alt}"></figure>
    <div class="files">${f.name}-light.svg · ${f.name}-dark.svg${f.animated ? '' : ` · png/${f.name}-light.png · png/${f.name}-dark.png`}</div>
  </section>`;
writeFileSync(
  join(out, 'index.html'),
  `<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ReactionFlow figures</title>
<style>
  :root { color-scheme: light dark; --bg: #ffffff; --ink: #1f2328; --ink2: #59636e; --line: #d1d9e0; }
  :root[data-theme="dark"] { --bg: #0d1117; --ink: #f0f6fc; --ink2: #a3adb8; --line: #3d444d; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1000px; margin: 0 auto; padding: 32px 16px 96px; }
  header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
  h1 { font-size: 28px; margin: 0; }
  h2 { font-size: 20px; margin: 56px 0 4px; }
  p { margin: 0 0 16px; color: var(--ink2); max-width: 76ch; }
  figure { margin: 0; }
  img { display: block; width: 100%; height: auto; }
  code, .files { font: 0.86em ui-monospace, SFMono-Regular, Menlo, monospace; }
  .files { margin-top: 10px; color: var(--ink2); }
  button { font: inherit; color: var(--ink); background: none; border: 1px solid var(--line); border-radius: 8px; padding: 6px 14px; cursor: pointer; }
  .dark { display: none; }
  :root[data-theme="dark"] .dark { display: block; }
  :root[data-theme="dark"] .light { display: none; }
</style>
<main>
  <header>
    <h1>ReactionFlow figures</h1>
    <button type="button" onclick="document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'">Light / dark</button>
  </header>
  <p>Generated by <code>node build.mjs</code>. Each figure has a light and a dark SVG and matching PNGs; the two animated ones are SVG only.</p>
${FIGURES.map(row).join('\n')}
</main>
</html>
`,
);
console.log('wrote index.html');
