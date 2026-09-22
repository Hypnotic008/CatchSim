// Render the 3-D mission to MP4, frame by frame, in headless Chromium.
//
//   node viz3d/render.mjs --out figs/catch3d.mp4 [--workers 3] [--ffmpeg PATH]
//   node viz3d/render.mjs --stills 0,600,1500 --outdir /tmp/stills
//
// Each worker is its own browser rendering a contiguous chunk of frames to
// its own MP4 segment; the segments are then concatenated losslessly.
// Frames are stepped explicitly (window.renderFrame(k)), so output is
// independent of how long each frame takes to render.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]);
  return acc;
}, []));
const CHROME = args.chrome || process.env.CHROME_PATH || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const FFMPEG = args.ffmpeg || process.env.FFMPEG || 'ffmpeg';
const WORKERS = parseInt(args.workers || '3', 10);

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
               '.json': 'application/json', '.css': 'text/css', '.png': 'image/png' };
function serve(root) {
  const srv = http.createServer((req, res) => {
    const p = path.join(root, decodeURIComponent(req.url.split('?')[0]));
    if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' });
    fs.createReadStream(p).pipe(res);
  });
  return new Promise((ok) => srv.listen(0, '127.0.0.1', () => ok(srv)));
}

async function openPage(port) {
  const browser = await chromium.launch({ executablePath: CHROME,
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('page error:', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.error('console:', m.text()); });
  await page.goto(`http://127.0.0.1:${port}/index.html?render=1${args.query ? '&' + args.query : ''}`);
  await page.waitForFunction(() => window.READY === true, null, { timeout: 180000 });
  return { browser, page };
}

async function renderChunk(port, k0, k1, outFile, tag) {
  const { browser, page } = await openPage(port);
  const ff = spawn(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', '30', '-c:v', 'mjpeg',
    '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', '-r', '30', outFile],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let k = k0; k < k1; k++) {
    await page.evaluate((i) => window.renderFrame(i), k);
    const buf = await page.screenshot({ type: 'jpeg', quality: 93 });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if ((k - k0) % 100 === 0) {
      const el = (Date.now() - t0) / 1000, done = k - k0 + 1;
      console.log(`[${tag}] frame ${k} (${done}/${k1 - k0}) ${(el / done).toFixed(2)} s/frame`);
    }
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  await browser.close();
}

const srv = await serve(here);
const port = srv.address().port;

if (args.stills) {
  const { browser, page } = await openPage(port);
  const outdir = args.outdir || os.tmpdir();
  fs.mkdirSync(outdir, { recursive: true });
  for (const k of String(args.stills).split(',').map(Number)) {
    const t0 = Date.now();
    await page.evaluate((i) => window.renderFrame(i), k);
    const f = path.join(outdir, `still_${String(k).padStart(5, '0')}.png`);
    await page.screenshot({ path: f });
    console.log(f, `${Date.now() - t0} ms`);
  }
  await browser.close();
} else {
  const out = path.resolve(args.out || path.join(here, '..', 'figs', 'catch3d.mp4'));
  const probe = await openPage(port);
  const n = await probe.page.evaluate(() => window.FRAME_COUNT);
  await probe.browser.close();
  const last = Math.min(n, parseInt(args.to || n, 10)), first = parseInt(args.from || '0', 10);
  const per = Math.ceil((last - first) / WORKERS);
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'catch3d-'));
  const segs = [];
  const jobs = [];
  for (let w = 0; w < WORKERS; w++) {
    const k0 = first + w * per, k1 = Math.min(last, k0 + per);
    if (k0 >= k1) continue;
    const f = path.join(tmp, `seg${w}.mp4`);
    segs.push(f);
    jobs.push(renderChunk(port, k0, k1, f, `w${w}`));
  }
  console.log(`rendering frames ${first}..${last} with ${jobs.length} workers`);
  await Promise.all(jobs);
  const list = path.join(tmp, 'list.txt');
  fs.writeFileSync(list, segs.map((s) => `file '${s}'`).join('\n'));
  execFileSync(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy',
                        '-movflags', '+faststart', out]);
  console.log(out);
}
srv.close();
