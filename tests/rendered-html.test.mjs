import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the contour labeling workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Palona Contour Lab<\/title>/i);
  assert.match(html, /Palona contour lab/);
  assert.match(html, /Video clip/);
  assert.match(html, /Control JSON/);
  assert.match(html, /Contours/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/);
});

test("includes frame sync, contour hit testing, and local file support", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /requestVideoFrameCallback/);
  assert.match(page, /function pointInPolygon/);
  assert.match(page, /stepFrame\(-1\)/);
  assert.match(page, /stepFrame\(1\)/);
  assert.match(page, /accept="video\/\*,\.mkv"/);
  assert.match(page, /new Worker/);
  assert.match(page, /track\.contours_xy/);
  assert.match(page, /for \(const track of currentFrame\?\.tracks \?\? \[\]\)/);
  assert.match(page, /track\.track_id === hoveredContour\?\.id/);
  assert.match(page, /\{hoveredContour\.id\} \{hoveredContour\.label\}/);
  assert.match(page, /catalog\.tracks\.filter\(\(track\) => track\.label === type\)/);
  assert.match(page, /trackIds\.every\(\(id\) => selectedTracks\.has\(id\)\)/);
  assert.doesNotMatch(page, /selectedTypes|setSelectedTypes/);
});
