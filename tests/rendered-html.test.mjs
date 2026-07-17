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
  assert.match(html, /Create interaction/);
  assert.match(html, /Defined interactions/);
  assert.match(html, /Export JSON/);
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
  assert.match(page, /interaction_type: type/);
  assert.match(page, /interactionDraft\?\.interaction_type \?\? interactionType/);
  assert.match(page, /object_id_list: \[\.\.\.selectedTracks\]\.sort\(naturalCompare\)/);
  assert.match(page, /start_time: annotationTime\(currentTime\)/);
  assert.match(page, /end_time: null/);
  assert.match(page, /nextInteractionId\(interactions\)/);
  assert.match(page, /return `i\$\{index\}`/);
  assert.match(page, /id="interaction-id"/);
  assert.match(page, /interaction_id: interactionDraft\.interaction_id\.trim\(\)/);
  assert.match(page, /Interaction ID \$\{savedInteraction\.interaction_id\} is already in use/);
  assert.match(page, /Use current time as end/);
  assert.match(page, /Use current time as start/);
  assert.match(page, /Jump to start/);
  assert.match(page, /saveInteraction/);
  assert.match(page, /discardInteraction/);
  assert.match(page, /seek\(interaction\.start_time\)/);
  assert.match(page, /setSelectedTracks\(new Set\(interaction\.object_id_list\)\)/);
  assert.match(page, /\.sort\(naturalCompare\)/);
  assert.match(page, /setInteractionDraft\(\{ \.\.\.interactionDraft, object_id_list: objectIds \}\)/);
  assert.doesNotMatch(page, /randomUUID|interaction-\$\{/);
  assert.match(page, /list="interaction-type-options"/);
  assert.match(page, /<datalist id="interaction-type-options">/);
  assert.match(page, /video: data\?\.video \|\| videoPath/);
  assert.match(page, /contour: contourPath/);
  assert.match(page, /interaction_list: sortedInteractions/);
  assert.match(page, /new Blob/);
  assert.match(page, /\.interactions\.json/);
  assert.match(page, /Jump to end/);
  assert.match(page, /disabled=\{!hasInteractionChanges\}/);
  assert.match(page, /setInteractionDraft\(cloneInteraction\(interaction\)\)/);
  assert.match(page, /interaction\.interaction_id === editingInteractionId \? savedInteraction : interaction/);
  assert.match(page, /window\.alert/);
  assert.match(page, /window\.confirm/);
  assert.match(page, /frame\.frame_index >= firstFrameIndex/);
  assert.match(page, /track\.contours_xy\.some\(\(contour\) => contour\.length >= 3\)/);
  assert.doesNotMatch(page, /onClick=\{createInteraction\} disabled=/);
  assert.match(page, /Saved interactions have not been exported/);
  assert.match(page, /setData\(null\)/);
  assert.match(page, /setInteractions\(\[\]\)/);
  assert.match(page, /setJsonName\("No control file selected"\)/);
  assert.match(page, /setExportedInteractionSignature\(interactionSignature\)/);
  assert.match(page, />Clear selection<\/button>/);
});
