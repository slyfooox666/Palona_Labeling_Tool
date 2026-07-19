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
  assert.match(html, /Depth cues JSON/);
  assert.match(html, /Annotation project/);
  assert.match(html, /Contours/);
  assert.match(html, /Relative depth/);
  assert.match(html, /Draw ROI/);
  assert.match(html, /Blackout outside/);
  assert.match(html, /Create interaction/);
  assert.match(html, /Defined interactions/);
  assert.match(html, /Save project/);
  assert.match(html, /Export minimal/);
  assert.match(html, /ID aliases/);
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
  assert.match(page, /controlWorkerRef\.current\?\.terminate\(\)/);
  assert.match(page, /sourceLoadGenerationRef\.current !== loadGeneration/);
  assert.match(page, /depthLoadGenerationRef\.current !== depthLoadGeneration/);
  assert.match(page, /frame\.source_frame_index \?\? frame\.frame_index/);
  assert.match(page, /frameIndex \/ declaredSourceFps/);
  assert.doesNotMatch(page, /frameIndex \/ 30/);
  assert.match(page, /has no timestamp_seconds and the Control JSON has no trusted source FPS/);
  assert.match(page, /const resolvedSourceFps = declaredSourceFps \?\? inferredSourceFps/);
  assert.match(page, /provide source FPS or increasing frame\/time pairs/);
  assert.match(page, /changes label from/);
  assert.match(page, /video duration mismatch/);
  assert.match(page, /video frame-count mismatch/);
  assert.ok((page.match(/expectedSourceFrameCount\(/gu) ?? []).length >= 3);
  assert.match(page, /track\.contours_xy/);
  assert.match(page, /const \[visibleTrackIds, setVisibleTrackIds\]/);
  assert.match(page, /const \[interactionTrackIds, setInteractionTrackIds\]/);
  assert.match(page, /for \(const track of visibleTracks\)/);
  assert.match(page, /hoveredContour\?\.frame_index === currentFrame\?\.frame_index/);
  assert.match(page, /track\.track_id === activeHoveredContour\?\.id/);
  assert.match(page, /\{activeHoveredContour\.id\} \{activeHoveredContour\.label\}/);
  assert.match(page, /catalog\.tracks\.filter\(\(track\) => track\.label === type\)/);
  assert.match(page, /trackIds\.every\(\(id\) => visibleTrackIds\.has\(id\)\)/);
  assert.doesNotMatch(page, /selectedTypes|setSelectedTypes/);
  assert.doesNotMatch(page, /selectedTracks|setSelectedTracks/);
  assert.match(page, /interaction_type: type/);
  assert.match(page, /interactionDraft\?\.interaction_type \?\? interactionType/);
  assert.match(page, /object_id_list: \[\.\.\.interactionTrackIds\]\.sort\(naturalCompare\)/);
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
  assert.match(page, /deleteInteraction/);
  assert.match(page, /confirmDiscardInteractionChanges/);
  assert.match(page, /Save or discard the current interaction draft before exporting/);
  assert.match(page, /Select at least one person participant/);
  assert.match(page, /Table interactions must include a table track/);
  assert.match(page, /seek\(interaction\.start_time\)/);
  assert.match(page, /setInteractionTrackIds\(new Set\(interaction\.object_id_list\)\)/);
  assert.match(page, /\.sort\(naturalCompare\)/);
  assert.doesNotMatch(page, /randomUUID|interaction-\$\{/);
  assert.match(page, /list="interaction-type-options"/);
  assert.match(page, /<datalist id="interaction-type-options">/);
  assert.match(page, /serializeAnnotationProject/);
  assert.match(page, /parseAnnotationProjectJson/);
  assert.match(page, /serializeMinimalInteractions/);
  assert.match(page, /localStorage\.setItem/);
  assert.match(page, /autosaveKey\(videoName, jsonName, videoFileSize, contourFileSize\)/);
  assert.match(page, /validateNormalizedRoiPolygon/);
  assert.match(page, /withIdAlias/);
  assert.match(page, /autosaveSuspendedRef/);
  assert.doesNotMatch(page, /interaction_list: sortedInteractions/);
  assert.match(page, /new Blob/);
  assert.match(page, /\.events-minimal\.json/);
  assert.match(page, /\.palona-project\.json/);
  assert.match(page, /Jump to end/);
  assert.match(page, /disabled=\{!hasInteractionChanges\}/);
  assert.match(page, /setInteractionDraft\(cloneInteraction\(interaction\)\)/);
  assert.match(page, /interaction\.interaction_id === editingInteractionId \? savedInteraction : interaction/);
  assert.match(page, /window\.alert/);
  assert.match(page, /window\.confirm/);
  assert.match(page, /frame\.frame_index >= firstFrameIndex/);
  assert.match(page, /track\.contours_xy\.some\(\(contour\) => contour\.length >= 3\)/);
  assert.match(page, /if \(!canvas \|\| !showMasks\) return null/);
  assert.match(page, /roiComplete && roiBlackout && !pointInPolygon\(normalizedPoint, roiPolygon\)/);
  assert.ok(
    page.indexOf("// Blackout is composited after every video cue")
      > page.indexOf("for (const track of renderedTracks)"),
    "ROI blackout must be composited after masks/labels so no cue leaks outside the ROI",
  );
  assert.doesNotMatch(page, /onClick=\{createInteraction\} disabled=/);
  assert.match(page, /Saved interactions have not been exported/);
  assert.ok((page.match(/setExportedInteractionSignature\(""\)/gu) ?? []).length >= 2);
  assert.match(page, /setData\(null\)/);
  assert.match(page, /setInteractions\(\[\]\)/);
  assert.match(page, /setJsonName\("No control file selected"\)/);
  assert.match(page, /setExportedInteractionSignature\(interactionSignature\)/);
  assert.match(page, />Clear selection<\/button>/);
  assert.match(page, /parseDepthFeatures/);
  assert.match(page, /findAlignedDepthFrame/);
  assert.match(page, /findAlignedControlFrame/);
  assert.match(page, /raw\.media\?\.sample_fps/);
  assert.match(page, /Outside Control coverage/);
  assert.match(page, /max_alignment_error_seconds/);
  assert.match(page, /video dimensions mismatch/);
  assert.match(page, /frame\.instances/);
  assert.match(page, /DepthEvidencePanel/);
});

test("ships a versioned non-metric depth cue contract", async () => {
  const parser = await readFile(new URL("../app/lib/depth-features.ts", import.meta.url), "utf8");
  const schema = JSON.parse(await readFile(new URL("../docs/depth-features.schema.json", import.meta.url), "utf8"));

  assert.equal(schema.properties.schema_version.const, "palona.depth-features/v1");
  assert.equal(schema.properties.depth_metadata.properties.metric.const, false);
  assert.equal(
    schema.properties.depth_metadata.properties.depth_semantics.const,
    "depth_rank: 0=near, 1=far",
  );
  assert.match(parser, /DEPTH_FEATURE_SCHEMA = "palona\.depth-features\/v1"/);
  assert.match(parser, /max_cue_age_seconds/);
  assert.match(parser, /delta > tolerance/);
  assert.match(parser, /frames must be strictly ordered/);
  assert.match(parser, /normalization\.aligned_high must exceed aligned_low/);
  assert.match(parser, /per_frame_robust_affine_to_clip_anchor/);
});

test("ships versioned project and exact minimal-export contracts", async () => {
  const projectSchema = JSON.parse(await readFile(
    new URL("../docs/annotation-project.schema.json", import.meta.url),
    "utf8",
  ));
  const projectModel = await readFile(new URL("../app/lib/labeling-project.ts", import.meta.url), "utf8");

  assert.equal(projectSchema.properties.schema_version.const, "palona.annotation-project/v1");
  assert.equal(projectSchema.properties.roi.properties.polygon.oneOf[1].minItems, 3);
  assert.equal(projectSchema.properties.source.properties.video_file_size_bytes.minimum, 1);
  assert.equal(projectSchema.properties.source.properties.contour_file_size_bytes.minimum, 1);
  assert.match(projectModel, /exportMinimalInteractions/);
  assert.match(projectModel, /person_id_list/);
  assert.match(projectModel, /table_id/);
  assert.match(projectModel, /alias cycle detected/);
  assert.match(projectModel, /self-intersection/);
});
