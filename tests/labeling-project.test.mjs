import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  ANNOTATION_PROJECT_SCHEMA,
  AnnotationProjectValidationError,
  canonicalizeInteraction,
  exportMinimalInteractions,
  frameToTime,
  normalizedToPixel,
  parseAnnotationProject,
  parseAnnotationProjectJson,
  pixelToNormalized,
  resolveTrackId,
  serializeAnnotationProject,
  serializeMinimalInteractions,
  timeToFrame,
  validateIdAliases,
  validateNormalizedRoiPolygon,
  withIdAlias,
  withoutIdAlias,
} from "../app/lib/labeling-project.ts";

function validProject() {
  return {
    schema_version: ANNOTATION_PROJECT_SCHEMA,
    source: {
      video_path: "source/synthetic.mp4",
      contour_path: "source/synthetic.contours.json",
      video_file_size_bytes: 123456,
      contour_file_size_bytes: 6543,
      video_fps: 30,
      video_width: 640,
      video_height: 360,
      frame_count: 300,
      duration_seconds: 10,
      available_categories: ["person", "table"],
    },
    interactions: [
      {
        interaction_id: "i0",
        interaction_type: "occupy_table",
        object_id_list: ["p0:27", "p0:12", "p1:0"],
        start_time: 0,
        end_time: 3,
        start_frame: 0,
        end_frame: 90,
        notes: "synthetic integration fixture",
      },
    ],
    roi: {
      polygon: [[0.1, 0.12], [0.82, 0.1], [0.9, 0.85], [0.08, 0.88]],
      blackout_enabled: true,
    },
    id_aliases: { "p0:27": "p0:11" },
    created_at: "2026-07-16T01:00:00.000Z",
    updated_at: "2026-07-16T01:01:00.000Z",
  };
}

test("strictly parses the versioned project contract", () => {
  const parsed = parseAnnotationProject(validProject());
  assert.equal(parsed.schema_version, "palona.annotation-project/v1");
  assert.equal(parsed.source.video_fps, 30);
  assert.equal(parsed.source.video_file_size_bytes, 123456);
  assert.equal(parsed.interactions[0].interaction_id, "i0");
  assert.deepEqual(parsed.roi.polygon?.[0], [0.1, 0.12]);

  const wrongVersion = structuredClone(validProject());
  wrongVersion.schema_version = "palona.annotation-project/v2";
  assert.throws(() => parseAnnotationProject(wrongVersion), /schema_version/);

  const unknownField = { ...validProject(), surprise: true };
  assert.throws(() => parseAnnotationProject(unknownField), /surprise: is not a supported field/);

  const duplicateEventId = structuredClone(validProject());
  duplicateEventId.interactions.push({ ...duplicateEventId.interactions[0] });
  assert.throws(() => parseAnnotationProject(duplicateEventId), /interaction_id: must be unique/);

  const outsideDuration = structuredClone(validProject());
  outsideDuration.interactions[0].end_time = 10.1;
  assert.throws(() => parseAnnotationProject(outsideDuration), /must not exceed the source duration/);

  const invalidSourceSize = structuredClone(validProject());
  invalidSourceSize.source.video_file_size_bytes = 0;
  assert.throws(() => parseAnnotationProject(invalidSourceSize), /video_file_size_bytes/);

  assert.throws(
    () => parseAnnotationProjectJson("{not json}"),
    (error) => error instanceof AnnotationProjectValidationError && /invalid JSON/.test(error.message),
  );
});

test("validates normalized ROI polygons and rejects unsafe geometry", () => {
  assert.deepEqual(
    validateNormalizedRoiPolygon([[0, 0], [1, 0], [1, 1], [0, 1]]),
    [[0, 0], [1, 0], [1, 1], [0, 1]],
  );
  assert.throws(() => validateNormalizedRoiPolygon([[0, 0], [1, 0]]), /at least three/);
  assert.throws(
    () => validateNormalizedRoiPolygon([[0, 0], [1.01, 0], [0, 1]]),
    /normalized range/,
  );
  assert.throws(
    () => validateNormalizedRoiPolygon([[0, 0], [1, 1], [0, 1], [1, 0]]),
    /non-zero area|self-intersection/,
  );
  assert.throws(
    () => validateNormalizedRoiPolygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
    /duplicate/,
  );
  assert.throws(
    () => validateNormalizedRoiPolygon([[0, 0], [1, 0], [0.5, 0], [1, 1], [0, 1]]),
    /overlapping adjacent edges/,
  );

  const projectWithoutRoi = validProject();
  projectWithoutRoi.roi = { polygon: null, blackout_enabled: true };
  assert.throws(() => parseAnnotationProject(projectWithoutRoi), /cannot be enabled without an ROI polygon/);
});

test("converts normalized coordinates and frame/time values deterministically", () => {
  assert.deepEqual(normalizedToPixel([0.25, 0.5], 640, 360), [160, 180]);
  assert.deepEqual(pixelToNormalized([160, 180], 640, 360), [0.25, 0.5]);
  assert.throws(() => pixelToNormalized([641, 180], 640, 360), /normalized range/);

  assert.equal(frameToTime(90, 30), 3);
  assert.equal(timeToFrame(3, 30), 90);
  assert.equal(timeToFrame(1.51, 10, "floor"), 15);
  assert.equal(timeToFrame(1.51, 10, "nearest"), 15);
  assert.equal(timeToFrame(1.51, 10, "ceil"), 16);
  assert.throws(() => frameToTime(-1, 30), /non-negative/);
  assert.throws(() => timeToFrame(1, 0), /greater than zero/);
});

test("resolves aliases transitively, canonicalizes participants, and rejects cycles", () => {
  const aliases = validateIdAliases({
    "p0:31": "p0:27",
    "p0:27": "p0:11",
  });
  assert.equal(resolveTrackId("p0:31", aliases), "p0:11");
  assert.equal(resolveTrackId("p0:12", aliases), "p0:12");

  const canonical = canonicalizeInteraction({
    interaction_id: "i0",
    interaction_type: "occupy_table",
    object_id_list: ["p0:31", "p0:11", "p1:0"],
    start_time: 0,
    end_time: 3,
  }, aliases);
  assert.deepEqual(canonical.object_id_list, ["p0:11", "p1:0"]);

  const added = withIdAlias({}, "p0:31", "p0:27");
  assert.equal(added["p0:31"], "p0:27");
  assert.deepEqual({ ...withoutIdAlias(added, "p0:31") }, {});

  assert.throws(
    () => validateIdAliases({ "p0:11": "p0:27", "p0:27": "p0:11" }),
    /alias cycle/,
  );
  assert.throws(
    () => withIdAlias({ "p0:27": "p0:11" }, "p0:11", "p0:27"),
    /alias cycle/,
  );
});

test("exports exactly the Agents.MD minimal event fields from canonical track labels", () => {
  const project = parseAnnotationProject(validProject());
  const trackLabels = {
    "p0:11": "person",
    "p0:12": "customer",
    "p0:27": "person",
    "p1:0": "dining_table",
  };
  const exported = exportMinimalInteractions(project.interactions, trackLabels, project.id_aliases);

  assert.deepEqual(exported, [{
    event: "occupy_table",
    person_id_list: ["p0:11", "p0:12"],
    table_id: "p1:0",
    start_time: 0,
    end_time: 3,
  }]);
  assert.deepEqual(Object.keys(exported[0]), [
    "event", "person_id_list", "table_id", "start_time", "end_time",
  ]);
  assert.equal(
    serializeMinimalInteractions(project.interactions, trackLabels, project.id_aliases),
    `${JSON.stringify(exported, null, 2)}\n`,
  );

  const noTable = { ...project.interactions[0], object_id_list: ["p0:12"] };
  assert.throws(() => exportMinimalInteractions([noTable], trackLabels), /exactly one table/);

  const machineIsNotAPerson = {
    ...project.interactions[0],
    object_id_list: ["cashier-machine", "p1:0"],
  };
  assert.throws(
    () => exportMinimalInteractions([machineIsNotAPerson], {
      "cashier-machine": "cashier machine",
      "p1:0": "table",
    }),
    /unsupported label/,
  );
});

test("saves, reopens, and exports a synthetic project without data loss", () => {
  const saved = serializeAnnotationProject(validProject());
  assert.match(saved, /"schema_version": "palona.annotation-project\/v1"/);

  const reopened = parseAnnotationProjectJson(saved);
  assert.deepEqual(reopened, parseAnnotationProject(validProject()));
  const minimal = exportMinimalInteractions(reopened.interactions, [
    { id: "p0:11", label: "person" },
    { id: "p0:12", label: "person" },
    { id: "p0:27", label: "person" },
    { id: "p1:0", label: "table" },
  ], reopened.id_aliases);
  assert.deepEqual(minimal[0], {
    event: "occupy_table",
    person_id_list: ["p0:11", "p0:12"],
    table_id: "p1:0",
    start_time: 0,
    end_time: 3,
  });
});

test("the checked-in synthetic example follows the same strict project contract", async () => {
  const text = await readFile(new URL("../examples/synthetic-project.json", import.meta.url), "utf8");
  const project = parseAnnotationProjectJson(text);
  assert.equal(project.source.video_path, "source/synthetic.mp4");
  assert.equal(project.roi.polygon?.length, 4);
  assert.equal(project.interactions[0].interaction_type, "occupy_table");
});
