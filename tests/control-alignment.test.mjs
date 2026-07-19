import assert from "node:assert/strict";
import test from "node:test";

import {
  controlMaxCueAgeSeconds,
  findAlignedControlFrame,
  inferControlSampleFps,
} from "../app/lib/control-alignment.ts";

const frames = Array.from({ length: 8 }, (_, index) => ({
  frame_index: index * 20,
  timestamp_seconds: index,
}));

test("aligns Control frames within the sampled cue age", () => {
  assert.equal(findAlignedControlFrame([], 0, { sourceFps: 20, sampleFps: 1 }), null);
  assert.equal(findAlignedControlFrame(frames, 3, { sourceFps: 20, sampleFps: 1 })?.frame_index, 60);
  assert.equal(findAlignedControlFrame(frames, 3.49, { sourceFps: 20, sampleFps: 1 })?.frame_index, 60);
  assert.equal(findAlignedControlFrame(frames, 3.51, { sourceFps: 20, sampleFps: 1 })?.frame_index, 80);
  assert.equal(findAlignedControlFrame(frames, 7.5251, { sourceFps: 20, sampleFps: 1 })?.frame_index, 140);
  assert.equal(findAlignedControlFrame(frames, 7.5252, { sourceFps: 20, sampleFps: 1 }), null);
  assert.equal(findAlignedControlFrame(frames, -0.5252, { sourceFps: 20, sampleFps: 1 }), null);
});

test("does not freeze a Control frame across a temporal gap", () => {
  const sparse = [
    { timestamp_seconds: 0 },
    { timestamp_seconds: 1 },
    { timestamp_seconds: 10 },
  ];
  assert.equal(findAlignedControlFrame(sparse, 5, { sourceFps: 20, sampleFps: 1 }), null);
});

test("infers median cadence and keeps a single frame bounded", () => {
  assert.equal(inferControlSampleFps(frames), 1);
  assert.equal(controlMaxCueAgeSeconds(frames, { sourceFps: 20 }), 0.5251);
  const single = [{ timestamp_seconds: 2 }];
  assert.equal(findAlignedControlFrame(single, 2.0251, { sourceFps: 20 })?.timestamp_seconds, 2);
  assert.equal(findAlignedControlFrame(single, 2.0252, { sourceFps: 20 }), null);
});
