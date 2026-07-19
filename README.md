# Palona Labeling Tool — Depth Prior

`Depth_Prior` adds estimated relative depth to the original contour-labeling tool. Interaction labels are still created by a person from SAM3 track IDs and time ranges. Depth is additional spatial-temporal evidence and is kept separate from the ground-truth labels.

## Changes from the original version

### Depth prior

The new `depth_pipeline/` package samples the source video, calls a compatible DA3 runtime, aligns the resulting depth maps with SAM3 Control frames, and writes a `palona.depth-features/v1` sidecar.

The UI can show:

- relative depth rank for each tracked instance
- person–object relative depth gap
- 2D mask gap and centroid distance
- `approaching`, `stable`, or `leaving` trends
- proximity duration and start/end boundary scores
- feature quality

Depth rank is normalized to `[0, 1]`: `0` is nearer to the camera and `1` is farther away. It is not distance in meters.

### Full-video Control generation

`scripts/full-video-preprocess.sh` wraps the existing `vision_pipeline` SAM3 command. It uses native whole-video tracking with configurable chunks and overlap; SAM3 inference and cross-chunk ID stitching remain in `vision_pipeline`.

The wrapper checks the CUDA environment, video dependencies, continuous temporal coverage, timestamps, contour bounds, and optional required labels. It writes the final Control JSON atomically after validation. Full-video runs do not use `--max-frames`.

### Control/video synchronization

The original UI always displayed the nearest Control frame. If a short Control file ended early, its final mask stayed on screen while the video continued.

This branch reads `source_frame_index`, `source_fps`, and `sample_fps` when available. A Control frame is used only inside a bounded time window derived from the source and sampling rates. Outside that window, the mask, hover state, and click target are removed together.

### Annotation workflow

The branch adds:

- save/reopen through `palona.annotation-project/v1` project files
- browser-local autosave for a matching video/Control pair
- editable interaction IDs and duplicate-ID checks
- polygon ROI drawing and blackout outside the ROI
- CLI export of an ROI-masked video and filtered Control JSON
- manual track-ID aliases for ID switches
- separate overlay visibility and event-participant selection
- frame stepping, playback speed, overlay controls, and an event timeline

The minimal training export remains a top-level array with exactly these fields:

```json
[
  {
    "event": "occupy_table",
    "person_id_list": ["p0:11", "p0:12"],
    "table_id": "p1:0",
    "start_time": 0.0,
    "end_time": 3.0
  }
]
```

The exporter uses Control labels, not ID prefixes, to identify people and tables. Track aliases are resolved before export. Depth features remain in their own sidecar.

## How it works

```text
video -> SAM3 -> Control JSON --------------------┐
video -> frame sampling -> DA3 depth maps --------+-> depth feature builder -> sidecar
video + Control JSON + optional sidecar ----------+-> labeling UI -> project / event JSON
```

### SAM3: identity and geometry

Each Control frame contains a frame index, timestamp, and tracked objects. An object has a `track_id` and label, and may include confidence and absolute-pixel contours.

The browser accepts both the native `vision_pipeline` format (`tracks`) and shared-runtime manifests (`instances`). A Web Worker normalizes either format before it enters the UI state.

### DA3: relative depth

DA3 predicts a relative depth map for each sampled image. Raw maps can have different scale and offset across frames, so the pipeline aligns them to a clip reference using untracked background pixels and then applies robust clip-level normalization.

For each object mask, the pipeline computes its median depth rank, spread, valid-pixel ratio, temporal velocity, and quality. For each configured person–target pair, it combines depth difference, 2D geometry, visibility, and time history to produce proximity, trend, duration, and boundary cues.

These cues describe whether two tracked objects are spatially close and how that relation changes over time. They do not determine the interaction type.

### Time alignment

The UI uses the nearest Control or Depth frame only when its timestamp falls inside the recorded cue window. Missing or partial model output therefore appears as missing evidence instead of a frozen overlay.

### Human labels remain independent

The annotator selects the participants, event type, start time, and end time. Those values are stored in the project and minimal event files. SAM3 masks and depth features can be regenerated without changing the saved labels.

Relative depth is not metric 3D distance, and SAM3 IDs can still switch after occlusion or between chunks. ID aliases provide a manual correction path. The current minimal exporter is defined for person–table events; other target types require their own downstream export schema.
