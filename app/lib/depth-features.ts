export const DEPTH_FEATURE_SCHEMA = "palona.depth-features/v1" as const;
export const DEPTH_RANK_SEMANTICS = "depth_rank: 0=near, 1=far" as const;

export type DepthInstanceFeature = {
  track_id: string;
  depth_rank: number;
  depth_iqr: number;
  valid_depth_ratio: number;
  depth_velocity: number;
  feature_quality: number;
};

export type DepthPairFeature = {
  source_id: string;
  target_id: string;
  depth_gap_abs: number;
  source_depth_rank: number;
  target_local_depth_rank: number;
  mask_gap_2d_norm: number;
  centroid_distance_2d_norm: number;
  target_visible_ratio: number;
  relative_depth_velocity: number;
  proximity_score: number;
  proximity_duration_seconds: number;
  trend: "approaching" | "stable" | "leaving";
  start_candidate_score: number;
  end_candidate_score: number;
  feature_quality: number;
};

export type DepthFeatureFrame = {
  frame_index: number;
  timestamp_seconds: number;
  instances: DepthInstanceFeature[];
  pairs: DepthPairFeature[];
};

export type DepthFeatureData = {
  schema_version: typeof DEPTH_FEATURE_SCHEMA;
  video: string;
  contour?: string;
  source: {
    video_width: number;
    video_height: number;
    video_fps: number;
    video_duration_seconds: number;
    video_frame_count: number | null;
    video_file_size_bytes?: number;
    contour_file_size_bytes?: number;
  };
  depth_metadata: {
    model: string;
    model_revision: string;
    metric: false;
    metric_units: null;
    raw_depth_direction: "larger_is_farther";
    depth_semantics: typeof DEPTH_RANK_SEMANTICS;
    inference_mode: "independent_depth_image";
    temporal_alignment: {
      method: "per_frame_robust_affine_to_clip_anchor";
      anchor_region: "common_untracked_background_with_full_frame_fallback";
      lower_quantile: 0.25;
      center_quantile: 0.5;
      upper_quantile: 0.75;
      canonical_median: number;
      canonical_iqr: number;
      frame_transforms: Array<{
        frame_index: number;
        timestamp_seconds: number;
        scale: number;
        shift: number;
        anchor_median_raw: number;
        anchor_iqr_raw: number;
        anchor_pixel_count_sampled: number;
        used_full_frame_fallback: boolean;
        stability_quality: number;
      }>;
    };
    normalization: {
      method: "clip_robust_quantile";
      low_quantile: number;
      high_quantile: number;
      aligned_low: number;
      aligned_high: number;
    };
    sample_fps: number;
    max_alignment_error_seconds: number;
    max_cue_age_seconds: number;
    max_temporal_gap_seconds: number;
  };
  frames: DepthFeatureFrame[];
  boundary_candidates: DepthBoundaryCandidate[];
};

export type DepthBoundaryCandidate = {
  candidate_id: string;
  source_id: string;
  target_id: string;
  start_time: number;
  end_time: number | null;
  peak_score: number;
  quality: number;
};

export type DepthFrameAlignment = {
  frame: DepthFeatureFrame;
  delta_seconds: number;
  exact_frame: boolean;
};

type JsonObject = Record<string, unknown>;

function objectAt(value: unknown, path: string): JsonObject {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} must be an object.`);
  }
  return value as JsonObject;
}

function arrayAt(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path} must be an array.`);
  return value;
}

function stringAt(value: unknown, path: string) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${path} must be a non-empty string.`);
  return value;
}

function numberAt(value: unknown, path: string, minimum?: number, maximum?: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be a finite number.`);
  if (minimum !== undefined && value < minimum) throw new Error(`${path} must be at least ${minimum}.`);
  if (maximum !== undefined && value > maximum) throw new Error(`${path} must be at most ${maximum}.`);
  return value;
}

function integerAt(value: unknown, path: string, minimum?: number) {
  const result = numberAt(value, path, minimum);
  if (!Number.isInteger(result)) throw new Error(`${path} must be an integer.`);
  return result;
}

export function depthPairKey(sourceId: string, targetId: string) {
  return `${sourceId}\u0000${targetId}`;
}

export function splitDepthPairKey(key: string): [string, string] {
  const separator = key.indexOf("\u0000");
  return separator < 0 ? [key, ""] : [key.slice(0, separator), key.slice(separator + 1)];
}

export function parseDepthFeatures(value: unknown): DepthFeatureData {
  const root = objectAt(value, "Depth features");
  if (root.schema_version !== DEPTH_FEATURE_SCHEMA) {
    throw new Error(`schema_version must be exactly ${DEPTH_FEATURE_SCHEMA}.`);
  }

  const metadata = objectAt(root.depth_metadata, "depth_metadata");
  const source = objectAt(root.source, "source");
  if (metadata.metric !== false) throw new Error("depth_metadata.metric must be false for estimated relative depth.");
  if (metadata.metric_units !== null) throw new Error("depth_metadata.metric_units must be null.");
  if (metadata.raw_depth_direction !== "larger_is_farther") {
    throw new Error("depth_metadata.raw_depth_direction must be larger_is_farther.");
  }
  if (metadata.inference_mode !== "independent_depth_image") {
    throw new Error("depth_metadata.inference_mode must be independent_depth_image.");
  }
  if (metadata.depth_semantics !== DEPTH_RANK_SEMANTICS) {
    throw new Error(`depth_metadata.depth_semantics must be exactly \"${DEPTH_RANK_SEMANTICS}\".`);
  }

  const frames = arrayAt(root.frames, "frames").map((rawFrame, framePosition) => {
    const path = `frames[${framePosition}]`;
    const frame = objectAt(rawFrame, path);
    const frameIndex = integerAt(frame.frame_index, `${path}.frame_index`, 0);

    const instanceIds = new Set<string>();
    const instances = arrayAt(frame.instances, `${path}.instances`).map((rawInstance, instancePosition) => {
      const instancePath = `${path}.instances[${instancePosition}]`;
      const instance = objectAt(rawInstance, instancePath);
      const trackId = stringAt(instance.track_id, `${instancePath}.track_id`);
      if (instanceIds.has(trackId)) throw new Error(`${path} contains duplicate instance ${trackId}.`);
      instanceIds.add(trackId);
      return {
        track_id: trackId,
        depth_rank: numberAt(instance.depth_rank, `${instancePath}.depth_rank`, 0, 1),
        depth_iqr: numberAt(instance.depth_iqr, `${instancePath}.depth_iqr`, 0),
        valid_depth_ratio: numberAt(instance.valid_depth_ratio, `${instancePath}.valid_depth_ratio`, 0, 1),
        depth_velocity: numberAt(instance.depth_velocity, `${instancePath}.depth_velocity`),
        feature_quality: numberAt(instance.feature_quality, `${instancePath}.feature_quality`, 0, 1),
      };
    });

    const pairIds = new Set<string>();
    const pairs = arrayAt(frame.pairs, `${path}.pairs`).map((rawPair, pairPosition) => {
      const pairPath = `${path}.pairs[${pairPosition}]`;
      const pair = objectAt(rawPair, pairPath);
      const sourceId = stringAt(pair.source_id, `${pairPath}.source_id`);
      const targetId = stringAt(pair.target_id, `${pairPath}.target_id`);
      if (sourceId === targetId) throw new Error(`${pairPath} must reference two different tracks.`);
      if (!instanceIds.has(sourceId) || !instanceIds.has(targetId)) {
        throw new Error(`${pairPath} must reference instances present in the same frame.`);
      }
      const key = depthPairKey(sourceId, targetId);
      if (pairIds.has(key)) throw new Error(`${path} contains duplicate pair ${sourceId} → ${targetId}.`);
      pairIds.add(key);
      const trend: DepthPairFeature["trend"] = pair.trend === "approaching"
        || pair.trend === "stable"
        || pair.trend === "leaving"
        ? pair.trend
        : (() => { throw new Error(`${pairPath}.trend must be approaching, stable, or leaving.`); })();
      return {
        source_id: sourceId,
        target_id: targetId,
        depth_gap_abs: numberAt(pair.depth_gap_abs, `${pairPath}.depth_gap_abs`, 0, 1),
        source_depth_rank: numberAt(pair.source_depth_rank, `${pairPath}.source_depth_rank`, 0, 1),
        target_local_depth_rank: numberAt(pair.target_local_depth_rank, `${pairPath}.target_local_depth_rank`, 0, 1),
        mask_gap_2d_norm: numberAt(pair.mask_gap_2d_norm, `${pairPath}.mask_gap_2d_norm`, 0),
        centroid_distance_2d_norm: numberAt(pair.centroid_distance_2d_norm, `${pairPath}.centroid_distance_2d_norm`, 0),
        target_visible_ratio: numberAt(pair.target_visible_ratio, `${pairPath}.target_visible_ratio`, 0, 1),
        relative_depth_velocity: numberAt(pair.relative_depth_velocity, `${pairPath}.relative_depth_velocity`),
        proximity_score: numberAt(pair.proximity_score, `${pairPath}.proximity_score`, 0, 1),
        proximity_duration_seconds: numberAt(pair.proximity_duration_seconds, `${pairPath}.proximity_duration_seconds`, 0),
        trend,
        start_candidate_score: numberAt(pair.start_candidate_score, `${pairPath}.start_candidate_score`, 0, 1),
        end_candidate_score: numberAt(pair.end_candidate_score, `${pairPath}.end_candidate_score`, 0, 1),
        feature_quality: numberAt(pair.feature_quality, `${pairPath}.feature_quality`, 0, 1),
      };
    });

    return {
      frame_index: frameIndex,
      timestamp_seconds: numberAt(frame.timestamp_seconds, `${path}.timestamp_seconds`, 0),
      instances,
      pairs,
    };
  });

  for (let index = 1; index < frames.length; index += 1) {
    if (frames[index].frame_index <= frames[index - 1].frame_index) {
      throw new Error("frames must be strictly ordered by frame_index without duplicates.");
    }
    if (frames[index].timestamp_seconds < frames[index - 1].timestamp_seconds) {
      throw new Error("frames must be ordered by timestamp_seconds.");
    }
  }

  const candidateIds = new Set<string>();
  const boundaryCandidates = arrayAt(root.boundary_candidates, "boundary_candidates").map((rawCandidate, index) => {
    const path = `boundary_candidates[${index}]`;
    const candidate = objectAt(rawCandidate, path);
    const candidateId = stringAt(candidate.candidate_id, `${path}.candidate_id`);
    if (candidateIds.has(candidateId)) throw new Error(`boundary_candidates contains duplicate candidate_id ${candidateId}.`);
    candidateIds.add(candidateId);
    const startTime = numberAt(candidate.start_time, `${path}.start_time`, 0);
    const endTime = candidate.end_time === null ? null : numberAt(candidate.end_time, `${path}.end_time`, 0);
    if (endTime !== null && endTime <= startTime) throw new Error(`${path}.end_time must be greater than start_time.`);
    return {
      candidate_id: candidateId,
      source_id: stringAt(candidate.source_id, `${path}.source_id`),
      target_id: stringAt(candidate.target_id, `${path}.target_id`),
      start_time: startTime,
      end_time: endTime,
      peak_score: numberAt(candidate.peak_score, `${path}.peak_score`, 0, 1),
      quality: numberAt(candidate.quality, `${path}.quality`, 0, 1),
    };
  });

  const temporalAlignmentValue = objectAt(metadata.temporal_alignment, "depth_metadata.temporal_alignment");
  if (temporalAlignmentValue.method !== "per_frame_robust_affine_to_clip_anchor") {
    throw new Error("depth_metadata.temporal_alignment.method is not supported.");
  }
  if (temporalAlignmentValue.anchor_region !== "common_untracked_background_with_full_frame_fallback") {
    throw new Error("depth_metadata.temporal_alignment.anchor_region is not supported.");
  }
  if (temporalAlignmentValue.lower_quantile !== 0.25
    || temporalAlignmentValue.center_quantile !== 0.5
    || temporalAlignmentValue.upper_quantile !== 0.75) {
    throw new Error("depth_metadata.temporal_alignment quantiles must be 0.25 / 0.5 / 0.75.");
  }
  const frameTransforms = arrayAt(
    temporalAlignmentValue.frame_transforms,
    "depth_metadata.temporal_alignment.frame_transforms",
  ).map((rawTransform, index) => {
    const path = `depth_metadata.temporal_alignment.frame_transforms[${index}]`;
    const transform = objectAt(rawTransform, path);
    if (typeof transform.used_full_frame_fallback !== "boolean") {
      throw new Error(`${path}.used_full_frame_fallback must be a boolean.`);
    }
    return {
      frame_index: integerAt(transform.frame_index, `${path}.frame_index`, 0),
      timestamp_seconds: numberAt(transform.timestamp_seconds, `${path}.timestamp_seconds`, 0),
      scale: numberAt(transform.scale, `${path}.scale`, Number.EPSILON),
      shift: numberAt(transform.shift, `${path}.shift`),
      anchor_median_raw: numberAt(transform.anchor_median_raw, `${path}.anchor_median_raw`),
      anchor_iqr_raw: numberAt(transform.anchor_iqr_raw, `${path}.anchor_iqr_raw`, Number.EPSILON),
      anchor_pixel_count_sampled: integerAt(
        transform.anchor_pixel_count_sampled,
        `${path}.anchor_pixel_count_sampled`,
        16,
      ),
      used_full_frame_fallback: transform.used_full_frame_fallback,
      stability_quality: numberAt(transform.stability_quality, `${path}.stability_quality`, 0, 1),
    };
  });
  if (frameTransforms.length !== frames.length) {
    throw new Error("temporal alignment must contain exactly one transform per depth frame.");
  }
  frameTransforms.forEach((transform, index) => {
    if (transform.frame_index !== frames[index].frame_index
      || transform.timestamp_seconds !== frames[index].timestamp_seconds) {
      throw new Error("temporal alignment transforms must match depth frames in order.");
    }
  });
  const temporalAlignment = {
    method: "per_frame_robust_affine_to_clip_anchor" as const,
    anchor_region: "common_untracked_background_with_full_frame_fallback" as const,
    lower_quantile: 0.25 as const,
    center_quantile: 0.5 as const,
    upper_quantile: 0.75 as const,
    canonical_median: numberAt(
      temporalAlignmentValue.canonical_median,
      "depth_metadata.temporal_alignment.canonical_median",
    ),
    canonical_iqr: numberAt(
      temporalAlignmentValue.canonical_iqr,
      "depth_metadata.temporal_alignment.canonical_iqr",
      Number.EPSILON,
    ),
    frame_transforms: frameTransforms,
  };

  const normalizationValue = objectAt(metadata.normalization, "depth_metadata.normalization");
  if (normalizationValue.method !== "clip_robust_quantile") {
    throw new Error("depth_metadata.normalization.method must be clip_robust_quantile.");
  }
  const lowQuantile = numberAt(normalizationValue.low_quantile, "depth_metadata.normalization.low_quantile", 0, 1);
  const highQuantile = numberAt(normalizationValue.high_quantile, "depth_metadata.normalization.high_quantile", 0, 1);
  const alignedLow = numberAt(normalizationValue.aligned_low, "depth_metadata.normalization.aligned_low");
  const alignedHigh = numberAt(normalizationValue.aligned_high, "depth_metadata.normalization.aligned_high");
  if (highQuantile <= lowQuantile) throw new Error("normalization.high_quantile must exceed low_quantile.");
  if (alignedHigh <= alignedLow) throw new Error("normalization.aligned_high must exceed aligned_low.");
  const normalization = {
    method: "clip_robust_quantile" as const,
    low_quantile: lowQuantile,
    high_quantile: highQuantile,
    aligned_low: alignedLow,
    aligned_high: alignedHigh,
  };

  return {
    schema_version: DEPTH_FEATURE_SCHEMA,
    video: stringAt(root.video, "video"),
    contour: root.contour === undefined ? undefined : stringAt(root.contour, "contour"),
    source: {
      video_width: integerAt(source.video_width, "source.video_width", 1),
      video_height: integerAt(source.video_height, "source.video_height", 1),
      video_fps: numberAt(source.video_fps, "source.video_fps", Number.EPSILON),
      video_duration_seconds: numberAt(source.video_duration_seconds, "source.video_duration_seconds", 0),
      video_frame_count: source.video_frame_count === null
        ? null
        : integerAt(source.video_frame_count, "source.video_frame_count", 1),
      video_file_size_bytes: source.video_file_size_bytes === undefined
        ? undefined
        : integerAt(source.video_file_size_bytes, "source.video_file_size_bytes", 1),
      contour_file_size_bytes: source.contour_file_size_bytes === undefined
        ? undefined
        : integerAt(source.contour_file_size_bytes, "source.contour_file_size_bytes", 1),
    },
    depth_metadata: {
      model: stringAt(metadata.model, "depth_metadata.model"),
      model_revision: stringAt(metadata.model_revision, "depth_metadata.model_revision"),
      metric: false,
      metric_units: null,
      raw_depth_direction: "larger_is_farther",
      depth_semantics: DEPTH_RANK_SEMANTICS,
      inference_mode: "independent_depth_image",
      temporal_alignment: temporalAlignment,
      normalization,
      sample_fps: numberAt(metadata.sample_fps, "depth_metadata.sample_fps", Number.EPSILON),
      max_alignment_error_seconds: numberAt(
        metadata.max_alignment_error_seconds,
        "depth_metadata.max_alignment_error_seconds",
        0,
      ),
      max_cue_age_seconds: numberAt(metadata.max_cue_age_seconds, "depth_metadata.max_cue_age_seconds", 0),
      max_temporal_gap_seconds: numberAt(
        metadata.max_temporal_gap_seconds,
        "depth_metadata.max_temporal_gap_seconds",
        0,
      ),
    },
    frames,
    boundary_candidates: boundaryCandidates,
  };
}

export function findAlignedDepthFrame(
  data: DepthFeatureData | null,
  frameIndex: number | undefined,
  timestampSeconds: number,
): DepthFrameAlignment | null {
  if (!data?.frames.length) return null;

  const tolerance = data.depth_metadata.max_cue_age_seconds;

  if (frameIndex !== undefined) {
    let low = 0;
    let high = data.frames.length - 1;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      const candidate = data.frames[middle];
      if (candidate.frame_index === frameIndex) {
        const delta = Math.abs(candidate.timestamp_seconds - timestampSeconds);
        if (delta > tolerance + Number.EPSILON) return null;
        return {
          frame: candidate,
          delta_seconds: delta,
          exact_frame: true,
        };
      }
      if (candidate.frame_index < frameIndex) low = middle + 1;
      else high = middle - 1;
    }
  }

  let low = 0;
  let high = data.frames.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (data.frames[middle].timestamp_seconds < timestampSeconds) low = middle + 1;
    else high = middle;
  }
  const next = data.frames[low];
  const previous = low > 0 ? data.frames[low - 1] : null;
  const closest = previous
    && Math.abs(previous.timestamp_seconds - timestampSeconds) <= Math.abs(next.timestamp_seconds - timestampSeconds)
    ? previous
    : next;
  const delta = Math.abs(closest.timestamp_seconds - timestampSeconds);
  if (delta > tolerance + Number.EPSILON) return null;
  return { frame: closest, delta_seconds: delta, exact_frame: false };
}
