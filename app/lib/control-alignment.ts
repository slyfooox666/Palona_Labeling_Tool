export type TimedControlFrame = {
  timestamp_seconds: number;
};

export type ControlAlignmentOptions = {
  sourceFps?: number;
  sampleFps?: number;
};

const ALIGNMENT_EPSILON_SECONDS = 1e-4;
const FLOAT_COMPARISON_EPSILON_SECONDS = 1e-9;

export function nearestTimedFrame<T extends TimedControlFrame>(frames: T[], time: number): T | null {
  if (!frames.length || !Number.isFinite(time)) return null;
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle].timestamp_seconds < time) low = middle + 1;
    else high = middle;
  }
  if (low > 0
    && Math.abs(frames[low - 1].timestamp_seconds - time)
      < Math.abs(frames[low].timestamp_seconds - time)) {
    return frames[low - 1];
  }
  return frames[low];
}

export function inferControlSampleFps(frames: TimedControlFrame[]): number | null {
  const intervals = frames.slice(1)
    .map((frame, index) => frame.timestamp_seconds - frames[index].timestamp_seconds)
    .filter((interval) => Number.isFinite(interval) && interval > 0)
    .sort((left, right) => left - right);
  if (!intervals.length) return null;
  const middle = Math.floor(intervals.length / 2);
  const median = intervals.length % 2
    ? intervals[middle]
    : (intervals[middle - 1] + intervals[middle]) / 2;
  return median > 0 ? 1 / median : null;
}

export function controlMaxCueAgeSeconds(
  frames: TimedControlFrame[],
  options: ControlAlignmentOptions = {},
): number {
  const sourceFps = options.sourceFps && Number.isFinite(options.sourceFps) && options.sourceFps > 0
    ? options.sourceFps
    : null;
  const sampleFps = options.sampleFps && Number.isFinite(options.sampleFps) && options.sampleFps > 0
    ? options.sampleFps
    : inferControlSampleFps(frames);
  const alignmentTolerance = (sourceFps ? 0.5 / sourceFps : 0) + ALIGNMENT_EPSILON_SECONDS;
  return sampleFps ? 0.5 / sampleFps + alignmentTolerance : alignmentTolerance;
}

export function findAlignedControlFrame<T extends TimedControlFrame>(
  frames: T[],
  time: number,
  options: ControlAlignmentOptions = {},
): T | null {
  const frame = nearestTimedFrame(frames, time);
  if (!frame) return null;
  const delta = Math.abs(frame.timestamp_seconds - time);
  return delta <= controlMaxCueAgeSeconds(frames, options) + FLOAT_COMPARISON_EPSILON_SECONDS
    ? frame
    : null;
}
