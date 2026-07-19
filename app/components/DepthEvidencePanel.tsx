"use client";

import { KeyboardEvent, useMemo } from "react";
import {
  DepthFeatureData,
  DepthFrameAlignment,
  depthPairKey,
  splitDepthPairKey,
} from "../lib/depth-features";

export type DepthPairOption = {
  key: string;
  source_id: string;
  target_id: string;
};

type Props = {
  data: DepthFeatureData | null;
  alignment: DepthFrameAlignment | null;
  pairOptions: DepthPairOption[];
  selectedPairKey: string;
  duration: number;
  showInstanceRanks: boolean;
  onPairChange: (key: string) => void;
  onToggleInstanceRanks: () => void;
  onSeek: (seconds: number) => void;
};

function cueNumber(value: number | undefined, digits = 3) {
  return value === undefined ? "—" : value.toFixed(digits);
}

function percent(value: number | undefined) {
  return value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

function downsample<T>(values: T[], maximum: number) {
  if (values.length <= maximum) return values;
  const result: T[] = [];
  const stride = (values.length - 1) / (maximum - 1);
  for (let index = 0; index < maximum; index += 1) result.push(values[Math.round(index * stride)]);
  return result;
}

function splitTimeline<T extends { time: number }>(values: T[], maximumGap: number) {
  const segments: T[][] = [];
  for (const value of values) {
    const current = segments.at(-1);
    if (!current?.length || value.time - current.at(-1)!.time > maximumGap) {
      segments.push([value]);
    } else {
      current.push(value);
    }
  }
  return segments;
}

export default function DepthEvidencePanel({
  data,
  alignment,
  pairOptions,
  selectedPairKey,
  duration,
  showInstanceRanks,
  onPairChange,
  onToggleInstanceRanks,
  onSeek,
}: Props) {
  const [sourceId, targetId] = splitDepthPairKey(selectedPairKey);
  const currentPair = alignment?.frame.pairs.find(
    (pair) => pair.source_id === sourceId && pair.target_id === targetId,
  );

  const timelineSamples = useMemo(() => {
    if (!data || !selectedPairKey) return [];
    return data.frames.flatMap((frame) => {
      const pair = frame.pairs.find((candidate) => depthPairKey(candidate.source_id, candidate.target_id) === selectedPairKey);
      return pair ? [{ time: frame.timestamp_seconds, pair }] : [];
    });
  }, [data, selectedPairKey]);
  const maximumGap = data?.depth_metadata.max_temporal_gap_seconds ?? 0.5;
  const timelineSegments = splitTimeline(timelineSamples, maximumGap)
    .map((segment) => downsample(segment, Math.max(2, Math.floor(700 / Math.max(1, timelineSamples.length) * segment.length))));
  const timelineDuration = Math.max(duration, timelineSamples.at(-1)?.time ?? 0, 0.001);
  const boundaryCandidates = (data?.boundary_candidates ?? []).filter(
    (candidate) => depthPairKey(candidate.source_id, candidate.target_id) === selectedPairKey,
  );

  function seekFromKey(event: KeyboardEvent<SVGGElement>, seconds: number) {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    onSeek(seconds);
  }

  return (
    <section className="depth-panel" aria-label="Estimated relative depth cues">
      <div className="depth-title">
        <div><span className="eyebrow">SPATIAL–TEMPORAL CUE</span><h2>Relative depth</h2></div>
        <span className={data ? "depth-ready" : "depth-off"}>{data ? "Ready" : "Optional"}</span>
      </div>

      {!data ? (
        <div className="depth-empty">
          Load an optional <strong>palona.depth-features/v1</strong> JSON file to inspect person–object proximity and boundary candidates.
        </div>
      ) : (
        <>
          <div className="depth-controls">
            <label htmlFor="depth-pair">Person–object pair</label>
            <select
              id="depth-pair"
              aria-label="Depth cue pair"
              value={selectedPairKey}
              onChange={(event) => onPairChange(event.target.value)}
              disabled={!pairOptions.length}
            >
              {!pairOptions.length && <option value="">No pair for current selection</option>}
              {pairOptions.map((pair) => (
                <option key={pair.key} value={pair.key}>{pair.source_id} → {pair.target_id}</option>
              ))}
            </select>
            <label className="rank-toggle">
              <input type="checkbox" checked={showInstanceRanks} onChange={onToggleInstanceRanks} />
              Show depth rank on overlay
            </label>
          </div>

          <div className="depth-evidence" aria-label="Depth cues for the current frame">
            {currentPair ? (
              <>
                <div><span>Person rank</span><strong>{cueNumber(currentPair.source_depth_rank)}</strong></div>
                <div><span>Local object rank</span><strong>{cueNumber(currentPair.target_local_depth_rank)}</strong></div>
                <div><span>Relative depth gap</span><strong>{cueNumber(currentPair.depth_gap_abs)}</strong></div>
                <div><span>2D mask gap</span><strong>{cueNumber(currentPair.mask_gap_2d_norm)}</strong></div>
                <div><span>Trend</span><strong className={`trend-${currentPair.trend}`}>{currentPair.trend}</strong></div>
                <div><span>Cue quality</span><strong>{percent(currentPair.feature_quality)}</strong></div>
                <div><span>Visible target</span><strong>{percent(currentPair.target_visible_ratio)}</strong></div>
                <div><span>Proximity</span><strong>{percent(currentPair.proximity_score)}</strong></div>
                <div><span>Close duration</span><strong>{currentPair.proximity_duration_seconds.toFixed(2)} s</strong></div>
                <div><span>Start candidate</span><strong>{percent(currentPair.start_candidate_score)}</strong></div>
                <div><span>End candidate</span><strong>{percent(currentPair.end_candidate_score)}</strong></div>
              </>
            ) : (
              <p>{alignment ? "The selected pair has no cue at this frame." : "No depth frame is aligned within the declared tolerance."}</p>
            )}
          </div>

          <div className="depth-alignment">
            {alignment
              ? `${alignment.exact_frame ? "Exact frame" : "Nearest sample"} · Δt ${alignment.delta_seconds.toFixed(3)} s`
              : "Cue unavailable at the current video time"}
          </div>

          <div className="feature-timeline">
            <div className="timeline-legend">
              <span className="gap-legend">Depth gap</span>
              <span className="candidate-legend">Boundary score</span>
              <em>Click candidate markers to seek</em>
            </div>
            {timelineSamples.length ? (
              <svg viewBox="0 0 1000 112" role="group" aria-label="Relative depth and boundary candidate timeline" preserveAspectRatio="none">
                <line x1="0" y1="88" x2="1000" y2="88" className="timeline-axis" />
                {timelineSegments.map((segment, index) => {
                  const gapPoints = segment.map(({ time, pair }) => (
                    `${(time / timelineDuration) * 1000},${88 - pair.depth_gap_abs * 68}`
                  )).join(" ");
                  const candidatePoints = segment.map(({ time, pair }) => (
                    `${(time / timelineDuration) * 1000},${88 - Math.max(pair.start_candidate_score, pair.end_candidate_score) * 68}`
                  )).join(" ");
                  return (
                    <g key={`${segment[0]?.time ?? index}-${index}`}>
                      <polyline points={gapPoints} className="depth-gap-line" />
                      <polyline points={candidatePoints} className="candidate-score-line" />
                    </g>
                  );
                })}
                {boundaryCandidates.map((candidate) => {
                  const startX = Math.min(1000, Math.max(0, (candidate.start_time / timelineDuration) * 1000));
                  const endX = candidate.end_time === null
                    ? startX
                    : Math.min(1000, Math.max(startX, (candidate.end_time / timelineDuration) * 1000));
                  return (
                    <g key={candidate.candidate_id}>
                      {candidate.end_time !== null && (
                        <rect x={startX} y="5" width={Math.max(2, endX - startX)} height="88" className="candidate-window" />
                      )}
                      <g
                        className="candidate-marker start"
                        role="button"
                        tabIndex={0}
                        aria-label={`Seek to candidate start ${candidate.start_time.toFixed(3)} seconds`}
                        onClick={() => onSeek(candidate.start_time)}
                        onKeyDown={(event) => seekFromKey(event, candidate.start_time)}
                      >
                        <line x1={startX} y1="4" x2={startX} y2="100" />
                        <circle cx={startX} cy="101" r="7"><title>Start {candidate.start_time.toFixed(3)} s · score {candidate.peak_score.toFixed(2)}</title></circle>
                      </g>
                      {candidate.end_time !== null && (
                        <g
                          className="candidate-marker end"
                          role="button"
                          tabIndex={0}
                          aria-label={`Seek to candidate end ${candidate.end_time.toFixed(3)} seconds`}
                          onClick={() => onSeek(candidate.end_time!)}
                          onKeyDown={(event) => seekFromKey(event, candidate.end_time!)}
                        >
                          <line x1={endX} y1="4" x2={endX} y2="100" />
                          <circle cx={endX} cy="101" r="7"><title>End {candidate.end_time.toFixed(3)} s · quality {candidate.quality.toFixed(2)}</title></circle>
                        </g>
                      )}
                    </g>
                  );
                })}
              </svg>
            ) : <p>No time-series samples for this pair.</p>}
          </div>
        </>
      )}

      <p className="relative-depth-warning">
        Estimated relative depth only: 0 = nearer, 1 = farther. Values are not metric meters and never change labels automatically.
      </p>
    </section>
  );
}
