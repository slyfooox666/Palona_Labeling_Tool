"use client";

import { ChangeEvent, MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Point = [number, number];
type Track = {
  track_id: string;
  label: string;
  confidence?: number;
  contours_xy: Point[][];
};
type Frame = {
  frame_index: number;
  timestamp_seconds: number;
  tracks: Track[];
};
type ControlData = {
  video?: string;
  frames: Frame[];
};
type HoveredContour = {
  id: string;
  label: string;
  x: number;
  y: number;
};

const PALETTE = ["#59d9ff", "#ffcb52", "#a78bfa", "#5ee6a8", "#ff7e9d", "#fb923c", "#67e8f9"];

function colorFor(value: string) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) hash = (hash * 31 + value.charCodeAt(i)) | 0;
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "00:00.000";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function nearestFrame(frames: Frame[], time: number) {
  if (!frames.length) return null;
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle].timestamp_seconds < time) low = middle + 1;
    else high = middle;
  }
  if (low > 0 && Math.abs(frames[low - 1].timestamp_seconds - time) < Math.abs(frames[low].timestamp_seconds - time)) {
    return frames[low - 1];
  }
  return frames[low];
}

function pointInPolygon(point: Point, polygon: Point[]) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const crosses = yi > point[1] !== yj > point[1];
    if (crosses && point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function polygonArea(polygon: Point[]) {
  return Math.abs(polygon.reduce((sum, point, index) => {
    const next = polygon[(index + 1) % polygon.length];
    return sum + point[0] * next[1] - next[0] * point[1];
  }, 0) / 2);
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const jsonInputRef = useRef<HTMLInputElement>(null);
  const videoUrlRef = useRef<string | null>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState("No video selected");
  const [jsonName, setJsonName] = useState("No control file selected");
  const [data, setData] = useState<ControlData | null>(null);
  const [loadState, setLoadState] = useState<"idle" | "reading" | "ready" | "error">("idle");
  const [message, setMessage] = useState("Choose a video and its control JSON to begin.");
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoSize, setVideoSize] = useState({ width: 16, height: 9 });
  const [selectedTracks, setSelectedTracks] = useState<Set<string>>(new Set());
  const [hoveredContour, setHoveredContour] = useState<HoveredContour | null>(null);

  const currentFrame = useMemo(() => nearestFrame(data?.frames ?? [], currentTime), [data, currentTime]);

  const catalog = useMemo(() => {
    const byTrack = new Map<string, { id: string; label: string; count: number }>();
    const types = new Set<string>();
    for (const frame of data?.frames ?? []) {
      for (const track of frame.tracks) {
        types.add(track.label);
        const item = byTrack.get(track.track_id);
        if (item) item.count += 1;
        else byTrack.set(track.track_id, { id: track.track_id, label: track.label, count: 1 });
      }
    }
    return {
      types: [...types].sort(),
      tracks: [...byTrack.values()].sort((a, b) => a.label.localeCompare(b.label) || a.id.localeCompare(b.id)),
    };
  }, [data]);

  const fps = useMemo(() => {
    const frames = data?.frames ?? [];
    for (let i = 1; i < Math.min(frames.length, 30); i += 1) {
      const delta = frames[i].timestamp_seconds - frames[i - 1].timestamp_seconds;
      if (delta > 0) return 1 / delta;
    }
    return 30;
  }, [data]);

  const visibleTracks = useMemo(() => (currentFrame?.tracks ?? []).filter(
    (track) => selectedTracks.has(track.track_id),
  ), [currentFrame, selectedTracks]);

  const renderedTracks = useMemo(() => (currentFrame?.tracks ?? []).filter(
    (track) => selectedTracks.has(track.track_id) || track.track_id === hoveredContour?.id,
  ), [currentFrame, hoveredContour?.id, selectedTracks]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const width = video.videoWidth || videoSize.width;
    const height = video.videoHeight || videoSize.height;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);

    for (const track of renderedTracks) {
      const highlighted = track.track_id === hoveredContour?.id;
      const color = colorFor(track.track_id);
      for (const contour of track.contours_xy ?? []) {
        if (contour.length < 3) continue;
        context.beginPath();
        context.moveTo(contour[0][0], contour[0][1]);
        for (let i = 1; i < contour.length; i += 1) context.lineTo(contour[i][0], contour[i][1]);
        context.closePath();
        context.fillStyle = `${color}${highlighted ? "62" : "2b"}`;
        context.strokeStyle = highlighted ? "#ffffff" : color;
        context.lineWidth = highlighted ? Math.max(7, width / 450) : Math.max(3, width / 900);
        context.fill();
        context.stroke();
      }
    }
  }, [hoveredContour?.id, renderedTracks, videoSize]);

  useEffect(() => draw(), [draw]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;
    let active = true;
    let frameCallback = 0;
    let animationFrame = 0;
    const update = () => {
      if (!active) return;
      setCurrentTime(video.currentTime);
      if ("requestVideoFrameCallback" in video) {
        frameCallback = video.requestVideoFrameCallback(update);
      } else {
        animationFrame = requestAnimationFrame(update);
      }
    };
    update();
    return () => {
      active = false;
      if (frameCallback && "cancelVideoFrameCallback" in video) video.cancelVideoFrameCallback(frameCallback);
      if (animationFrame) cancelAnimationFrame(animationFrame);
    };
  }, [videoUrl]);

  useEffect(() => () => {
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement)?.matches("input, button")) return;
      if (event.code === "Space") {
        event.preventDefault();
        const video = videoRef.current;
        if (!video) return;
        if (video.paused) void video.play();
        else video.pause();
      }
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        stepFrame(event.key === "ArrowLeft" ? -1 : 1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  function loadVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
    const url = URL.createObjectURL(file);
    videoUrlRef.current = url;
    setVideoUrl(url);
    setVideoName(file.name);
    setMessage("Video loaded. Select the matching control JSON.");
    setCurrentTime(0);
    setDuration(0);
    setHoveredContour(null);
  }

  function loadJson(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setJsonName(file.name);
    setLoadState("reading");
    setMessage(`Reading ${file.name}. Large control files may take a moment…`);

    const workerSource = `
      self.onmessage = async ({ data: file }) => {
        try {
          const raw = JSON.parse(await file.text());
          const frames = raw.frames.map((frame) => ({
            frame_index: frame.frame_index,
            timestamp_seconds: frame.timestamp_seconds,
            tracks: (frame.tracks || []).map((track) => ({
              track_id: track.track_id,
              label: track.label,
              confidence: track.confidence,
              contours_xy: track.contours_xy || track.metadata?.contours_xy || [],
            })),
          }));
          self.postMessage({ ok: true, value: { video: raw.video, frames } });
        } catch (error) {
          self.postMessage({ ok: false, error: error instanceof Error ? error.message : String(error) });
        }
      };
    `;
    const workerUrl = URL.createObjectURL(new Blob([workerSource], { type: "text/javascript" }));
    const worker = new Worker(workerUrl);
    worker.onmessage = (workerEvent: MessageEvent<{ ok: boolean; value?: ControlData; error?: string }>) => {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      if (!workerEvent.data.ok || !workerEvent.data.value) {
        setLoadState("error");
        setMessage(`Could not read control JSON: ${workerEvent.data.error ?? "Unknown error"}`);
        return;
      }
      const nextData = workerEvent.data.value;
      const tracks = new Set<string>();
      for (const frame of nextData.frames) {
        for (const track of frame.tracks) {
          tracks.add(track.track_id);
        }
      }
      setData(nextData);
      setSelectedTracks(tracks);
      setLoadState("ready");
      setMessage(`${nextData.frames.length.toLocaleString()} annotated frames loaded.`);
    };
    worker.onerror = (error) => {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      setLoadState("error");
      setMessage(`Could not read control JSON: ${error.message}`);
    };
    worker.postMessage(file);
    event.target.value = "";
  }

  function stepFrame(direction: -1 | 1) {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    const next = Math.min(video.duration || Number.POSITIVE_INFINITY, Math.max(0, video.currentTime + direction / fps));
    video.currentTime = next;
    setCurrentTime(next);
  }

  function seek(value: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = value;
    setCurrentTime(value);
  }

  function toggleType(type: string) {
    const trackIds = catalog.tracks.filter((track) => track.label === type).map((track) => track.id);
    const shouldSelect = trackIds.some((id) => !selectedTracks.has(id));
    setSelectedTracks((current) => {
      const next = new Set(current);
      for (const id of trackIds) {
        if (shouldSelect) next.add(id);
        else next.delete(id);
      }
      return next;
    });
    setHoveredContour(null);
  }

  function isTypeSelected(type: string) {
    const trackIds = catalog.tracks.filter((track) => track.label === type).map((track) => track.id);
    return trackIds.length > 0 && trackIds.every((id) => selectedTracks.has(id));
  }

  function toggleTrack(id: string) {
    setSelectedTracks((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setHoveredContour(null);
  }

  function hitTest(event: MouseEvent<HTMLCanvasElement>): HoveredContour | null {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const bounds = canvas.getBoundingClientRect();
    const point: Point = [
      ((event.clientX - bounds.left) / bounds.width) * canvas.width,
      ((event.clientY - bounds.top) / bounds.height) * canvas.height,
    ];
    const hits: { id: string; label: string; area: number }[] = [];
    for (const track of currentFrame?.tracks ?? []) {
      for (const contour of track.contours_xy ?? []) {
        if (contour.length >= 3 && pointInPolygon(point, contour)) {
          hits.push({ id: track.track_id, label: track.label, area: polygonArea(contour) });
        }
      }
    }
    hits.sort((a, b) => a.area - b.area);
    const hit = hits[0];
    if (!hit) return null;
    return {
      id: hit.id,
      label: hit.label,
      x: Math.min(event.clientX - bounds.left + 12, Math.max(12, bounds.width - 130)),
      y: Math.min(event.clientY - bounds.top + 12, Math.max(12, bounds.height - 34)),
    };
  }

  function toggleTrackFromCanvas(contour: HoveredContour) {
    const isVisible = selectedTracks.has(contour.id);
    if (isVisible) {
      setSelectedTracks((current) => {
        const next = new Set(current);
        next.delete(contour.id);
        return next;
      });
    } else {
      setSelectedTracks((current) => new Set(current).add(contour.id));
    }
    setHoveredContour(null);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark">PL</div>
        <div>
          <h1>Palona contour lab</h1>
          <p>Inspect frame-accurate tracks on local video</p>
        </div>
        <div className={`status-pill ${loadState}`}><span />{loadState === "reading" ? "Parsing control data" : loadState === "ready" ? "Annotations ready" : "Local session"}</div>
      </header>

      <section className="load-strip" aria-label="Local files">
        <button className="file-card" onClick={() => videoInputRef.current?.click()}>
          <span className="file-icon">▶</span>
          <span><strong>Video clip</strong><small>{videoName}</small></span>
          <b>Choose</b>
        </button>
        <button className="file-card" onClick={() => jsonInputRef.current?.click()} disabled={loadState === "reading"}>
          <span className="file-icon json">{`{ }`}</span>
          <span><strong>Control JSON</strong><small>{jsonName}</small></span>
          <b>{loadState === "reading" ? "Reading…" : "Choose"}</b>
        </button>
        <input ref={videoInputRef} className="visually-hidden" type="file" accept="video/*,.mkv" onChange={loadVideo} />
        <input ref={jsonInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={loadJson} />
        <p className="session-message">{message}</p>
      </section>

      <section className="workspace">
        <div className="viewer-panel">
          <div className="viewer-head">
            <div><span className="eyebrow">FRAME VIEWER</span><strong>{videoName}</strong></div>
            <div className="frame-readout"><span>FRAME</span><b>{currentFrame?.frame_index ?? "—"}</b><span>VISIBLE</span><b>{visibleTracks.length}</b></div>
          </div>

          <div className="stage-wrap" style={{ aspectRatio: `${videoSize.width} / ${videoSize.height}` }}>
            {videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  playsInline
                  onLoadedMetadata={(event) => {
                    const video = event.currentTarget;
                    setDuration(video.duration);
                    setVideoSize({ width: video.videoWidth, height: video.videoHeight });
                    setMessage(`Video ready · ${video.videoWidth}×${video.videoHeight}`);
                  }}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onSeeked={(event) => setCurrentTime(event.currentTarget.currentTime)}
                  onError={() => setMessage("This browser could not decode the selected video. MKV/HEVC support varies by browser.")}
                />
                <canvas
                  ref={canvasRef}
                  aria-label="Interactive contour overlay"
                  onMouseMove={(event) => setHoveredContour(hitTest(event))}
                  onMouseLeave={() => setHoveredContour(null)}
                  onClick={(event) => {
                    const contour = hitTest(event);
                    if (contour) toggleTrackFromCanvas(contour);
                  }}
                />
                {hoveredContour && (
                  <div className="hover-label" style={{ left: hoveredContour.x, top: hoveredContour.y }}>
                    {hoveredContour.id} {hoveredContour.label}
                  </div>
                )}
              </>
            ) : (
              <div className="empty-stage"><span>▶</span><strong>Load a local video clip</strong><p>MKV, MP4, MOV, or WebM</p></div>
            )}
          </div>

          <div className="transport">
            <button aria-label="Previous frame" onClick={() => stepFrame(-1)} disabled={!videoUrl}>│◀</button>
            <button className="play-button" aria-label={isPlaying ? "Pause" : "Play"} onClick={() => {
              const video = videoRef.current;
              if (!video) return;
              if (video.paused) void video.play(); else video.pause();
            }} disabled={!videoUrl}>{isPlaying ? "Ⅱ" : "▶"}</button>
            <button aria-label="Next frame" onClick={() => stepFrame(1)} disabled={!videoUrl}>▶│</button>
            <span className="timecode">{formatTime(currentTime)}</span>
            <input aria-label="Video position" type="range" min="0" max={duration || 0} step="0.001" value={Math.min(currentTime, duration || 0)} onChange={(event) => seek(Number(event.target.value))} disabled={!videoUrl} />
            <span className="timecode muted">{formatTime(duration)}</span>
            <span className="fps">{fps.toFixed(2)} FPS</span>
          </div>
          <p className="shortcut-hint"><kbd>Space</kbd> play / pause <kbd>←</kbd><kbd>→</kbd> step one frame</p>
        </div>

        <aside className="inspector">
          <div className="inspector-title"><div><span className="eyebrow">OVERLAY FILTERS</span><h2>Contours</h2></div><span>{selectedTracks.size}/{catalog.tracks.length}</span></div>

          <section className="filter-section">
            <div className="section-heading"><h3>Object type</h3><div><button onClick={() => setSelectedTracks(new Set(catalog.tracks.map((track) => track.id)))}>All</button><button onClick={() => setSelectedTracks(new Set())}>None</button></div></div>
            <div className="type-grid">
              {catalog.types.length ? catalog.types.map((type) => (
                <label key={type} className={isTypeSelected(type) ? "checked" : ""}>
                  <input type="checkbox" checked={isTypeSelected(type)} onChange={() => toggleType(type)} />
                  <span className="checkmark">✓</span><span>{type}</span>
                </label>
              )) : <p className="empty-list">Load control JSON to see object types.</p>}
            </div>
          </section>

          <section className="filter-section tracks-section">
            <div className="section-heading"><h3>Track ID</h3><div><button onClick={() => setSelectedTracks(new Set(catalog.tracks.map((track) => track.id)))}>All</button><button onClick={() => setSelectedTracks(new Set())}>None</button></div></div>
            <div className="track-list">
              {catalog.tracks.map((track) => (
                <label key={track.id} className={`${selectedTracks.has(track.id) ? "checked" : ""} ${hoveredContour?.id === track.id ? "hovered" : ""}`}>
                  <input type="checkbox" checked={selectedTracks.has(track.id)} onChange={() => toggleTrack(track.id)} />
                  <span className="color-dot" style={{ background: colorFor(track.id) }} />
                  <span><strong>{track.id}</strong><small>{track.label}</small></span>
                  <em>{track.count.toLocaleString()}f</em>
                  <span className="checkmark">✓</span>
                </label>
              ))}
            </div>
          </section>

          <div className="current-summary">
            <span className="pulse-dot" />
            <div><strong>Current frame</strong><small>{currentFrame ? `${currentFrame.tracks.length} tracks · ${formatTime(currentFrame.timestamp_seconds)}` : "No annotation data"}</small></div>
          </div>
        </aside>
      </section>
    </main>
  );
}
