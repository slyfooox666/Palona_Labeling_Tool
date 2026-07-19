"use client";

import {
  ChangeEvent,
  MouseEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import DepthEvidencePanel, { DepthPairOption } from "./components/DepthEvidencePanel";
import { findAlignedControlFrame } from "./lib/control-alignment";
import {
  DepthFeatureData,
  depthPairKey,
  findAlignedDepthFrame,
  parseDepthFeatures,
} from "./lib/depth-features";
import {
  ANNOTATION_PROJECT_SCHEMA,
  AnnotationProject,
  IdAliases,
  Interaction as ProjectInteraction,
  isPersonTrackLabel,
  isTableTrackLabel,
  parseAnnotationProjectJson,
  resolveTrackId,
  serializeAnnotationProject,
  serializeMinimalInteractions,
  timeToFrame,
  validateNormalizedRoiPolygon,
  withIdAlias,
  withoutIdAlias,
} from "./lib/labeling-project";

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
  video_fps?: number;
  sample_fps?: number;
  video_width?: number;
  video_height?: number;
  video_frame_count?: number;
  video_duration_seconds?: number;
  frames: Frame[];
};
type HoveredContour = {
  id: string;
  label: string;
  frame_index: number;
  x: number;
  y: number;
};
type Interaction = {
  interaction_type: string;
  interaction_id: string;
  object_id_list: string[];
  start_time: number;
  end_time: number | null;
};
type MaskMode = "fill" | "contour";

const PALETTE = ["#59d9ff", "#ffcb52", "#a78bfa", "#5ee6a8", "#ff7e9d", "#fb923c", "#67e8f9"];
const NATURAL_COLLATOR = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

function naturalCompare(left: string, right: string) {
  return NATURAL_COLLATOR.compare(left, right);
}

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

function annotationTime(seconds: number) {
  return Number(seconds.toFixed(3));
}

function fileBaseName(path: string) {
  return path.replaceAll("\\", "/").split("/").at(-1) ?? path;
}

function fileStem(path: string) {
  return fileBaseName(path).replace(/\.[^.]+$/u, "");
}

function autosaveKey(video: string, contour: string, videoBytes: number, contourBytes: number) {
  return `palona.annotation-project/v1/${encodeURIComponent(fileBaseName(video))}`
    + `/${Math.max(0, videoBytes)}/${encodeURIComponent(fileBaseName(contour))}/${Math.max(0, contourBytes)}`;
}

function downloadJsonText(contents: string, filename: string) {
  const blob = new Blob([contents], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function nextInteractionId(interactions: Interaction[]) {
  const usedIds = new Set(interactions.map((interaction) => interaction.interaction_id));
  let index = 0;
  while (usedIds.has(`i${index}`)) index += 1;
  return `i${index}`;
}

function interactionsEqual(left: Interaction, right: Interaction) {
  return left.interaction_type === right.interaction_type
    && left.interaction_id === right.interaction_id
    && left.start_time === right.start_time
    && left.end_time === right.end_time
    && left.object_id_list.length === right.object_id_list.length
    && left.object_id_list.every((id, index) => id === right.object_id_list[index]);
}

function cloneInteraction(interaction: Interaction): Interaction {
  return { ...interaction, object_id_list: [...interaction.object_id_list] };
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

function inferControlFps(control: ControlData | null) {
  if (control?.video_fps && Number.isFinite(control.video_fps) && control.video_fps > 0) {
    return control.video_fps;
  }
  const frames = control?.frames ?? [];
  for (let index = 1; index < Math.min(frames.length, 30); index += 1) {
    const delta = frames[index].timestamp_seconds - frames[index - 1].timestamp_seconds;
    const frameDelta = frames[index].frame_index - frames[index - 1].frame_index;
    if (delta > 0 && frameDelta > 0) return frameDelta / delta;
  }
  return 30;
}

function expectedSourceFrameCount(control: ControlData | null, durationSeconds: number, fps: number) {
  const lastControlFrame = control?.frames.at(-1)?.frame_index ?? 0;
  const fromDuration = durationSeconds > 0 ? Math.ceil(durationSeconds * fps) : 0;
  return Math.max(1, control?.video_frame_count ?? 0, lastControlFrame + 1, fromDuration);
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

function isInteractiveTarget(target: EventTarget | null) {
  return target instanceof Element
    && Boolean(target.closest("input, button, select, textarea, a, [role='button'], [contenteditable]"));
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const jsonInputRef = useRef<HTMLInputElement>(null);
  const depthInputRef = useRef<HTMLInputElement>(null);
  const projectInputRef = useRef<HTMLInputElement>(null);
  const interactionTypeInputRef = useRef<HTMLInputElement>(null);
  const videoUrlRef = useRef<string | null>(null);
  const draggingRoiVertexRef = useRef<number | null>(null);
  const autosaveSuspendedRef = useRef(true);
  const buildAnnotationProjectRef = useRef<(() => AnnotationProject) | null>(null);
  const controlWorkerRef = useRef<Worker | null>(null);
  const controlWorkerUrlRef = useRef<string | null>(null);
  const sourceLoadGenerationRef = useRef(0);
  const depthLoadGenerationRef = useRef(0);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState("No video selected");
  const [videoPath, setVideoPath] = useState("");
  const [videoFileSize, setVideoFileSize] = useState(0);
  const [jsonName, setJsonName] = useState("No control file selected");
  const [contourPath, setContourPath] = useState("");
  const [contourFileSize, setContourFileSize] = useState(0);
  const [depthName, setDepthName] = useState("No depth cues selected");
  const [projectName, setProjectName] = useState("Unsaved project");
  const [projectCreatedAt, setProjectCreatedAt] = useState(() => new Date().toISOString());
  const [autosaveStatus, setAutosaveStatus] = useState("Autosave waits for video + Control JSON");
  const [data, setData] = useState<ControlData | null>(null);
  const [depthData, setDepthData] = useState<DepthFeatureData | null>(null);
  const [loadState, setLoadState] = useState<"idle" | "reading" | "ready" | "error">("idle");
  const [depthLoadState, setDepthLoadState] = useState<"idle" | "reading" | "ready" | "error">("idle");
  const [message, setMessage] = useState("Choose a video and its control JSON to begin.");
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [videoSize, setVideoSize] = useState({ width: 16, height: 9 });
  const [showMasks, setShowMasks] = useState(true);
  const [maskMode, setMaskMode] = useState<MaskMode>("fill");
  const [maskFillOpacity, setMaskFillOpacity] = useState(0.17);
  const [showTrackIds, setShowTrackIds] = useState(true);
  const [roiPolygon, setRoiPolygon] = useState<Point[]>([]);
  const [roiComplete, setRoiComplete] = useState(false);
  const [roiEditing, setRoiEditing] = useState(false);
  const [roiBlackout, setRoiBlackout] = useState(true);
  const [draggingRoiVertex, setDraggingRoiVertex] = useState<number | null>(null);
  const [visibleTrackIds, setVisibleTrackIds] = useState<Set<string>>(new Set());
  const [interactionTrackIds, setInteractionTrackIds] = useState<Set<string>>(new Set());
  const [hoveredContour, setHoveredContour] = useState<HoveredContour | null>(null);
  const [selectedDepthPairKey, setSelectedDepthPairKey] = useState("");
  const [showDepthRanks, setShowDepthRanks] = useState(true);
  const [interactionType, setInteractionType] = useState("");
  const [interactionDraft, setInteractionDraft] = useState<Interaction | null>(null);
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [selectedInteractionId, setSelectedInteractionId] = useState<string | null>(null);
  const [editingInteractionId, setEditingInteractionId] = useState<string | null>(null);
  const [exportedInteractionSignature, setExportedInteractionSignature] = useState("");
  const [idAliases, setIdAliases] = useState<IdAliases>({});
  const [aliasSource, setAliasSource] = useState("");
  const [aliasTarget, setAliasTarget] = useState("");

  const fps = useMemo(() => inferControlFps(data), [data]);
  const currentFrame = useMemo(() => findAlignedControlFrame(
    data?.frames ?? [],
    currentTime,
    { sourceFps: fps, sampleFps: data?.sample_fps },
  ), [currentTime, data, fps]);
  const activeHoveredContour = hoveredContour?.frame_index === currentFrame?.frame_index
    ? hoveredContour
    : null;
  const selectedInteraction = useMemo(
    () => interactions.find((interaction) => interaction.interaction_id === selectedInteractionId) ?? null,
    [interactions, selectedInteractionId],
  );
  const displayedInteraction = useMemo(() => interactionDraft
    ? { ...interactionDraft, object_id_list: [...interactionTrackIds].sort(naturalCompare) }
    : selectedInteraction, [interactionDraft, interactionTrackIds, selectedInteraction]);
  const originalInteraction = useMemo(
    () => interactions.find((interaction) => interaction.interaction_id === editingInteractionId) ?? null,
    [editingInteractionId, interactions],
  );
  const hasInteractionChanges = Boolean(
    displayedInteraction && interactionDraft
      && (!originalInteraction || !interactionsEqual(displayedInteraction, originalInteraction)),
  );
  const sortedInteractions = useMemo(
    () => [...interactions].sort((left, right) => naturalCompare(left.interaction_id, right.interaction_id)),
    [interactions],
  );
  const interactionSignature = useMemo(() => JSON.stringify(sortedInteractions), [sortedInteractions]);
  const interactionTypes = useMemo(
    () => [...new Set([
      "occupy_table",
      "table_touch",
      ...interactions.map((interaction) => interaction.interaction_type),
    ])].sort(naturalCompare),
    [interactions],
  );

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
      types: [...types].sort(naturalCompare),
      tracks: [...byTrack.values()].sort((a, b) => naturalCompare(a.label, b.label) || naturalCompare(a.id, b.id)),
    };
  }, [data]);

  const canonicalTrackIds = useMemo(() => new Map(
    catalog.tracks.map((track) => [track.id, resolveTrackId(track.id, idAliases)]),
  ), [catalog.tracks, idAliases]);

  const visibleTracks = useMemo(() => (currentFrame?.tracks ?? []).filter(
    (track) => visibleTrackIds.has(track.track_id),
  ), [currentFrame, visibleTrackIds]);

  const renderedTracks = useMemo(() => showMasks ? visibleTracks : [], [showMasks, visibleTracks]);

  const depthAlignment = useMemo(
    () => findAlignedDepthFrame(depthData, currentFrame?.frame_index, currentTime),
    [currentFrame?.frame_index, currentTime, depthData],
  );
  const currentDepthRanks = useMemo(() => new Map(
    (depthAlignment?.frame.instances ?? []).map((instance) => [instance.track_id, instance.depth_rank]),
  ), [depthAlignment]);

  const depthPairCatalog = useMemo(() => {
    const pairs = new Map<string, DepthPairOption>();
    for (const frame of depthData?.frames ?? []) {
      for (const pair of frame.pairs) {
        const key = depthPairKey(pair.source_id, pair.target_id);
        pairs.set(key, { key, source_id: pair.source_id, target_id: pair.target_id });
      }
    }
    return [...pairs.values()].sort(
      (left, right) => naturalCompare(left.source_id, right.source_id) || naturalCompare(left.target_id, right.target_id),
    );
  }, [depthData]);
  const depthPairOptions = useMemo(() => {
    if (!interactionTrackIds.size) return depthPairCatalog;
    return depthPairCatalog.filter((pair) => (
      interactionTrackIds.has(pair.source_id) || interactionTrackIds.has(pair.target_id)
    ) && (interactionTrackIds.size < 2 || (
      interactionTrackIds.has(pair.source_id) && interactionTrackIds.has(pair.target_id)
    )));
  }, [depthPairCatalog, interactionTrackIds]);
  const activeDepthPairKey = depthPairOptions.some((pair) => pair.key === selectedDepthPairKey)
    ? selectedDepthPairKey
    : depthPairOptions[0]?.key ?? "";

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

    const roiPixels = roiPolygon.map(([x, y]): Point => [x * width, y * height]);
    for (const track of renderedTracks) {
      const highlighted = track.track_id === activeHoveredContour?.id;
      const interactionSelected = interactionTrackIds.has(track.track_id);
      const color = colorFor(track.track_id);
      for (const contour of track.contours_xy ?? []) {
        if (contour.length < 3) continue;
        context.beginPath();
        context.moveTo(contour[0][0], contour[0][1]);
        for (let i = 1; i < contour.length; i += 1) context.lineTo(contour[i][0], contour[i][1]);
        context.closePath();
        if (maskMode === "fill") {
          context.save();
          context.globalAlpha = highlighted ? Math.max(maskFillOpacity, 0.38) : maskFillOpacity;
          context.fillStyle = color;
          context.fill();
          context.restore();
        }
        context.strokeStyle = highlighted ? "#ffffff" : interactionSelected ? "#5ee6a8" : color;
        context.lineWidth = highlighted || interactionSelected ? Math.max(7, width / 450) : Math.max(3, width / 900);
        context.stroke();
      }
      const labels: string[] = [];
      if (showTrackIds) labels.push(`${track.track_id} · ${track.label}`);
      const depthRank = currentDepthRanks.get(track.track_id);
      if (showDepthRanks && depthRank !== undefined) labels.push(`zᵣ ${depthRank.toFixed(2)}`);
      if (labels.length) {
        const points = track.contours_xy.flat();
        if (points.length) {
          const x = Math.min(...points.map((point) => point[0]));
          const y = Math.min(...points.map((point) => point[1]));
          const fontSize = Math.max(13, width / 120);
          const rowHeight = fontSize + 10;
          context.font = `${fontSize}px monospace`;
          labels.forEach((label, index) => {
            const labelWidth = context.measureText(label).width + 12;
            const labelX = Math.max(0, Math.min(x, width - labelWidth));
            const firstY = Math.max(0, Math.min(y - labels.length * rowHeight, height - labels.length * rowHeight));
            const labelY = firstY + index * rowHeight;
            context.fillStyle = "#071014dc";
            context.fillRect(labelX, labelY, labelWidth, rowHeight - 2);
            context.fillStyle = index === 0 ? "#edf4f7" : "#b7a6ff";
            context.fillText(label, labelX + 6, labelY + fontSize + 1);
          });
        }
      }
    }

    // Blackout is composited after every video cue so nothing outside the ROI
    // remains visible. The ROI outline/handles are drawn last for editing.
    if (roiComplete && roiBlackout && roiPixels.length >= 3) {
      context.save();
      context.beginPath();
      context.rect(0, 0, width, height);
      context.moveTo(roiPixels[0][0], roiPixels[0][1]);
      for (let index = 1; index < roiPixels.length; index += 1) {
        context.lineTo(roiPixels[index][0], roiPixels[index][1]);
      }
      context.closePath();
      context.fillStyle = "#000000";
      context.fill("evenodd");
      context.restore();
    }

    if (roiPixels.length) {
      context.save();
      context.beginPath();
      context.moveTo(roiPixels[0][0], roiPixels[0][1]);
      for (let index = 1; index < roiPixels.length; index += 1) {
        context.lineTo(roiPixels[index][0], roiPixels[index][1]);
      }
      if (roiComplete) context.closePath();
      if (roiEditing && roiComplete) {
        context.fillStyle = "#59d9ff12";
        context.fill();
      }
      context.strokeStyle = roiEditing ? "#59d9ff" : "#5ee6a8";
      context.lineWidth = Math.max(3, width / 900);
      context.setLineDash(roiComplete ? [] : [Math.max(8, width / 180), Math.max(5, width / 300)]);
      context.stroke();
      context.setLineDash([]);

      if (roiEditing) {
        const radius = Math.max(8, width / 180);
        roiPixels.forEach(([x, y], index) => {
          context.beginPath();
          context.arc(x, y, index === draggingRoiVertex ? radius * 1.35 : radius, 0, Math.PI * 2);
          context.fillStyle = index === draggingRoiVertex ? "#ffffff" : "#091014";
          context.strokeStyle = "#59d9ff";
          context.lineWidth = Math.max(3, width / 850);
          context.fill();
          context.stroke();
        });
      }
      context.restore();
    }
  }, [
    currentDepthRanks,
    draggingRoiVertex,
    activeHoveredContour?.id,
    interactionTrackIds,
    maskFillOpacity,
    maskMode,
    renderedTracks,
    roiBlackout,
    roiComplete,
    roiEditing,
    roiPolygon,
    showDepthRanks,
    showTrackIds,
    videoSize,
  ]);

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
    sourceLoadGenerationRef.current += 1;
    depthLoadGenerationRef.current += 1;
    controlWorkerRef.current?.terminate();
    controlWorkerRef.current = null;
    if (controlWorkerUrlRef.current) URL.revokeObjectURL(controlWorkerUrlRef.current);
    controlWorkerUrlRef.current = null;
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isInteractiveTarget(event.target) || event.metaKey || event.ctrlKey || event.altKey) return;
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
      const key = event.key.toLocaleLowerCase();
      if (key === "i") {
        event.preventDefault();
        if (interactionDraft) setInteractionStartTime();
        else setMessage("Create or select an interaction before marking its start.");
      }
      if (key === "o") {
        event.preventDefault();
        if (interactionDraft) setInteractionEndTime();
        else setMessage("Create or select an interaction before marking its end.");
      }
      if (key === "r") {
        event.preventDefault();
        toggleRoiEditing();
      }
      if (key === "m") {
        event.preventDefault();
        const next = !showMasks;
        setShowMasks(next);
        setHoveredContour(null);
        setMessage(next ? "Contour overlay shown." : "Contour overlay hidden.");
      }
      if (event.key === "Escape") {
        event.preventDefault();
        setInteractionTrackIds(new Set());
        setHoveredContour(null);
        if (roiEditing) {
          draggingRoiVertexRef.current = null;
          setDraggingRoiVertex(null);
          setRoiEditing(false);
        }
        setMessage(roiEditing ? "ROI editing exited and participant selection cleared." : "Participant selection cleared.");
      }
      if ((event.key === "Delete" || event.key === "Backspace") && originalInteraction) {
        event.preventDefault();
        deleteInteraction();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  buildAnnotationProjectRef.current = buildAnnotationProject;

  useEffect(() => {
    if (autosaveSuspendedRef.current || !videoUrl || !data?.frames.length || duration <= 0) return;
    const timeout = window.setTimeout(() => {
      try {
        const project = buildAnnotationProjectRef.current?.();
        if (!project) throw new Error("Project state is not ready.");
        localStorage.setItem(
          autosaveKey(videoName, jsonName, videoFileSize, contourFileSize),
          serializeAnnotationProject(project),
        );
        setAutosaveStatus(`Autosaved locally · ${new Date().toLocaleTimeString()}`);
      } catch (error) {
        setAutosaveStatus(`Autosave paused · ${error instanceof Error ? error.message : String(error)}`);
      }
    }, 600);
    return () => window.clearTimeout(timeout);
  }, [
    data,
    duration,
    fps,
    idAliases,
    interactionDraft,
    interactionTrackIds,
    interactions,
    jsonName,
    projectCreatedAt,
    roiBlackout,
    roiComplete,
    roiPolygon,
    contourFileSize,
    videoFileSize,
    videoName,
    videoSize.height,
    videoSize.width,
    videoUrl,
  ]);

  function projectFrameCount() {
    return expectedSourceFrameCount(data, duration, fps);
  }

  function projectInteractions(): ProjectInteraction[] {
    const frameCount = projectFrameCount();
    return sortedInteractions.map((interaction) => {
      if (interaction.end_time === null) {
        throw new Error(`Interaction ${interaction.interaction_id} has no end time.`);
      }
      return {
        ...interaction,
        end_time: interaction.end_time,
        start_frame: Math.min(frameCount - 1, timeToFrame(interaction.start_time, fps)),
        end_frame: Math.min(frameCount - 1, timeToFrame(interaction.end_time, fps)),
      };
    });
  }

  function buildAnnotationProject(updatedAt = new Date().toISOString()): AnnotationProject {
    if (!videoUrl || !data?.frames.length || duration <= 0) {
      throw new Error("Load a playable video and Control JSON before saving a project.");
    }
    if (interactionDraft && hasInteractionChanges) {
      throw new Error("Save or discard the current interaction draft before saving.");
    }
    if (roiPolygon.length && !roiComplete) {
      throw new Error("Complete or clear the ROI draft before saving.");
    }
    const polygon = roiComplete ? validateNormalizedRoiPolygon(roiPolygon) : null;
    return {
      schema_version: ANNOTATION_PROJECT_SCHEMA,
      source: {
        video_path: data.video || videoPath || videoName,
        contour_path: contourPath || jsonName,
        ...(videoFileSize > 0 ? { video_file_size_bytes: videoFileSize } : {}),
        ...(contourFileSize > 0 ? { contour_file_size_bytes: contourFileSize } : {}),
        video_fps: fps,
        video_width: videoSize.width,
        video_height: videoSize.height,
        frame_count: projectFrameCount(),
        duration_seconds: duration,
        available_categories: [...catalog.types],
      },
      interactions: projectInteractions(),
      roi: {
        polygon,
        blackout_enabled: Boolean(polygon && roiBlackout),
      },
      id_aliases: idAliases,
      created_at: projectCreatedAt,
      updated_at: updatedAt,
    };
  }

  function applyAnnotationProject(
    project: AnnotationProject,
    controlData: ControlData,
    selectedVideoName = videoName,
    selectedContourName = jsonName,
  ) {
    if (fileStem(project.source.video_path) !== fileStem(selectedVideoName)) {
      throw new Error(
        `project video mismatch: expected ${selectedVideoName}, received ${fileBaseName(project.source.video_path)}.`,
      );
    }
    if (fileBaseName(project.source.contour_path) !== fileBaseName(selectedContourName)) {
      throw new Error(
        `project Control mismatch: expected ${selectedContourName}, received ${fileBaseName(project.source.contour_path)}.`,
      );
    }
    if (project.source.video_file_size_bytes !== undefined && videoFileSize > 0
      && project.source.video_file_size_bytes !== videoFileSize) {
      throw new Error(
        `project video size mismatch: expected ${videoFileSize.toLocaleString()} bytes, `
        + `received ${project.source.video_file_size_bytes.toLocaleString()} bytes.`,
      );
    }
    if (project.source.contour_file_size_bytes !== undefined && contourFileSize > 0
      && project.source.contour_file_size_bytes !== contourFileSize) {
      throw new Error(
        `project Control size mismatch: expected ${contourFileSize.toLocaleString()} bytes, `
        + `received ${project.source.contour_file_size_bytes.toLocaleString()} bytes.`,
      );
    }
    if (duration > 0 && (
      project.source.video_width !== videoSize.width || project.source.video_height !== videoSize.height
    )) {
      throw new Error(
        `project dimensions mismatch: expected ${videoSize.width}×${videoSize.height}, `
        + `received ${project.source.video_width}×${project.source.video_height}.`,
      );
    }
    const durationTolerance = Math.max(0.25, 1 / project.source.video_fps);
    if (duration > 0 && Math.abs(project.source.duration_seconds - duration) > durationTolerance) {
      throw new Error(
        `project duration mismatch: expected ${duration.toFixed(3)} s, `
        + `received ${project.source.duration_seconds.toFixed(3)} s.`,
      );
    }
    const controlFps = inferControlFps(controlData);
    if (Math.abs(project.source.video_fps - controlFps) > Math.max(0.01, controlFps * 0.01)) {
      throw new Error(
        `project FPS mismatch: expected ${controlFps.toFixed(4)}, received ${project.source.video_fps.toFixed(4)}.`,
      );
    }
    const expectedFrameCount = expectedSourceFrameCount(controlData, duration, controlFps);
    if (Math.abs(project.source.frame_count - expectedFrameCount) > Math.max(2, expectedFrameCount * 0.005)) {
      throw new Error(
        `project frame-count mismatch: expected about ${expectedFrameCount}, received ${project.source.frame_count}.`,
      );
    }

    const labelsByTrack = new Map<string, string>();
    for (const frame of controlData.frames) {
      for (const track of frame.tracks) labelsByTrack.set(track.track_id, track.label);
    }
    for (const [alias, target] of Object.entries(project.id_aliases)) {
      if (!labelsByTrack.has(alias)) throw new Error(`alias source ${alias} is absent from this Control JSON.`);
      const canonical = resolveTrackId(target, project.id_aliases);
      if (!labelsByTrack.has(canonical)) throw new Error(`alias target ${canonical} is absent from this Control JSON.`);
    }
    for (const interaction of project.interactions) {
      for (const trackId of interaction.object_id_list) {
        if (!labelsByTrack.has(trackId) && !Object.hasOwn(project.id_aliases, trackId)) {
          throw new Error(`interaction ${interaction.interaction_id} references unknown track ${trackId}.`);
        }
        const canonical = resolveTrackId(trackId, project.id_aliases);
        if (!labelsByTrack.has(canonical)) {
          throw new Error(`interaction ${interaction.interaction_id} resolves to unknown track ${canonical}.`);
        }
      }
    }

    setInteractions(project.interactions.map((interaction): Interaction => ({ ...interaction })));
    setInteractionDraft(null);
    setInteractionTrackIds(new Set());
    setSelectedInteractionId(null);
    setEditingInteractionId(null);
    setIdAliases(project.id_aliases);
    setAliasSource("");
    setAliasTarget("");
    setRoiPolygon(project.roi.polygon ?? []);
    setRoiComplete(project.roi.polygon !== null);
    setRoiEditing(false);
    setRoiBlackout(project.roi.blackout_enabled);
    setProjectCreatedAt(project.created_at);
  }

  function persistAutosaveNow() {
    if (!videoUrl || !data?.frames.length || duration <= 0) return false;
    try {
      localStorage.setItem(
        autosaveKey(videoName, jsonName, videoFileSize, contourFileSize),
        serializeAnnotationProject(buildAnnotationProject()),
      );
      setAutosaveStatus("Autosaved locally before switching files");
      return true;
    } catch (error) {
      setAutosaveStatus(`Autosave failed · ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  function hasAnnotationWork() {
    return interactions.length > 0 || interactionDraft !== null || roiPolygon.length > 0 || Object.keys(idAliases).length > 0;
  }

  function confirmReplacingSession(kind: string) {
    if (!hasAnnotationWork()) return true;
    if (interactionDraft && hasInteractionChanges) {
      const detail = "Save or discard the current interaction draft before switching files.";
      window.alert(detail);
      setMessage(detail);
      return false;
    }
    const saved = persistAutosaveNow();
    const hasUnexportedInteractions = interactions.length > 0
      && interactionSignature !== exportedInteractionSignature;
    const warning = hasUnexportedInteractions
      ? "Saved interactions have not been exported."
      : `${kind} will replace the current annotations, ROI, and aliases.`;
    return window.confirm(
      `${warning} `
      + `${saved ? "A local autosave was created." : "Autosave was not available."} Continue?`,
    );
  }

  function confirmDiscardInteractionChanges(kind: string) {
    if (!interactionDraft || !hasInteractionChanges) return true;
    const shouldDiscard = window.confirm(`${kind} will discard the unsaved interaction changes. Continue?`);
    if (!shouldDiscard) setMessage(`${kind} canceled. The interaction draft is still open.`);
    return shouldDiscard;
  }

  function loadVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!confirmReplacingSession("Loading another video")) return;
    sourceLoadGenerationRef.current += 1;
    depthLoadGenerationRef.current += 1;
    controlWorkerRef.current?.terminate();
    controlWorkerRef.current = null;
    if (controlWorkerUrlRef.current) URL.revokeObjectURL(controlWorkerUrlRef.current);
    controlWorkerUrlRef.current = null;
    autosaveSuspendedRef.current = true;
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
    const url = URL.createObjectURL(file);
    videoUrlRef.current = url;
    setVideoUrl(url);
    setVideoName(file.name);
    setVideoPath(file.webkitRelativePath || file.name);
    setVideoFileSize(file.size);
    setData(null);
    setJsonName("No control file selected");
    setContourPath("");
    setContourFileSize(0);
    setLoadState("idle");
    setDepthData(null);
    setDepthName("No depth cues selected");
    setDepthLoadState("idle");
    setVisibleTrackIds(new Set());
    setInteractionTrackIds(new Set());
    setInteractions([]);
    setExportedInteractionSignature("");
    setInteractionDraft(null);
    setSelectedInteractionId(null);
    setEditingInteractionId(null);
    setSelectedDepthPairKey("");
    setInteractionType("");
    setIdAliases({});
    setAliasSource("");
    setAliasTarget("");
    setProjectName("Unsaved project");
    setProjectCreatedAt(new Date().toISOString());
    setAutosaveStatus("Autosave waits for the matching Control JSON");
    setRoiPolygon([]);
    setRoiComplete(false);
    setRoiEditing(false);
    setRoiBlackout(true);
    draggingRoiVertexRef.current = null;
    setDraggingRoiVertex(null);
    if (jsonInputRef.current) jsonInputRef.current.value = "";
    if (depthInputRef.current) depthInputRef.current.value = "";
    setMessage("Video loaded. Previous annotations were cleared; select the matching control JSON.");
    setCurrentTime(0);
    setDuration(0);
    setIsPlaying(false);
    setHoveredContour(null);
  }

  function loadJson(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!videoUrl || duration <= 0) {
      setMessage("Wait for the selected video metadata before loading its Control JSON.");
      return;
    }
    const hasControlBoundWork = interactions.length > 0
      || interactionDraft !== null
      || Object.keys(idAliases).length > 0;
    if (hasControlBoundWork && !confirmReplacingSession("Loading another Control JSON")) return;
    const loadGeneration = sourceLoadGenerationRef.current + 1;
    sourceLoadGenerationRef.current = loadGeneration;
    depthLoadGenerationRef.current += 1;
    controlWorkerRef.current?.terminate();
    controlWorkerRef.current = null;
    if (controlWorkerUrlRef.current) URL.revokeObjectURL(controlWorkerUrlRef.current);
    controlWorkerUrlRef.current = null;
    autosaveSuspendedRef.current = true;
    setJsonName(file.name);
    setContourPath(file.webkitRelativePath || file.name);
    setContourFileSize(file.size);
    setLoadState("reading");
    setData(null);
    setDepthData(null);
    setDepthName("No depth cues selected");
    setDepthLoadState("idle");
    setVisibleTrackIds(new Set());
    setInteractionTrackIds(new Set());
    setInteractions([]);
    setExportedInteractionSignature("");
    setInteractionDraft(null);
    setSelectedInteractionId(null);
    setEditingInteractionId(null);
    setSelectedDepthPairKey("");
    setInteractionType("");
    setIdAliases({});
    setAliasSource("");
    setAliasTarget("");
    setProjectName("Unsaved project");
    setProjectCreatedAt(new Date().toISOString());
    if (depthInputRef.current) depthInputRef.current.value = "";
    setMessage(`Reading ${file.name}. Large control files may take a moment…`);

    const workerSource = `
      self.onmessage = async ({ data }) => {
        try {
          const file = data.file;
          const expectedWidth = Number(data.videoWidth);
          const expectedHeight = Number(data.videoHeight);
          const raw = JSON.parse(await file.text());
          if (!raw || !Array.isArray(raw.frames)) throw new Error("frames must be an array");
          const positiveNumber = (path, ...values) => {
            let provided = false;
            for (const value of values) {
              if (value === undefined || value === null) continue;
              provided = true;
              const parsed = Number(value);
              if (Number.isFinite(parsed) && parsed > 0) return parsed;
            }
            if (provided) throw new Error(path + " must be a positive finite number");
            return undefined;
          };
          const positiveInteger = (path, ...values) => {
            let provided = false;
            for (const value of values) {
              if (value === undefined || value === null) continue;
              provided = true;
              const parsed = Number(value);
              if (Number.isSafeInteger(parsed) && parsed > 0) return parsed;
            }
            if (provided) throw new Error(path + " must be a positive safe integer");
            return undefined;
          };
          const declaredSourceFps = positiveNumber(
            "video FPS",
            raw.media?.source_fps,
            raw.video_fps,
            raw.fps,
            raw.source?.video_fps,
          );
          const declaredSampleFps = positiveNumber(
            "sample FPS",
            raw.media?.sample_fps,
            raw.sample_fps,
            raw.source?.sample_fps,
          );
          const labelsByTrack = new Map();
          const normalizeContours = (value, path) => {
            if (!Array.isArray(value)) throw new Error(path + " must be an array");
            return value.map((contour, contourIndex) => {
              if (!Array.isArray(contour)) throw new Error(path + "[" + contourIndex + "] must be an array");
              return contour.map((point, pointIndex) => {
                if (!Array.isArray(point) || point.length < 2) {
                  throw new Error(path + "[" + contourIndex + "][" + pointIndex + "] must be an [x, y] point");
                }
                const x = Number(point[0]);
                const y = Number(point[1]);
                if (!Number.isFinite(x) || !Number.isFinite(y)) {
                  throw new Error(path + " contains a non-finite contour coordinate");
                }
                if (Number.isFinite(expectedWidth) && Number.isFinite(expectedHeight)
                  && (x < 0 || y < 0 || x > expectedWidth || y > expectedHeight)) {
                  throw new Error(path + " contains point [" + x + ", " + y + "] outside "
                    + expectedWidth + "x" + expectedHeight);
                }
                return [x, y];
              });
            });
          };
          const frames = raw.frames.map((frame, ordinal) => {
            const frameIndex = Number(frame.source_frame_index ?? frame.frame_index ?? ordinal);
            if ((frame.timestamp_seconds === undefined || frame.timestamp_seconds === null)
              && declaredSourceFps === undefined) {
              throw new Error(
                "frame " + frameIndex + " has no timestamp_seconds and the Control JSON has no trusted source FPS",
              );
            }
            const timestamp = Number(
              frame.timestamp_seconds ?? frameIndex / declaredSourceFps,
            );
            if (!Number.isInteger(frameIndex) || frameIndex < 0 || !Number.isFinite(timestamp) || timestamp < 0) {
              throw new Error("each frame needs a valid frame_index and timestamp_seconds");
            }
            const sourceTracks = Array.isArray(frame.tracks)
              ? frame.tracks
              : Array.isArray(frame.instances) ? frame.instances : [];
            const trackIds = new Set();
            return {
              frame_index: frameIndex,
              timestamp_seconds: timestamp,
              tracks: sourceTracks.map((track, trackIndex) => {
                const trackId = String(track.track_id ?? track.instance_id ?? track.object_id ?? "").trim();
                if (!trackId) throw new Error("frame " + frameIndex + " track " + trackIndex + " has no ID");
                if (trackIds.has(trackId)) throw new Error("frame " + frameIndex + " contains duplicate track " + trackId);
                trackIds.add(trackId);
                const confidenceValue = track.confidence ?? track.score;
                const confidence = confidenceValue === undefined || confidenceValue === null
                  ? undefined
                  : Number(confidenceValue);
                if (confidence !== undefined && !Number.isFinite(confidence)) {
                  throw new Error("frame " + frameIndex + " track " + trackId + " has invalid confidence");
                }
                const label = String(track.label ?? track.prompt_label ?? "unknown").trim();
                if (!label) throw new Error("frame " + frameIndex + " track " + trackId + " has no label");
                const previousLabel = labelsByTrack.get(trackId);
                if (previousLabel !== undefined && previousLabel !== label) {
                  throw new Error(
                    "track " + trackId + " changes label from " + previousLabel + " to " + label,
                  );
                }
                labelsByTrack.set(trackId, label);
                const contourValue = track.contours_xy ?? track.contours ?? track.metadata?.contours_xy ?? [];
                return {
                  track_id: trackId,
                  label,
                  confidence,
                  contours_xy: normalizeContours(contourValue, "frame " + frameIndex + " track " + trackId + " contours"),
                };
              }),
            };
          }).sort((left, right) => left.frame_index - right.frame_index);
          for (let index = 1; index < frames.length; index += 1) {
            if (frames[index].frame_index === frames[index - 1].frame_index) {
              throw new Error("frames contain duplicate frame_index " + frames[index].frame_index);
            }
            if (frames[index].timestamp_seconds < frames[index - 1].timestamp_seconds) {
              throw new Error("timestamps must be non-decreasing after ordering by frame_index");
            }
          }
          let inferredSourceFps;
          for (let index = 1; index < frames.length; index += 1) {
            const frameDelta = frames[index].frame_index - frames[index - 1].frame_index;
            const timeDelta = frames[index].timestamp_seconds - frames[index - 1].timestamp_seconds;
            if (frameDelta > 0 && timeDelta > 0) {
              const candidate = frameDelta / timeDelta;
              if (Number.isFinite(candidate) && candidate > 0) {
                inferredSourceFps = candidate;
                break;
              }
            }
          }
          const resolvedSourceFps = declaredSourceFps ?? inferredSourceFps;
          if (resolvedSourceFps === undefined) {
            throw new Error(
              "Control JSON has no trusted source FPS; provide source FPS or increasing frame/time pairs",
            );
          }
          const video = typeof raw.video === "string"
            ? raw.video
            : typeof raw.input_path === "string"
              ? raw.input_path
              : typeof raw.source?.input_path === "string" ? raw.source.input_path : undefined;
          self.postMessage({
            ok: true,
            value: {
              video,
              video_fps: resolvedSourceFps,
              sample_fps: declaredSampleFps,
              video_width: positiveInteger("video width", raw.media?.width, raw.video_width, raw.source?.video_width),
              video_height: positiveInteger("video height", raw.media?.height, raw.video_height, raw.source?.video_height),
              video_frame_count: positiveInteger(
                "video frame count",
                raw.media?.source_frame_count,
                raw.video_frame_count,
                raw.source?.video_frame_count,
              ),
              video_duration_seconds: positiveNumber(
                "video duration",
                raw.media?.duration_seconds,
                raw.video_duration_seconds,
                raw.source?.video_duration_seconds,
              ),
              frames,
            },
          });
        } catch (error) {
          self.postMessage({ ok: false, error: error instanceof Error ? error.message : String(error) });
        }
      };
    `;
    const workerUrl = URL.createObjectURL(new Blob([workerSource], { type: "text/javascript" }));
    const worker = new Worker(workerUrl);
    controlWorkerRef.current = worker;
    controlWorkerUrlRef.current = workerUrl;
    worker.onmessage = (workerEvent: MessageEvent<{ ok: boolean; value?: ControlData; error?: string }>) => {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      if (controlWorkerRef.current === worker) {
        controlWorkerRef.current = null;
        controlWorkerUrlRef.current = null;
      }
      if (sourceLoadGenerationRef.current !== loadGeneration) return;
      if (!workerEvent.data.ok || !workerEvent.data.value) {
        setLoadState("error");
        setMessage(`Could not read control JSON: ${workerEvent.data.error ?? "Unknown error"}`);
        return;
      }
      const nextData = workerEvent.data.value;
      try {
        if (nextData.video && fileStem(nextData.video) !== fileStem(videoName)) {
          throw new Error(
            `video mismatch: expected ${videoName}, received ${fileBaseName(nextData.video)}.`,
          );
        }
        if (nextData.video_width && nextData.video_height && (
          nextData.video_width !== videoSize.width || nextData.video_height !== videoSize.height
        )) {
          throw new Error(
            `video dimensions mismatch: expected ${videoSize.width}×${videoSize.height}, `
            + `received ${nextData.video_width}×${nextData.video_height}.`,
          );
        }
        const controlFps = inferControlFps(nextData);
        const timeTolerance = Math.max(0.25, 1 / controlFps);
        if (!nextData.frames.length) throw new Error("frames must contain at least one annotated frame.");
        if (nextData.video_duration_seconds !== undefined
          && Math.abs(nextData.video_duration_seconds - duration) > timeTolerance) {
          throw new Error(
            `video duration mismatch: expected ${duration.toFixed(3)} s, `
            + `received ${nextData.video_duration_seconds.toFixed(3)} s.`,
          );
        }
        if (nextData.video_frame_count !== undefined) {
          const estimatedFrameCount = Math.max(1, Math.ceil(duration * controlFps));
          const frameCountTolerance = Math.max(2, Math.ceil(estimatedFrameCount * 0.01));
          if (Math.abs(nextData.video_frame_count - estimatedFrameCount) > frameCountTolerance) {
            throw new Error(
              `video frame-count mismatch: expected about ${estimatedFrameCount}, `
              + `received ${nextData.video_frame_count}.`,
            );
          }
          const lastFrameIndex = nextData.frames.at(-1)?.frame_index ?? 0;
          if (lastFrameIndex >= nextData.video_frame_count) {
            throw new Error(
              `frame ${lastFrameIndex} is outside declared source frame count ${nextData.video_frame_count}.`,
            );
          }
        }
        for (const frame of nextData.frames) {
          if (frame.timestamp_seconds > duration + timeTolerance) {
            throw new Error(
              `frame ${frame.frame_index} timestamp ${frame.timestamp_seconds.toFixed(3)} s exceeds `
              + `video duration ${duration.toFixed(3)} s.`,
            );
          }
        }
      } catch (error) {
        setLoadState("error");
        setMessage(`Control JSON does not match the selected video: ${error instanceof Error ? error.message : String(error)}`);
        return;
      }
      const tracks = new Set<string>();
      for (const frame of nextData.frames) {
        for (const track of frame.tracks) {
          tracks.add(track.track_id);
        }
      }
      setData(nextData);
      setVisibleTrackIds(tracks);
      setInteractionTrackIds(new Set());
      setLoadState("ready");
      const key = autosaveKey(videoName, file.name, videoFileSize, file.size);
      const saved = localStorage.getItem(key);
      if (saved) {
        try {
          const project = parseAnnotationProjectJson(saved);
          applyAnnotationProject(project, nextData, videoName, file.name);
          setProjectName("Recovered local autosave");
          setAutosaveStatus("Recovered the matching local autosave");
          setMessage(
            `${nextData.frames.length.toLocaleString()} annotated frames loaded · `
            + `${project.interactions.length} interactions and ROI restored from autosave.`,
          );
        } catch (error) {
          localStorage.setItem(`${key}.invalid.${Date.now()}`, saved);
          localStorage.removeItem(key);
          setAutosaveStatus("An invalid autosave was quarantined; a fresh autosave will be created");
          setMessage(
            `${nextData.frames.length.toLocaleString()} annotated frames loaded. `
            + `A stale autosave was not applied: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      } else {
        setMessage(`${nextData.frames.length.toLocaleString()} annotated frames loaded.`);
      }
      autosaveSuspendedRef.current = false;
    };
    worker.onerror = (error) => {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      if (controlWorkerRef.current === worker) {
        controlWorkerRef.current = null;
        controlWorkerUrlRef.current = null;
      }
      if (sourceLoadGenerationRef.current !== loadGeneration) return;
      setLoadState("error");
      setMessage(`Could not read control JSON: ${error.message}`);
    };
    worker.postMessage({ file, videoWidth: videoSize.width, videoHeight: videoSize.height });
  }

  async function loadDepthJson(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!data?.frames.length || !videoUrl) {
      setMessage("Load the matching video and control JSON before loading depth cues.");
      return;
    }
    const sourceLoadGeneration = sourceLoadGenerationRef.current;
    const depthLoadGeneration = depthLoadGenerationRef.current + 1;
    depthLoadGenerationRef.current = depthLoadGeneration;
    setDepthName(file.name);
    setDepthLoadState("reading");
    setDepthData(null);
    setSelectedDepthPairKey("");
    setMessage(`Validating ${file.name}…`);
    try {
      const contents = await file.text();
      if (sourceLoadGenerationRef.current !== sourceLoadGeneration
        || depthLoadGenerationRef.current !== depthLoadGeneration) return;
      const parsed = parseDepthFeatures(JSON.parse(contents));
      const sidecarVideoName = fileBaseName(parsed.video);
      if (fileStem(sidecarVideoName) !== fileStem(videoName)) {
        throw new Error(`video mismatch: expected ${videoName}, received ${fileBaseName(parsed.video)}.`);
      }
      if (parsed.contour && fileBaseName(parsed.contour) !== fileBaseName(jsonName)) {
        throw new Error(`contour mismatch: expected ${jsonName}, received ${fileBaseName(parsed.contour)}.`);
      }
      if (sidecarVideoName === fileBaseName(videoName)
        && parsed.source.video_file_size_bytes !== undefined
        && videoFileSize > 0
        && parsed.source.video_file_size_bytes !== videoFileSize) {
        throw new Error(
          `video file-size mismatch for ${videoName}: expected ${videoFileSize.toLocaleString()} bytes, `
          + `received ${parsed.source.video_file_size_bytes.toLocaleString()} bytes.`,
        );
      }
      if (parsed.source.contour_file_size_bytes !== undefined
        && contourFileSize > 0
        && parsed.source.contour_file_size_bytes !== contourFileSize) {
        throw new Error(
          `Control file-size mismatch for ${jsonName}: expected ${contourFileSize.toLocaleString()} bytes, `
          + `received ${parsed.source.contour_file_size_bytes.toLocaleString()} bytes.`,
        );
      }
      if (duration > 0 && (
        parsed.source.video_width !== videoSize.width || parsed.source.video_height !== videoSize.height
      )) {
        throw new Error(
          `video dimensions mismatch: expected ${videoSize.width}×${videoSize.height}, `
          + `received ${parsed.source.video_width}×${parsed.source.video_height}.`,
        );
      }
      const durationTolerance = Math.max(0.25, 1 / parsed.source.video_fps);
      if (duration > 0 && parsed.source.video_duration_seconds > 0
        && Math.abs(parsed.source.video_duration_seconds - duration) > durationTolerance) {
        throw new Error(
          `video duration mismatch: expected ${duration.toFixed(3)} s, `
          + `received ${parsed.source.video_duration_seconds.toFixed(3)} s.`,
        );
      }

      const controlFrames = new Map(data.frames.map((frame) => [frame.frame_index, frame]));
      const controlTrackIds = new Set(catalog.tracks.map((track) => track.id));
      for (const frame of parsed.frames) {
        const controlFrame = controlFrames.get(frame.frame_index);
        if (!controlFrame) throw new Error(`frame ${frame.frame_index} does not exist in the control JSON.`);
        const controlFrameTrackIds = new Set(controlFrame.tracks.map((track) => track.track_id));
        const error = Math.abs(controlFrame.timestamp_seconds - frame.timestamp_seconds);
        if (error > parsed.depth_metadata.max_alignment_error_seconds + Number.EPSILON) {
          throw new Error(`frame ${frame.frame_index} is misaligned by ${error.toFixed(6)} s.`);
        }
        for (const instance of frame.instances) {
          if (!controlFrameTrackIds.has(instance.track_id)) {
            throw new Error(`depth instance ${instance.track_id} is absent from control frame ${frame.frame_index}.`);
          }
        }
        for (const pair of frame.pairs) {
          if (!controlFrameTrackIds.has(pair.source_id)) {
            throw new Error(`depth source ${pair.source_id} is absent from control frame ${frame.frame_index}.`);
          }
          if (!controlFrameTrackIds.has(pair.target_id)) {
            throw new Error(`depth target ${pair.target_id} is absent from control frame ${frame.frame_index}.`);
          }
        }
      }
      for (const candidate of parsed.boundary_candidates) {
        if (!controlTrackIds.has(candidate.source_id) || !controlTrackIds.has(candidate.target_id)) {
          throw new Error(`boundary candidate ${candidate.candidate_id} references an unknown track.`);
        }
        if (duration > 0 && (candidate.start_time > duration || (candidate.end_time !== null && candidate.end_time > duration))) {
          throw new Error(`boundary candidate ${candidate.candidate_id} is outside the video duration.`);
        }
      }

      if (sourceLoadGenerationRef.current !== sourceLoadGeneration
        || depthLoadGenerationRef.current !== depthLoadGeneration) return;
      setDepthData(parsed);
      setDepthLoadState("ready");
      setMessage(`${parsed.frames.length.toLocaleString()} depth cue frames loaded · ${parsed.depth_metadata.model}.`);
    } catch (error) {
      if (sourceLoadGenerationRef.current !== sourceLoadGeneration
        || depthLoadGenerationRef.current !== depthLoadGeneration) return;
      setDepthLoadState("error");
      setDepthName("Invalid depth cues file");
      setMessage(`Could not read depth cues JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
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

  function createInteraction() {
    const type = (interactionDraft?.interaction_type ?? interactionType).trim();
    if (!type) {
      setMessage("Enter an interaction type before creating the interaction.");
      interactionTypeInputRef.current?.focus();
      return;
    }
    if (!confirmDiscardInteractionChanges("Creating another interaction")) return;
    const draft: Interaction = {
      interaction_type: type,
      interaction_id: nextInteractionId(interactions),
      object_id_list: [...interactionTrackIds].sort(naturalCompare),
      start_time: annotationTime(currentTime),
      end_time: null,
    };
    setInteractionDraft(draft);
    setEditingInteractionId(null);
    setSelectedInteractionId(null);
    setMessage(`Interaction draft created with ${draft.object_id_list.length} selected objects.`);
  }

  function setInteractionStartTime() {
    if (!interactionDraft) return;
    const startTime = annotationTime(currentTime);
    setInteractionDraft({ ...interactionDraft, start_time: startTime });
    setMessage(`Interaction start set to ${formatTime(startTime)}.`);
  }

  function setInteractionEndTime() {
    if (!interactionDraft) return;
    const endTime = annotationTime(currentTime);
    if (endTime <= interactionDraft.start_time) {
      setMessage("The interaction end time must be later than its start time.");
      return;
    }
    setInteractionDraft({ ...interactionDraft, end_time: endTime });
    setMessage(`Interaction end set to ${formatTime(endTime)}.`);
  }

  function saveInteraction() {
    if (!interactionDraft) return;

    const savedInteraction: Interaction = {
      ...interactionDraft,
      interaction_type: interactionDraft.interaction_type.trim(),
      interaction_id: interactionDraft.interaction_id.trim(),
      object_id_list: [...interactionTrackIds].sort(naturalCompare),
    };
    const validationErrors: string[] = [];
    if (!savedInteraction.interaction_type) validationErrors.push("Interaction type is required.");
    if (!savedInteraction.interaction_id.trim()) validationErrors.push("Interaction ID is required.");
    if (interactions.some((interaction) => interaction.interaction_id === savedInteraction.interaction_id
      && interaction.interaction_id !== editingInteractionId)) {
      validationErrors.push(`Interaction ID ${savedInteraction.interaction_id} is already in use.`);
    }
    if (!savedInteraction.object_id_list.length) validationErrors.push("Select at least one object.");
    const labelsByTrack = new Map(catalog.tracks.map((track) => [track.id, track.label]));
    const selectedLabels = savedInteraction.object_id_list
      .map((trackId) => labelsByTrack.get(trackId))
      .filter((label): label is string => Boolean(label));
    if (!selectedLabels.some(isPersonTrackLabel)) validationErrors.push("Select at least one person participant.");
    if (savedInteraction.interaction_type.toLocaleLowerCase().includes("table")
      && !selectedLabels.some(isTableTrackLabel)) {
      validationErrors.push("Table interactions must include a table track.");
    }
    if (!Number.isFinite(savedInteraction.start_time)) validationErrors.push("Start time is required.");
    if (savedInteraction.end_time === null || !Number.isFinite(savedInteraction.end_time)) {
      validationErrors.push("End time is required.");
    } else if (savedInteraction.end_time <= savedInteraction.start_time) {
      validationErrors.push("End time must be later than start time.");
    }
    if (!data?.frames.length) validationErrors.push("Control JSON is required to validate contour visibility.");

    if (!validationErrors.length && savedInteraction.interaction_type.toLocaleLowerCase().includes("table")) {
      try {
        serializeMinimalInteractions(
          [{ ...savedInteraction, end_time: savedInteraction.end_time! }],
          catalog.tracks,
          idAliases,
        );
      } catch (error) {
        validationErrors.push(
          `Training export validation failed: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    }

    if (validationErrors.length) {
      window.alert(`Cannot save this interaction:\n\n${validationErrors.join("\n")}`);
      setMessage(validationErrors[0]);
      return;
    }

    const startFrame = nearestFrame(data!.frames, savedInteraction.start_time)!;
    const endFrame = nearestFrame(data!.frames, savedInteraction.end_time!)!;
    const firstFrameIndex = Math.min(startFrame.frame_index, endFrame.frame_index);
    const lastFrameIndex = Math.max(startFrame.frame_index, endFrame.frame_index);
    const framesInWindow = data!.frames.filter(
      (frame) => frame.frame_index >= firstFrameIndex && frame.frame_index <= lastFrameIndex,
    );
    const missingCoverage = savedInteraction.object_id_list.map((objectId) => ({
      objectId,
      missingFrames: framesInWindow.filter((frame) => {
        const track = frame.tracks.find((candidate) => candidate.track_id === objectId);
        return !track || !track.contours_xy.some((contour) => contour.length >= 3);
      }).length,
    })).filter((item) => item.missingFrames > 0);

    if (missingCoverage.length) {
      const preview = missingCoverage.slice(0, 8)
        .map((item) => `${item.objectId}: missing on ${item.missingFrames} of ${framesInWindow.length} frames`)
        .join("\n");
      const remainder = missingCoverage.length > 8 ? `\n…and ${missingCoverage.length - 8} more objects` : "";
      const shouldSave = window.confirm(
        `Some selected objects are not visible throughout frames ${firstFrameIndex}–${lastFrameIndex}:\n\n${preview}${remainder}\n\nIs this intended? Choose Cancel to return to editing.`,
      );
      if (!shouldSave) {
        setMessage("Save canceled. The interaction is still open for editing.");
        return;
      }
    }

    setInteractions((current) => editingInteractionId
      ? current.map((interaction) => interaction.interaction_id === editingInteractionId ? savedInteraction : interaction)
      : [...current, savedInteraction]);
    setSelectedInteractionId(savedInteraction.interaction_id);
    setEditingInteractionId(savedInteraction.interaction_id);
    setInteractionDraft(cloneInteraction(savedInteraction));
    setInteractionType("");
    setMessage(`Saved interaction ${savedInteraction.interaction_type}.`);
  }

  function discardInteraction() {
    if (originalInteraction) {
      setInteractionDraft(cloneInteraction(originalInteraction));
      setInteractionTrackIds(new Set(originalInteraction.object_id_list));
      setMessage("Unsaved interaction changes discarded.");
      return;
    }
    setInteractionDraft(null);
    setEditingInteractionId(null);
    setInteractionType("");
    setMessage("Interaction draft discarded.");
  }

  function deleteInteraction() {
    if (!originalInteraction) return;
    if (!window.confirm(`Delete ${originalInteraction.interaction_id} (${originalInteraction.interaction_type})?`)) {
      setMessage("Delete canceled.");
      return;
    }
    setInteractions((current) => current.filter(
      (interaction) => interaction.interaction_id !== originalInteraction.interaction_id,
    ));
    setInteractionDraft(null);
    setEditingInteractionId(null);
    setSelectedInteractionId(null);
    setInteractionTrackIds(new Set());
    setMessage(`Deleted interaction ${originalInteraction.interaction_id}.`);
  }

  function exportInteractions() {
    if (interactionDraft && hasInteractionChanges) {
      const detail = "Save or discard the current interaction draft before exporting.";
      window.alert(detail);
      setMessage(detail);
      return;
    }
    try {
      const contents = serializeMinimalInteractions(projectInteractions(), catalog.tracks, idAliases);
      const baseName = fileStem(videoName) || "interactions";
      downloadJsonText(contents, `${baseName}.events-minimal.json`);
      setExportedInteractionSignature(interactionSignature);
      setMessage(`Exported ${sortedInteractions.length} canonical interactions in the exact training schema.`);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      window.alert(`Cannot export minimal interaction JSON:\n\n${detail}`);
      setMessage(`Minimal export failed: ${detail}`);
    }
  }

  function saveProject() {
    try {
      const updatedAt = new Date().toISOString();
      const project = buildAnnotationProject(updatedAt);
      const contents = serializeAnnotationProject(project);
      const filename = `${fileStem(videoName) || "annotation"}.palona-project.json`;
      localStorage.setItem(autosaveKey(videoName, jsonName, videoFileSize, contourFileSize), contents);
      downloadJsonText(contents, filename);
      setProjectName(filename);
      setAutosaveStatus(`Project and local autosave updated · ${new Date().toLocaleTimeString()}`);
      setMessage(
        `Saved ${project.interactions.length} interactions, `
        + `${project.roi.polygon?.length ?? 0}-point ROI, and ${Object.keys(project.id_aliases).length} aliases.`,
      );
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      window.alert(`Cannot save project:\n\n${detail}`);
      setMessage(`Project save failed: ${detail}`);
    }
  }

  async function loadProject(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!videoUrl || !data?.frames.length || duration <= 0) {
      setMessage("Load the matching playable video and Control JSON before reopening a project.");
      return;
    }
    if (!confirmReplacingSession("Opening a project")) return;
    try {
      const project = parseAnnotationProjectJson(await file.text());
      applyAnnotationProject(project, data);
      const contents = serializeAnnotationProject(project);
      localStorage.setItem(autosaveKey(videoName, jsonName, videoFileSize, contourFileSize), contents);
      autosaveSuspendedRef.current = false;
      setProjectName(file.name);
      setAutosaveStatus("Opened project and refreshed the matching local autosave");
      setMessage(
        `Reopened ${file.name} · ${project.interactions.length} interactions · `
        + `${project.roi.polygon?.length ?? 0}-point ROI · ${Object.keys(project.id_aliases).length} aliases.`,
      );
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      setMessage(`Could not open project: ${detail}`);
    }
  }

  function addIdAlias() {
    if (!aliasSource || !aliasTarget) {
      setMessage("Choose both an alias ID and its canonical target.");
      return;
    }
    if (aliasSource === aliasTarget) {
      setMessage("An ID cannot alias itself.");
      return;
    }
    try {
      const labels = new Map(catalog.tracks.map((track) => [track.id, track.label]));
      const canonicalTarget = resolveTrackId(aliasTarget, idAliases);
      const sourceLabel = labels.get(aliasSource);
      const targetLabel = labels.get(canonicalTarget);
      const labelKind = (label: string) => isPersonTrackLabel(label)
        ? "person"
        : isTableTrackLabel(label) ? "table" : `other:${label.toLocaleLowerCase("en-US")}`;
      if (!sourceLabel || !targetLabel) {
        throw new Error("Both IDs must exist in the loaded Control JSON.");
      }
      if (labelKind(sourceLabel) !== labelKind(targetLabel)) {
        throw new Error(
          `Cross-category aliases are unsafe (${sourceLabel} → ${targetLabel}); choose an ID of the same semantic class.`,
        );
      }
      setIdAliases(withIdAlias(idAliases, aliasSource, aliasTarget));
      setAliasSource("");
      setAliasTarget("");
      setMessage(`Alias saved: ${aliasSource} → ${resolveTrackId(aliasTarget, idAliases)}.`);
    } catch (error) {
      setMessage(`Could not create alias: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  function deleteIdAlias(alias: string) {
    setIdAliases(withoutIdAlias(idAliases, alias));
    setMessage(`Alias ${alias} removed. Existing interactions keep their original source IDs.`);
  }

  function selectInteraction(interaction: Interaction) {
    if (!confirmDiscardInteractionChanges(`Opening ${interaction.interaction_id}`)) return;
    setInteractionDraft(cloneInteraction(interaction));
    setEditingInteractionId(interaction.interaction_id);
    setSelectedInteractionId(interaction.interaction_id);
    setInteractionTrackIds(new Set(interaction.object_id_list));
    seek(interaction.start_time);
    setMessage(`Jumped to the start of ${interaction.interaction_type}.`);
  }

  function toggleType(type: string) {
    const trackIds = catalog.tracks.filter((track) => track.label === type).map((track) => track.id);
    const shouldSelect = trackIds.some((id) => !visibleTrackIds.has(id));
    setVisibleTrackIds((current) => {
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
    return trackIds.length > 0 && trackIds.every((id) => visibleTrackIds.has(id));
  }

  function toggleTrack(id: string) {
    setVisibleTrackIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setHoveredContour(null);
  }

  function toggleInteractionTrack(id: string) {
    setInteractionTrackIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setHoveredContour(null);
  }

  function normalizedCanvasPoint(event: MouseEvent<HTMLCanvasElement> | ReactPointerEvent<HTMLCanvasElement>): Point {
    const canvas = canvasRef.current;
    if (!canvas) return [0, 0];
    const bounds = canvas.getBoundingClientRect();
    return [
      Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
    ];
  }

  function roiVertexAt(event: ReactPointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const bounds = canvas.getBoundingClientRect();
    const [x, y] = normalizedCanvasPoint(event);
    let nearestIndex: number | null = null;
    let nearestDistance = 15;
    roiPolygon.forEach(([vertexX, vertexY], index) => {
      const distance = Math.hypot((vertexX - x) * bounds.width, (vertexY - y) * bounds.height);
      if (distance <= nearestDistance) {
        nearestDistance = distance;
        nearestIndex = index;
      }
    });
    return nearestIndex;
  }

  function toggleRoiEditing() {
    if (!videoUrl) {
      setMessage("Load a video before drawing an ROI.");
      return;
    }
    if (roiEditing) {
      draggingRoiVertexRef.current = null;
      setDraggingRoiVertex(null);
      setRoiEditing(false);
      setMessage(roiComplete
        ? "ROI editing exited. The completed polygon was kept."
        : `ROI draft paused with ${roiPolygon.length} point${roiPolygon.length === 1 ? "" : "s"}.`);
      return;
    }
    videoRef.current?.pause();
    setHoveredContour(null);
    setRoiEditing(true);
    setMessage(roiComplete
      ? "ROI editing enabled. Drag a vertex, then finish editing."
      : "ROI editing enabled. Click the video to add normalized polygon points.");
  }

  function completeRoi() {
    if (roiPolygon.length < 3) {
      setMessage("An ROI polygon needs at least three points before it can be completed.");
      return;
    }
    try {
      validateNormalizedRoiPolygon(roiPolygon);
    } catch (error) {
      setMessage(`ROI is invalid: ${error instanceof Error ? error.message : String(error)}`);
      return;
    }
    draggingRoiVertexRef.current = null;
    setDraggingRoiVertex(null);
    if (!roiComplete) setRoiBlackout(true);
    setRoiComplete(true);
    setRoiEditing(false);
    setMessage(`ROI completed with ${roiPolygon.length} normalized vertices.`);
  }

  function clearRoi() {
    if (!roiPolygon.length) return;
    if (!window.confirm("Clear the current ROI polygon? This cannot be undone.")) {
      setMessage("ROI clear canceled.");
      return;
    }
    draggingRoiVertexRef.current = null;
    setDraggingRoiVertex(null);
    setRoiPolygon([]);
    setRoiComplete(false);
    setRoiEditing(false);
    setRoiBlackout(true);
    setMessage("ROI polygon cleared.");
  }

  function handleCanvasPointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!roiEditing) return;
    event.preventDefault();
    const vertexIndex = roiVertexAt(event);
    if (vertexIndex !== null) {
      draggingRoiVertexRef.current = vertexIndex;
      setDraggingRoiVertex(vertexIndex);
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    if (roiComplete) {
      setMessage("Drag an existing ROI vertex. Clear the polygon to draw a new shape.");
      return;
    }
    const point = normalizedCanvasPoint(event);
    setRoiPolygon((current) => [...current, point]);
    setMessage(`ROI point ${roiPolygon.length + 1} added at [${point[0].toFixed(3)}, ${point[1].toFixed(3)}].`);
  }

  function handleCanvasPointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!roiEditing) {
      setHoveredContour(hitTest(event));
      return;
    }
    setHoveredContour(null);
    const vertexIndex = draggingRoiVertexRef.current;
    if (vertexIndex === null) return;
    event.preventDefault();
    const point = normalizedCanvasPoint(event);
    setRoiPolygon((current) => {
      const next = current.map((vertex, index) => index === vertexIndex ? point : vertex);
      if (!roiComplete) return next;
      try {
        return validateNormalizedRoiPolygon(next);
      } catch {
        // Keep the last valid completed polygon while the pointer crosses an
        // edge or collapses its area. This makes every editable saved state valid.
        return current;
      }
    });
  }

  function finishCanvasPointer(event: ReactPointerEvent<HTMLCanvasElement>) {
    const movedVertex = draggingRoiVertexRef.current;
    if (movedVertex !== null) setMessage(`ROI vertex ${movedVertex + 1} moved.`);
    draggingRoiVertexRef.current = null;
    setDraggingRoiVertex(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function hitTest(event: MouseEvent<HTMLCanvasElement> | ReactPointerEvent<HTMLCanvasElement>): HoveredContour | null {
    const canvas = canvasRef.current;
    if (!canvas || !showMasks) return null;
    const bounds = canvas.getBoundingClientRect();
    const normalizedPoint: Point = [
      (event.clientX - bounds.left) / bounds.width,
      (event.clientY - bounds.top) / bounds.height,
    ];
    if (roiComplete && roiBlackout && !pointInPolygon(normalizedPoint, roiPolygon)) return null;
    const point: Point = [normalizedPoint[0] * canvas.width, normalizedPoint[1] * canvas.height];
    const hits: { id: string; label: string; area: number }[] = [];
    for (const track of visibleTracks) {
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
      frame_index: currentFrame?.frame_index ?? -1,
      x: Math.min(event.clientX - bounds.left + 12, Math.max(12, bounds.width - 130)),
      y: Math.min(event.clientY - bounds.top + 12, Math.max(12, bounds.height - 34)),
    };
  }

  function toggleTrackFromCanvas(contour: HoveredContour) {
    toggleInteractionTrack(contour.id);
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
        <button
          className="file-card"
          onClick={() => jsonInputRef.current?.click()}
          disabled={!videoUrl || duration <= 0 || loadState === "reading"}
        >
          <span className="file-icon json">{`{ }`}</span>
          <span><strong>Control JSON</strong><small>{jsonName}</small></span>
          <b>{loadState === "reading" ? "Reading…" : "Choose"}</b>
        </button>
        <button
          className="file-card"
          onClick={() => depthInputRef.current?.click()}
          disabled={!videoUrl || !data || depthLoadState === "reading"}
        >
          <span className="file-icon depth">Z</span>
          <span><strong>Depth cues JSON</strong><small>{depthName}</small></span>
          <b>{depthLoadState === "reading" ? "Reading…" : depthLoadState === "ready" ? "Replace" : "Choose"}</b>
        </button>
        <button
          className="file-card"
          onClick={() => projectInputRef.current?.click()}
          disabled={!videoUrl || !data?.frames.length || duration <= 0}
        >
          <span className="file-icon project">P</span>
          <span><strong>Annotation project</strong><small>{projectName}</small></span>
          <b>Open</b>
        </button>
        <input ref={videoInputRef} className="visually-hidden" type="file" accept="video/*,.mkv" onChange={loadVideo} />
        <input ref={jsonInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={loadJson} />
        <input ref={depthInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={loadDepthJson} />
        <input ref={projectInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={loadProject} />
        <p className="session-message">{message}</p>
      </section>

      <section className="workspace">
        <div className="viewer-panel">
          <div className="viewer-head">
            <div><span className="eyebrow">FRAME VIEWER</span><strong>{videoName}</strong></div>
            <div className="frame-readout"><span>FRAME</span><b>{currentFrame?.frame_index ?? "—"}</b><span>VISIBLE</span><b>{renderedTracks.length}</b><span>USED</span><b>{interactionTrackIds.size}</b></div>
          </div>

          <div className={`stage-wrap ${videoUrl ? "has-video" : ""}`} style={{ aspectRatio: `${videoSize.width} / ${videoSize.height}` }}>
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
                    video.playbackRate = playbackRate;
                    setMessage(`Video ready · ${video.videoWidth}×${video.videoHeight}`);
                  }}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onSeeked={(event) => setCurrentTime(event.currentTarget.currentTime)}
                  onError={() => setMessage("This browser could not decode the selected video. MKV/HEVC support varies by browser.")}
                />
                <canvas
                  ref={canvasRef}
                  className={`${roiEditing ? "roi-editing" : ""} ${draggingRoiVertex !== null ? "roi-dragging" : ""}`}
                  aria-label={roiEditing ? "ROI polygon editor" : "Interactive contour overlay"}
                  aria-describedby={roiEditing ? "roi-edit-instructions" : undefined}
                  onPointerDown={handleCanvasPointerDown}
                  onPointerMove={handleCanvasPointerMove}
                  onPointerUp={finishCanvasPointer}
                  onPointerCancel={finishCanvasPointer}
                  onPointerLeave={() => setHoveredContour(null)}
                  onClick={(event) => {
                    if (roiEditing) return;
                    const contour = hitTest(event);
                    if (contour) toggleTrackFromCanvas(contour);
                  }}
                />
                {roiEditing && (
                  <div id="roi-edit-instructions" className="roi-edit-hint">
                    {roiComplete
                      ? "Drag a vertex to reshape the completed ROI"
                      : `Click to add points · ${roiPolygon.length} point${roiPolygon.length === 1 ? "" : "s"}`}
                  </div>
                )}
                {activeHoveredContour && !roiEditing && (
                  <div className="hover-label" style={{ left: activeHoveredContour.x, top: activeHoveredContour.y }}>
                    {activeHoveredContour.id} {activeHoveredContour.label}
                    {currentDepthRanks.has(activeHoveredContour.id) && ` · zᵣ ${currentDepthRanks.get(activeHoveredContour.id)!.toFixed(2)}`}
                    {interactionTrackIds.has(activeHoveredContour.id) ? " · click to remove" : " · click to use"}
                  </div>
                )}
              </>
            ) : (
              <div className="empty-stage"><span>▶</span><strong>Load a local video clip</strong><p>MKV, MP4, MOV, or WebM</p></div>
            )}
          </div>

          <div className="viewer-tools" aria-label="Overlay and ROI controls">
            <div className="tool-group mask-tools">
              <span className="tool-group-title">Masks</span>
              <button
                type="button"
                className={showMasks ? "tool-toggle active" : "tool-toggle"}
                aria-pressed={showMasks}
                onClick={() => {
                  setShowMasks((current) => !current);
                  setHoveredContour(null);
                }}
              >{showMasks ? "Shown" : "Hidden"}</button>
              <label className="compact-control">
                <span>Mode</span>
                <select value={maskMode} onChange={(event) => setMaskMode(event.target.value as MaskMode)} disabled={!showMasks}>
                  <option value="fill">Filled</option>
                  <option value="contour">Contour only</option>
                </select>
              </label>
              <label className="opacity-control">
                <span>Fill opacity</span>
                <input
                  type="range"
                  min="0"
                  max="0.65"
                  step="0.01"
                  value={maskFillOpacity}
                  onChange={(event) => setMaskFillOpacity(Number(event.target.value))}
                  disabled={!showMasks || maskMode === "contour"}
                />
                <output>{Math.round(maskFillOpacity * 100)}%</output>
              </label>
              <label className="tool-check">
                <input type="checkbox" checked={showTrackIds} onChange={(event) => setShowTrackIds(event.target.checked)} disabled={!showMasks} />
                IDs
              </label>
            </div>

            <div className="tool-group roi-tools">
              <span className="tool-group-title">ROI</span>
              <span className={`roi-status ${roiComplete ? "ready" : roiPolygon.length ? "draft" : ""}`}>
                {roiComplete ? `${roiPolygon.length} points` : roiPolygon.length ? `${roiPolygon.length} point draft` : "Not set"}
              </span>
              <button
                type="button"
                className={roiEditing ? "tool-toggle active" : "tool-toggle"}
                aria-pressed={roiEditing}
                onClick={toggleRoiEditing}
                disabled={!videoUrl}
              >{roiEditing ? "Exit edit" : roiComplete ? "Edit ROI" : roiPolygon.length ? "Resume ROI" : "Draw ROI"}</button>
              <button type="button" onClick={completeRoi} disabled={!roiEditing || roiPolygon.length < 3}>
                {roiComplete ? "Finish edit" : "Complete"}
              </button>
              <button type="button" className="clear-roi" onClick={clearRoi} disabled={!roiPolygon.length}>Clear</button>
              <label className="tool-check blackout-toggle">
                <input
                  type="checkbox"
                  checked={roiBlackout}
                  onChange={(event) => setRoiBlackout(event.target.checked)}
                  disabled={!roiComplete}
                />
                Blackout outside
              </label>
            </div>
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
            <label className="speed-control">
              <span className="visually-hidden">Playback speed</span>
              <select
                aria-label="Playback speed"
                value={playbackRate}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  setPlaybackRate(next);
                  if (videoRef.current) videoRef.current.playbackRate = next;
                }}
                disabled={!videoUrl}
              >
                {[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2].map((rate) => (
                  <option key={rate} value={rate}>{rate}×</option>
                ))}
              </select>
            </label>
          </div>
          <div className="event-timeline-strip" aria-label="Saved interaction timeline">
            <div className="event-timeline-track">
              {duration > 0 && sortedInteractions.map((interaction, index) => {
                const start = Math.max(0, Math.min(duration, interaction.start_time));
                const end = Math.max(start, Math.min(duration, interaction.end_time ?? duration));
                return (
                  <button
                    key={interaction.interaction_id}
                    type="button"
                    className={selectedInteractionId === interaction.interaction_id ? "selected" : ""}
                    style={{
                      left: `${(start / duration) * 100}%`,
                      width: `${Math.max(0.6, ((end - start) / duration) * 100)}%`,
                      top: `${4 + (index % 3) * 8}px`,
                    }}
                    title={`${interaction.interaction_type} · ${formatTime(start)} → ${formatTime(end)}`}
                    aria-label={`Open ${interaction.interaction_type} at ${formatTime(start)}`}
                    onClick={() => selectInteraction(interaction)}
                  />
                );
              })}
              {duration > 0 && (
                <span className="timeline-playhead" style={{ left: `${Math.min(100, (currentTime / duration) * 100)}%` }} />
              )}
            </div>
          </div>
          <div className="shortcut-help" role="note" aria-label="Keyboard shortcuts">
            <span><kbd>Space</kbd> play / pause</span>
            <span><kbd>←</kbd><kbd>→</kbd> frame</span>
            <span><kbd>I</kbd> mark start</span>
            <span><kbd>O</kbd> mark end</span>
            <span><kbd>R</kbd> ROI edit</span>
            <span><kbd>M</kbd> masks</span>
            <span><kbd>Esc</kbd> clear selection</span>
            <span><kbd>Delete</kbd> delete event</span>
          </div>
        </div>

        <div className="side-column">
        <aside className="inspector">
          <div className="inspector-title">
            <div><span className="eyebrow">OVERLAY FILTERS</span><h2>Contours</h2></div>
            <div className="inspector-title-actions">
              <span>{visibleTrackIds.size}/{catalog.tracks.length}</span>
              <button
                onClick={() => setInteractionTrackIds(new Set())}
                disabled={!interactionTrackIds.size}
              >Clear selection</button>
            </div>
          </div>

          <section className="filter-section">
            <div className="section-heading"><h3>Object type</h3><div><button onClick={() => setVisibleTrackIds(new Set(catalog.tracks.map((track) => track.id)))}>All</button><button onClick={() => setVisibleTrackIds(new Set())}>None</button></div></div>
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
            <div className="section-heading"><h3>Track ID · visibility / event</h3><div><button onClick={() => setVisibleTrackIds(new Set(catalog.tracks.map((track) => track.id)))}>All</button><button onClick={() => setVisibleTrackIds(new Set())}>None</button></div></div>
            <div className="track-list">
              {catalog.tracks.map((track) => (
                <div key={track.id} className={`track-row ${visibleTrackIds.has(track.id) ? "checked" : ""} ${activeHoveredContour?.id === track.id ? "hovered" : ""} ${interactionTrackIds.has(track.id) ? "in-interaction" : ""}`}>
                  <label>
                    <input type="checkbox" checked={visibleTrackIds.has(track.id)} onChange={() => toggleTrack(track.id)} />
                    <span className="color-dot" style={{ background: colorFor(track.id) }} />
                    <span>
                      <strong>{track.id}</strong>
                      <small>
                        {track.label}
                        {canonicalTrackIds.get(track.id) !== track.id && ` · → ${canonicalTrackIds.get(track.id)}`}
                      </small>
                    </span>
                    <em>{track.count.toLocaleString()}f</em>
                    <span className="checkmark">✓</span>
                  </label>
                  <button
                    type="button"
                    className="use-track"
                    aria-pressed={interactionTrackIds.has(track.id)}
                    aria-label={`${interactionTrackIds.has(track.id) ? "Remove" : "Use"} ${track.id} ${interactionTrackIds.has(track.id) ? "from" : "in"} interaction`}
                    onClick={() => toggleInteractionTrack(track.id)}
                  >
                    {interactionTrackIds.has(track.id) ? "Using" : "Use"}
                  </button>
                </div>
              ))}
            </div>
          </section>

          <div className="current-summary">
            <span className="pulse-dot" />
            <div><strong>{interactionTrackIds.size} event participant{interactionTrackIds.size === 1 ? "" : "s"}</strong><small>{currentFrame ? `${currentFrame.tracks.length} tracks · ${formatTime(currentFrame.timestamp_seconds)}` : data ? "Outside Control coverage" : "No annotation data"}</small></div>
          </div>
        </aside>

        <DepthEvidencePanel
          data={depthData}
          alignment={depthAlignment}
          pairOptions={depthPairOptions}
          selectedPairKey={activeDepthPairKey}
          duration={duration}
          showInstanceRanks={showDepthRanks}
          onPairChange={setSelectedDepthPairKey}
          onToggleInstanceRanks={() => setShowDepthRanks((current) => !current)}
          onSeek={seek}
        />

        <section className="interactions-panel" aria-label="Interactions">
          <div className="interactions-title">
            <div><span className="eyebrow">ANNOTATION EVENTS</span><h2>Interactions</h2></div>
            <div className="interactions-title-actions">
              <span>{interactions.length}</span>
              <button onClick={saveProject} disabled={!videoUrl || !data?.frames.length || duration <= 0}>Save project</button>
              <button
                onClick={exportInteractions}
                disabled={!interactions.length || Boolean(interactionDraft && hasInteractionChanges)}
              >Export minimal</button>
            </div>
          </div>

          <div className="project-status">
            <strong>{projectName}</strong>
            <span>{autosaveStatus}</span>
            <small>Project files contain labels, ROI, and aliases only—never the source video.</small>
          </div>

          <div className="interaction-create">
            <label htmlFor="interaction-type">Interaction type</label>
            <div>
              <input
                ref={interactionTypeInputRef}
                id="interaction-type"
                list="interaction-type-options"
                value={interactionDraft?.interaction_type ?? interactionType}
                onChange={(event) => {
                  if (interactionDraft) {
                    setInteractionDraft({ ...interactionDraft, interaction_type: event.target.value });
                  } else {
                    setInteractionType(event.target.value);
                  }
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") createInteraction();
                }}
                placeholder="e.g. serving_food"
              />
              <datalist id="interaction-type-options">
                {interactionTypes.map((type) => <option key={type} value={type} />)}
              </datalist>
              <button className="primary-action" onClick={createInteraction}>Create interaction</button>
            </div>
          </div>

          <div className="alias-editor">
            <div className="section-heading"><h3>ID aliases / manual merge</h3><span>{Object.keys(idAliases).length}</span></div>
            <div className="alias-form">
              <select aria-label="Alias source track" value={aliasSource} onChange={(event) => setAliasSource(event.target.value)}>
                <option value="">Alias ID…</option>
                {catalog.tracks.map((track) => <option key={track.id} value={track.id}>{track.id} · {track.label}</option>)}
              </select>
              <span>→</span>
              <select aria-label="Canonical target track" value={aliasTarget} onChange={(event) => setAliasTarget(event.target.value)}>
                <option value="">Canonical ID…</option>
                {catalog.tracks.filter((track) => track.id !== aliasSource).map((track) => (
                  <option key={track.id} value={track.id}>{track.id} · {track.label}</option>
                ))}
              </select>
              <button type="button" onClick={addIdAlias} disabled={!aliasSource || !aliasTarget}>Merge</button>
            </div>
            {Object.keys(idAliases).length > 0 && (
              <div className="alias-list">
                {Object.entries(idAliases).map(([alias, target]) => (
                  <div key={alias}>
                    <code>{alias} → {resolveTrackId(target, idAliases)}</code>
                    <button type="button" onClick={() => deleteIdAlias(alias)}>Remove</button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {displayedInteraction ? (
            <div className="interaction-detail">
              <div className="detail-heading">
                <span className={originalInteraction ? "saved-badge" : "draft-badge"}>{originalInteraction ? "Editing" : "Draft"}</span>
                <strong>{displayedInteraction.interaction_type}</strong>
              </div>
              <label className="interaction-id-field" htmlFor="interaction-id">
                <span>Interaction ID</span>
                <input
                  id="interaction-id"
                  value={interactionDraft?.interaction_id ?? displayedInteraction.interaction_id}
                  onChange={(event) => {
                    if (interactionDraft) {
                      setInteractionDraft({ ...interactionDraft, interaction_id: event.target.value });
                    }
                  }}
                />
              </label>
              <pre>{JSON.stringify(displayedInteraction, null, 2)}</pre>
              <div className="interaction-actions">
                <button onClick={() => seek(displayedInteraction.start_time)}>Jump to start</button>
                <button onClick={() => displayedInteraction.end_time !== null && seek(displayedInteraction.end_time)} disabled={displayedInteraction.end_time === null}>Jump to end</button>
                <button onClick={setInteractionStartTime}>Use current time as start</button>
                <button onClick={setInteractionEndTime}>Use current time as end</button>
              </div>
              <div className="interaction-actions">
                <button className="save-action" onClick={saveInteraction} disabled={!hasInteractionChanges}>Save</button>
                <button onClick={discardInteraction}>{originalInteraction ? "Discard changes" : "Discard"}</button>
                {originalInteraction && <button className="danger-action" onClick={deleteInteraction}>Delete</button>}
              </div>
            </div>
          ) : (
            <div className="interaction-empty">Create an interaction from the currently selected object IDs and video time.</div>
          )}

          <div className="defined-interactions">
            <div className="section-heading"><h3>Defined interactions</h3></div>
            <div className="interaction-list">
              {sortedInteractions.length ? sortedInteractions.map((interaction) => (
                <button
                  key={interaction.interaction_id}
                  className={selectedInteractionId === interaction.interaction_id ? "selected" : ""}
                  onClick={() => selectInteraction(interaction)}
                >
                  <span><strong>{interaction.interaction_type}</strong><small>{interaction.interaction_id}</small></span>
                  <em>{formatTime(interaction.start_time)} → {interaction.end_time === null ? "—" : formatTime(interaction.end_time)}</em>
                </button>
              )) : <p>No saved interactions yet.</p>}
            </div>
          </div>
        </section>
        </div>
      </section>
    </main>
  );
}
