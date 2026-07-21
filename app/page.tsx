"use client";

import { ChangeEvent, MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Point = [number, number];
type Track = {
  track_id: string;
  label: string;
  confidence?: number;
  bbox_xyxy?: [number, number, number, number];
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
type Interaction = {
  interaction_type: string;
  interaction_id: string;
  object_id_list: string[];
  start_time: number;
  end_time: number | null;
};

const PALETTE = ["#59d9ff", "#ffcb52", "#a78bfa", "#5ee6a8", "#ff7e9d", "#fb923c", "#67e8f9"];
const NATURAL_COLLATOR = new Intl.Collator("en", { numeric: true, sensitivity: "base" });
const STEP_HOLD_DELAY_MS = 350;
const STEP_REPEAT_MS = 100;

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

function frameAtOrBefore(frames: Frame[], time: number) {
  if (!frames.length) return null;
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (frames[middle].timestamp_seconds <= time) low = middle;
    else high = middle - 1;
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
  const interactionTypeInputRef = useRef<HTMLInputElement>(null);
  const videoUrlRef = useRef<string | null>(null);
  const stepHoldDelayRef = useRef<number | null>(null);
  const stepRepeatRef = useRef<number | null>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState("No video selected");
  const [videoPath, setVideoPath] = useState("");
  const [jsonName, setJsonName] = useState("No control file selected");
  const [contourPath, setContourPath] = useState("");
  const [data, setData] = useState<ControlData | null>(null);
  const [loadState, setLoadState] = useState<"idle" | "reading" | "ready" | "error">("idle");
  const [message, setMessage] = useState("Choose a video and its control JSON to begin.");
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [presentedTime, setPresentedTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoSize, setVideoSize] = useState({ width: 16, height: 9 });
  const [selectedTracks, setSelectedTracks] = useState<Set<string>>(new Set());
  const [hoveredContour, setHoveredContour] = useState<HoveredContour | null>(null);
  const [interactionType, setInteractionType] = useState("");
  const [interactionDraft, setInteractionDraft] = useState<Interaction | null>(null);
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [selectedInteractionId, setSelectedInteractionId] = useState<string | null>(null);
  const [editingInteractionId, setEditingInteractionId] = useState<string | null>(null);
  const [exportedInteractionSignature, setExportedInteractionSignature] = useState("");

  const currentFrame = useMemo(
    () => frameAtOrBefore(data?.frames ?? [], presentedTime),
    [data, presentedTime],
  );
  const currentAnnotationTime = currentFrame?.timestamp_seconds ?? currentTime;
  const selectedInteraction = useMemo(
    () => interactions.find((interaction) => interaction.interaction_id === selectedInteractionId) ?? null,
    [interactions, selectedInteractionId],
  );
  const displayedInteraction = interactionDraft ?? selectedInteraction;
  const originalInteraction = useMemo(
    () => interactions.find((interaction) => interaction.interaction_id === editingInteractionId) ?? null,
    [editingInteractionId, interactions],
  );
  const hasInteractionChanges = Boolean(
    interactionDraft && (!originalInteraction || !interactionsEqual(interactionDraft, originalInteraction)),
  );
  const sortedInteractions = useMemo(
    () => [...interactions].sort((left, right) => naturalCompare(left.interaction_id, right.interaction_id)),
    [interactions],
  );
  const interactionSignature = useMemo(() => JSON.stringify(sortedInteractions), [sortedInteractions]);
  const interactionTypes = useMemo(
    () => [...new Set(interactions.map((interaction) => interaction.interaction_type))].sort(naturalCompare),
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

  const cadence = useMemo(() => {
    const frames = data?.frames ?? [];
    let minimumDelta = Number.POSITIVE_INFINITY;
    let maximumDelta = 0;
    for (let i = 1; i < frames.length; i += 1) {
      const delta = frames[i].timestamp_seconds - frames[i - 1].timestamp_seconds;
      if (delta > 0) {
        minimumDelta = Math.min(minimumDelta, delta);
        maximumDelta = Math.max(maximumDelta, delta);
      }
    }
    if (!Number.isFinite(minimumDelta)) return { averageFps: 30, variable: false };
    const durationSeconds = frames[frames.length - 1].timestamp_seconds - frames[0].timestamp_seconds;
    const averageFps = durationSeconds > 0 ? (frames.length - 1) / durationSeconds : 30;
    return { averageFps, variable: maximumDelta > minimumDelta * 1.5 };
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
      let drewPolygon = false;
      for (const contour of track.contours_xy ?? []) {
        if (contour.length < 3) continue;
        drewPolygon = true;
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
      if (!drewPolygon && track.bbox_xyxy) {
        const [left, top, right, bottom] = track.bbox_xyxy;
        context.save();
        context.setLineDash([Math.max(8, width / 300), Math.max(5, width / 500)]);
        context.fillStyle = `${color}${highlighted ? "4d" : "1f"}`;
        context.strokeStyle = highlighted ? "#ffffff" : color;
        context.lineWidth = highlighted ? Math.max(7, width / 450) : Math.max(3, width / 900);
        context.fillRect(left, top, right - left, bottom - top);
        context.strokeRect(left, top, right - left, bottom - top);
        context.restore();
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
    const update = (_now?: number, metadata?: { mediaTime?: number }) => {
      if (!active) return;
      const frameTime = metadata?.mediaTime ?? video.currentTime;
      setCurrentTime(frameTime);
      setPresentedTime(frameTime);
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
    stopFrameStepping();
  }, []);

  useEffect(() => {
    if (!interactionDraft) return;
    const objectIds = [...selectedTracks].sort(naturalCompare);
    const isUnchanged = objectIds.length === interactionDraft.object_id_list.length
      && objectIds.every((id, index) => id === interactionDraft.object_id_list[index]);
    if (!isUnchanged) {
      setInteractionDraft({ ...interactionDraft, object_id_list: objectIds });
    }
  }, [interactionDraft, selectedTracks]);

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
    const hasUnexportedInteractions = interactions.length > 0
      && interactionSignature !== exportedInteractionSignature;
    if (hasUnexportedInteractions || hasInteractionChanges) {
      const warning = hasUnexportedInteractions
        ? "Saved interactions have not been exported. Loading another video will remove the current control JSON and all interactions.\n\nChoose Cancel to export them first."
        : "The interaction editor has unsaved changes. Loading another video will discard them and remove the current control JSON.\n\nChoose Cancel to return to editing.";
      if (!window.confirm(warning)) {
        event.target.value = "";
        setMessage("Video loading canceled. Your current labeling session is unchanged.");
        return;
      }
    }

    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
    const url = URL.createObjectURL(file);
    videoUrlRef.current = url;
    setVideoUrl(url);
    setVideoName(file.name);
    setVideoPath(file.webkitRelativePath || file.name);
    setData(null);
    setJsonName("No control file selected");
    setContourPath("");
    setLoadState("idle");
    setSelectedTracks(new Set());
    setInteractions([]);
    setInteractionDraft(null);
    setInteractionType("");
    setSelectedInteractionId(null);
    setEditingInteractionId(null);
    setExportedInteractionSignature("");
    if (jsonInputRef.current) jsonInputRef.current.value = "";
    setMessage("Video loaded. Select the matching control JSON.");
    setCurrentTime(0);
    setPresentedTime(0);
    setDuration(0);
    setIsPlaying(false);
    setHoveredContour(null);
  }

  function loadJson(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setJsonName(file.name);
    setContourPath(file.webkitRelativePath || file.name);
    setLoadState("reading");
    setMessage(`Reading ${file.name}. Large control files may take a moment…`);

    const workerSource = `
      self.onmessage = async ({ data: file }) => {
        try {
          const reader = file.stream().getReader();
          const decoder = new TextDecoder();
          const framesMarker = /"frames"\\s*:\\s*\\[/;
          let header = "";
          let foundFrames = false;
          let finishedFrames = false;
          let video;
          let depth = 0;
          let inString = false;
          let escaped = false;
          let frameText = "";
          let batch = [];
          let processed = 0;

          const emitFrame = () => {
            const frame = JSON.parse(frameText);
            batch.push({
              frame_index: frame.frame_index,
              timestamp_seconds: frame.timestamp_seconds,
              tracks: (frame.tracks || []).map((track) => ({
                track_id: track.track_id,
                label: track.label,
                confidence: track.confidence,
                bbox_xyxy: track.bbox_xyxy,
                contours_xy: track.contours_xy?.length ? track.contours_xy : track.metadata?.contours_xy || [],
              })),
            });
            processed += 1;
            frameText = "";
            if (batch.length >= 20) {
              self.postMessage({ type: "batch", video, frames: batch, processed });
              batch = [];
            }
          };

          const processFrames = (text) => {
            for (let index = 0; index < text.length && !finishedFrames; index += 1) {
              const character = text[index];
              if (depth === 0) {
                if (character === "{") {
                  depth = 1;
                  frameText = character;
                } else if (character === "]") {
                  finishedFrames = true;
                }
                continue;
              }

              frameText += character;
              if (inString) {
                if (escaped) escaped = false;
                else if (character.charCodeAt(0) === 92) escaped = true;
                else if (character.charCodeAt(0) === 34) inString = false;
              } else if (character.charCodeAt(0) === 34) {
                inString = true;
              } else if (character === "{") {
                depth += 1;
              } else if (character === "}") {
                depth -= 1;
                if (depth === 0) emitFrame();
              }
            }
          };

          while (!finishedFrames) {
            const { value, done } = await reader.read();
            const text = decoder.decode(value || new Uint8Array(), { stream: !done });
            if (!foundFrames) {
              header += text;
              const marker = framesMarker.exec(header);
              if (marker) {
                const headerObject = JSON.parse(header.slice(0, marker.index) + '"frames":[]}');
                video = headerObject.video;
                foundFrames = true;
                processFrames(header.slice(marker.index + marker[0].length));
                header = "";
              } else if (header.length > 5_000_000) {
                throw new Error("Could not find the frames array near the start of the file.");
              }
            } else {
              processFrames(text);
            }
            if (done) break;
          }

          if (!foundFrames || !finishedFrames || depth !== 0) {
            throw new Error("The control JSON ended before the frames array was complete.");
          }
          if (batch.length) self.postMessage({ type: "batch", video, frames: batch, processed });
          self.postMessage({ type: "done", video, processed });
          await reader.cancel();
        } catch (error) {
          self.postMessage({ type: "error", error: error instanceof Error ? error.message : String(error) });
        }
      };
    `;
    const workerUrl = URL.createObjectURL(new Blob([workerSource], { type: "text/javascript" }));
    const worker = new Worker(workerUrl);
    const loadedFrames: Frame[] = [];
    worker.onmessage = (workerEvent: MessageEvent<{
      type: "batch" | "done" | "error";
      video?: string;
      frames?: Frame[];
      processed?: number;
      error?: string;
    }>) => {
      if (workerEvent.data.type === "batch" && workerEvent.data.frames) {
        loadedFrames.push(...workerEvent.data.frames);
        setMessage(`Reading ${file.name}… ${loadedFrames.length.toLocaleString()} frames parsed.`);
        return;
      }
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      if (workerEvent.data.type === "error") {
        setLoadState("error");
        setMessage(`Could not read control JSON: ${workerEvent.data.error ?? "Unknown error"}`);
        return;
      }
      const nextData: ControlData = { video: workerEvent.data.video, frames: loadedFrames };
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
    const frames = data?.frames ?? [];
    if (!frames.length) {
      seek(Math.min(video.duration || Number.POSITIVE_INFINITY, Math.max(0, video.currentTime + direction / 30)));
      return;
    }
    const displayedFrame = frameAtOrBefore(frames, video.currentTime);
    const displayedIndex = displayedFrame ? frames.indexOf(displayedFrame) : 0;
    const targetIndex = Math.min(frames.length - 1, Math.max(0, displayedIndex + direction));
    seekToFrame(targetIndex);
  }

  function stopFrameStepping() {
    if (stepHoldDelayRef.current !== null) window.clearTimeout(stepHoldDelayRef.current);
    if (stepRepeatRef.current !== null) window.clearInterval(stepRepeatRef.current);
    stepHoldDelayRef.current = null;
    stepRepeatRef.current = null;
  }

  function startFrameStepping(direction: -1 | 1) {
    stopFrameStepping();
    stepFrame(direction);
    stepHoldDelayRef.current = window.setTimeout(() => {
      stepRepeatRef.current = window.setInterval(() => stepFrame(direction), STEP_REPEAT_MS);
    }, STEP_HOLD_DELAY_MS);
  }

  function seekToFrame(frameIndex: number) {
    const video = videoRef.current;
    const frames = data?.frames ?? [];
    const targetFrame = frames[frameIndex];
    if (!video || !targetFrame) return;
    const followingFrame = frames[frameIndex + 1];
    const seekPosition = followingFrame
      ? targetFrame.timestamp_seconds + (followingFrame.timestamp_seconds - targetFrame.timestamp_seconds) / 2
      : targetFrame.timestamp_seconds;
    video.currentTime = seekPosition;
    setCurrentTime(targetFrame.timestamp_seconds);
    setPresentedTime(targetFrame.timestamp_seconds);
  }

  function seek(value: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = value;
    setCurrentTime(value);
    const targetFrame = frameAtOrBefore(data?.frames ?? [], value);
    setPresentedTime(targetFrame?.timestamp_seconds ?? value);
  }

  function createInteraction() {
    const type = (interactionDraft?.interaction_type ?? interactionType).trim();
    if (!type) {
      setMessage("Enter an interaction type before creating the interaction.");
      interactionTypeInputRef.current?.focus();
      return;
    }
    const draft: Interaction = {
      interaction_type: type,
      interaction_id: nextInteractionId(interactions),
      object_id_list: [...selectedTracks].sort(naturalCompare),
      start_time: annotationTime(currentAnnotationTime),
      end_time: null,
    };
    setInteractionDraft(draft);
    setEditingInteractionId(null);
    setSelectedInteractionId(null);
    setMessage(`Interaction draft created with ${draft.object_id_list.length} selected objects.`);
  }

  function setInteractionStartTime() {
    if (!interactionDraft) return;
    const startTime = annotationTime(currentAnnotationTime);
    setInteractionDraft({ ...interactionDraft, start_time: startTime });
    setMessage(`Interaction start set to ${formatTime(startTime)}.`);
  }

  function setInteractionEndTime() {
    if (!interactionDraft) return;
    const endTime = annotationTime(currentAnnotationTime);
    if (endTime < interactionDraft.start_time) {
      setMessage("The interaction end time cannot be earlier than its start time.");
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
      object_id_list: [...interactionDraft.object_id_list].sort(naturalCompare),
    };
    const validationErrors: string[] = [];
    if (!savedInteraction.interaction_type) validationErrors.push("Interaction type is required.");
    if (!savedInteraction.interaction_id.trim()) validationErrors.push("Interaction ID is required.");
    if (interactions.some((interaction) => interaction.interaction_id === savedInteraction.interaction_id
      && interaction.interaction_id !== editingInteractionId)) {
      validationErrors.push(`Interaction ID ${savedInteraction.interaction_id} is already in use.`);
    }
    if (!savedInteraction.object_id_list.length) validationErrors.push("Select at least one object.");
    if (!Number.isFinite(savedInteraction.start_time)) validationErrors.push("Start time is required.");
    if (savedInteraction.end_time === null || !Number.isFinite(savedInteraction.end_time)) {
      validationErrors.push("End time is required.");
    } else if (savedInteraction.end_time < savedInteraction.start_time) {
      validationErrors.push("End time cannot be earlier than start time.");
    }
    if (!data?.frames.length) validationErrors.push("Control JSON is required to validate contour visibility.");

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
      setSelectedTracks(new Set(originalInteraction.object_id_list));
      setMessage("Unsaved interaction changes discarded.");
      return;
    }
    setInteractionDraft(null);
    setEditingInteractionId(null);
    setInteractionType("");
    setMessage("Interaction draft discarded.");
  }

  function exportInteractions() {
    const payload = {
      video: data?.video || videoPath,
      contour: contourPath,
      interaction_list: sortedInteractions,
    };
    const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const baseName = videoName.replace(/\.[^.]+$/, "") || "interactions";
    link.href = url;
    link.download = `${baseName}.interactions.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setExportedInteractionSignature(interactionSignature);
    setMessage(`Exported ${sortedInteractions.length} interactions.`);
  }

  function selectInteraction(interaction: Interaction) {
    setInteractionDraft(cloneInteraction(interaction));
    setEditingInteractionId(interaction.interaction_id);
    setSelectedInteractionId(interaction.interaction_id);
    setSelectedTracks(new Set(interaction.object_id_list));
    seek(interaction.start_time);
    setMessage(`Jumped to the start of ${interaction.interaction_type}.`);
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
      let hasPolygon = false;
      for (const contour of track.contours_xy ?? []) {
        if (contour.length >= 3 && pointInPolygon(point, contour)) {
          hasPolygon = true;
          hits.push({ id: track.track_id, label: track.label, area: polygonArea(contour) });
        }
        if (contour.length >= 3) hasPolygon = true;
      }
      if (!hasPolygon && track.bbox_xyxy) {
        const [left, top, right, bottom] = track.bbox_xyxy;
        if (point[0] >= left && point[0] <= right && point[1] >= top && point[1] <= bottom) {
          hits.push({ id: track.track_id, label: track.label, area: (right - left) * (bottom - top) });
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
                  onSeeked={(event) => {
                    const time = event.currentTarget.currentTime;
                    const displayedFrame = frameAtOrBefore(data?.frames ?? [], time);
                    setCurrentTime(displayedFrame?.timestamp_seconds ?? time);
                    setPresentedTime(displayedFrame?.timestamp_seconds ?? time);
                  }}
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
            <button
              aria-label="Previous frame"
              onPointerDown={(event) => {
                event.preventDefault();
                event.currentTarget.setPointerCapture(event.pointerId);
                startFrameStepping(-1);
              }}
              onPointerUp={stopFrameStepping}
              onPointerCancel={stopFrameStepping}
              onLostPointerCapture={stopFrameStepping}
              onClick={(event) => { if (event.detail === 0) stepFrame(-1); }}
              disabled={!videoUrl}
            >│◀</button>
            <button className="play-button" aria-label={isPlaying ? "Pause" : "Play"} onClick={() => {
              const video = videoRef.current;
              if (!video) return;
              if (video.paused) void video.play(); else video.pause();
            }} disabled={!videoUrl}>{isPlaying ? "Ⅱ" : "▶"}</button>
            <button
              aria-label="Next frame"
              onPointerDown={(event) => {
                event.preventDefault();
                event.currentTarget.setPointerCapture(event.pointerId);
                startFrameStepping(1);
              }}
              onPointerUp={stopFrameStepping}
              onPointerCancel={stopFrameStepping}
              onLostPointerCapture={stopFrameStepping}
              onClick={(event) => { if (event.detail === 0) stepFrame(1); }}
              disabled={!videoUrl}
            >▶│</button>
            <span className="timecode">{formatTime(currentTime)}</span>
            <input aria-label="Video position" type="range" min="0" max={duration || 0} step="0.001" value={Math.min(currentTime, duration || 0)} onChange={(event) => seek(Number(event.target.value))} disabled={!videoUrl} />
            <span className="timecode muted">{formatTime(duration)}</span>
            <span className="fps">{cadence.variable ? "VFR · " : ""}{cadence.averageFps.toFixed(2)} avg FPS</span>
          </div>
          <p className="shortcut-hint"><kbd>Space</kbd> play / pause <kbd>←</kbd><kbd>→</kbd> step one frame</p>
        </div>

        <div className="side-column">
        <aside className="inspector">
          <div className="inspector-title">
            <div><span className="eyebrow">OVERLAY FILTERS</span><h2>Contours</h2></div>
            <div className="inspector-title-actions">
              <span>{selectedTracks.size}/{catalog.tracks.length}</span>
              <button onClick={() => setSelectedTracks(new Set())} disabled={!selectedTracks.size}>Clear selection</button>
            </div>
          </div>

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

        <section className="interactions-panel" aria-label="Interactions">
          <div className="interactions-title">
            <div><span className="eyebrow">ANNOTATION EVENTS</span><h2>Interactions</h2></div>
            <div className="interactions-title-actions">
              <span>{interactions.length}</span>
              <button onClick={exportInteractions} disabled={!interactions.length}>Export JSON</button>
            </div>
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
                <button className="danger-action" onClick={discardInteraction}>{originalInteraction ? "Discard changes" : "Discard"}</button>
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
