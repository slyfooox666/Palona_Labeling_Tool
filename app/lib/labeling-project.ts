export const ANNOTATION_PROJECT_SCHEMA = "palona.annotation-project/v1" as const;

const EPSILON = 1e-12;
const NATURAL_COLLATOR = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

export type NormalizedPoint = [number, number];

export type Interaction = {
  interaction_id: string;
  interaction_type: string;
  object_id_list: string[];
  start_time: number;
  end_time: number;
  start_frame?: number;
  end_frame?: number;
  notes?: string;
  created_at?: string;
  updated_at?: string;
};

export type ProjectSource = {
  video_path: string;
  contour_path: string;
  video_file_size_bytes?: number;
  contour_file_size_bytes?: number;
  video_fps: number;
  video_width: number;
  video_height: number;
  frame_count: number;
  duration_seconds: number;
  available_categories: string[];
};

export type RoiState = {
  polygon: NormalizedPoint[] | null;
  blackout_enabled: boolean;
};

export type IdAliases = Record<string, string>;

export type AnnotationProject = {
  schema_version: typeof ANNOTATION_PROJECT_SCHEMA;
  source: ProjectSource;
  interactions: Interaction[];
  roi: RoiState;
  id_aliases: IdAliases;
  created_at: string;
  updated_at: string;
};

export type TrackLabel = {
  id: string;
  label: string;
};

export type TrackLabelSource =
  | Readonly<Record<string, string>>
  | ReadonlyMap<string, string>
  | readonly TrackLabel[];

export type MinimalInteraction = {
  event: string;
  person_id_list: string[];
  table_id: string;
  start_time: number;
  end_time: number;
};

export class AnnotationProjectValidationError extends Error {
  readonly path: string;

  constructor(path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = "AnnotationProjectValidationError";
    this.path = path;
  }
}

type JsonObject = Record<string, unknown>;

function fail(path: string, message: string): never {
  throw new AnnotationProjectValidationError(path, message);
}

function isPlainObject(value: unknown): value is JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function expectObject(value: unknown, path: string): JsonObject {
  if (!isPlainObject(value)) fail(path, "must be a JSON object");
  return value;
}

function assertExactKeys(
  value: JsonObject,
  required: readonly string[],
  optional: readonly string[],
  path: string,
) {
  const allowed = new Set([...required, ...optional]);
  for (const key of required) {
    if (!Object.hasOwn(value, key)) fail(`${path}.${key}`, "is required");
  }
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) fail(`${path}.${key}`, "is not a supported field");
  }
}

function expectString(value: unknown, path: string): string {
  if (typeof value !== "string") fail(path, "must be a string");
  if (!value.trim()) fail(path, "must not be empty");
  if (value !== value.trim()) fail(path, "must not have leading or trailing whitespace");
  return value;
}

function expectFiniteNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(path, "must be a finite number");
  return Object.is(value, -0) ? 0 : value;
}

function expectPositiveNumber(value: unknown, path: string): number {
  const number = expectFiniteNumber(value, path);
  if (number <= 0) fail(path, "must be greater than zero");
  return number;
}

function expectNonNegativeInteger(value: unknown, path: string): number {
  const number = expectFiniteNumber(value, path);
  if (!Number.isSafeInteger(number) || number < 0) fail(path, "must be a non-negative safe integer");
  return number;
}

function expectPositiveInteger(value: unknown, path: string): number {
  const number = expectNonNegativeInteger(value, path);
  if (number === 0) fail(path, "must be greater than zero");
  return number;
}

function expectBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") fail(path, "must be a boolean");
  return value;
}

function expectIsoTimestamp(value: unknown, path: string): string {
  const timestamp = expectString(value, path);
  if (!Number.isFinite(Date.parse(timestamp))) fail(path, "must be a valid ISO-8601 timestamp");
  return timestamp;
}

function expectStringArray(value: unknown, path: string, allowEmpty: boolean): string[] {
  if (!Array.isArray(value)) fail(path, "must be an array");
  if (!allowEmpty && value.length === 0) fail(path, "must contain at least one item");

  const seen = new Set<string>();
  return value.map((item, index) => {
    const parsed = expectString(item, `${path}[${index}]`);
    if (seen.has(parsed)) fail(`${path}[${index}]`, `duplicates ${JSON.stringify(parsed)}`);
    seen.add(parsed);
    return parsed;
  });
}

function cross(a: NormalizedPoint, b: NormalizedPoint, c: NormalizedPoint): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function pointOnSegment(a: NormalizedPoint, b: NormalizedPoint, point: NormalizedPoint): boolean {
  return Math.abs(cross(a, b, point)) <= EPSILON
    && point[0] >= Math.min(a[0], b[0]) - EPSILON
    && point[0] <= Math.max(a[0], b[0]) + EPSILON
    && point[1] >= Math.min(a[1], b[1]) - EPSILON
    && point[1] <= Math.max(a[1], b[1]) + EPSILON;
}

function segmentsIntersect(
  a: NormalizedPoint,
  b: NormalizedPoint,
  c: NormalizedPoint,
  d: NormalizedPoint,
): boolean {
  const abC = cross(a, b, c);
  const abD = cross(a, b, d);
  const cdA = cross(c, d, a);
  const cdB = cross(c, d, b);

  if (((abC > EPSILON && abD < -EPSILON) || (abC < -EPSILON && abD > EPSILON))
    && ((cdA > EPSILON && cdB < -EPSILON) || (cdA < -EPSILON && cdB > EPSILON))) {
    return true;
  }
  return (Math.abs(abC) <= EPSILON && pointOnSegment(a, b, c))
    || (Math.abs(abD) <= EPSILON && pointOnSegment(a, b, d))
    || (Math.abs(cdA) <= EPSILON && pointOnSegment(c, d, a))
    || (Math.abs(cdB) <= EPSILON && pointOnSegment(c, d, b));
}

function parseNormalizedPoint(value: unknown, path: string): NormalizedPoint {
  if (!Array.isArray(value) || value.length !== 2) fail(path, "must be a [x, y] pair");
  const x = expectFiniteNumber(value[0], `${path}[0]`);
  const y = expectFiniteNumber(value[1], `${path}[1]`);
  if (x < 0 || x > 1) fail(`${path}[0]`, "must be within normalized range [0, 1]");
  if (y < 0 || y > 1) fail(`${path}[1]`, "must be within normalized range [0, 1]");
  return [x, y];
}

/** Validate and copy one normalized, simple (non-self-intersecting) ROI polygon. */
export function validateNormalizedRoiPolygon(value: unknown, path = "roi.polygon"): NormalizedPoint[] {
  if (!Array.isArray(value)) fail(path, "must be an array");
  if (value.length < 3) fail(path, "must contain at least three vertices");
  const points = value.map((point, index) => parseNormalizedPoint(point, `${path}[${index}]`));

  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    if (Math.abs(current[0] - next[0]) <= EPSILON && Math.abs(current[1] - next[1]) <= EPSILON) {
      fail(`${path}[${index}]`, "must not duplicate an adjacent vertex");
    }
  }

  for (let left = 0; left < points.length; left += 1) {
    for (let right = left + 1; right < points.length; right += 1) {
      if (Math.abs(points[left][0] - points[right][0]) <= EPSILON
        && Math.abs(points[left][1] - points[right][1]) <= EPSILON) {
        fail(`${path}[${right}]`, `duplicates vertex ${left}`);
      }
    }
  }

  for (let index = 0; index < points.length; index += 1) {
    const previous = points[(index - 1 + points.length) % points.length];
    const current = points[index];
    const next = points[(index + 1) % points.length];
    const adjacentEdgesOverlap = Math.abs(cross(previous, current, next)) <= EPSILON
      && (previous[0] - current[0]) * (next[0] - current[0])
        + (previous[1] - current[1]) * (next[1] - current[1]) > EPSILON;
    if (adjacentEdgesOverlap) {
      fail(`${path}[${index}]`, "creates overlapping adjacent edges");
    }
  }

  const twiceArea = points.reduce((sum, point, index) => {
    const next = points[(index + 1) % points.length];
    return sum + point[0] * next[1] - next[0] * point[1];
  }, 0);
  if (Math.abs(twiceArea) <= EPSILON) fail(path, "must enclose a non-zero area");

  for (let first = 0; first < points.length; first += 1) {
    const firstNext = (first + 1) % points.length;
    for (let second = first + 1; second < points.length; second += 1) {
      const secondNext = (second + 1) % points.length;
      const adjacent = first === second
        || firstNext === second
        || secondNext === first;
      if (adjacent) continue;
      if (segmentsIntersect(points[first], points[firstNext], points[second], points[secondNext])) {
        fail(path, `self-intersection between edges ${first} and ${second}`);
      }
    }
  }

  return points;
}

function parseInteraction(value: unknown, path: string): Interaction {
  const object = expectObject(value, path);
  assertExactKeys(
    object,
    ["interaction_id", "interaction_type", "object_id_list", "start_time", "end_time"],
    ["start_frame", "end_frame", "notes", "created_at", "updated_at"],
    path,
  );

  const startTime = expectFiniteNumber(object.start_time, `${path}.start_time`);
  const endTime = expectFiniteNumber(object.end_time, `${path}.end_time`);
  if (startTime < 0) fail(`${path}.start_time`, "must not be negative");
  if (endTime <= startTime) fail(`${path}.end_time`, "must be later than start_time");

  const interaction: Interaction = {
    interaction_id: expectString(object.interaction_id, `${path}.interaction_id`),
    interaction_type: expectString(object.interaction_type, `${path}.interaction_type`),
    object_id_list: expectStringArray(object.object_id_list, `${path}.object_id_list`, false),
    start_time: startTime,
    end_time: endTime,
  };

  if (Object.hasOwn(object, "start_frame")) {
    interaction.start_frame = expectNonNegativeInteger(object.start_frame, `${path}.start_frame`);
  }
  if (Object.hasOwn(object, "end_frame")) {
    interaction.end_frame = expectNonNegativeInteger(object.end_frame, `${path}.end_frame`);
  }
  if ((interaction.start_frame === undefined) !== (interaction.end_frame === undefined)) {
    fail(path, "start_frame and end_frame must either both be present or both be absent");
  }
  if (interaction.start_frame !== undefined && interaction.end_frame! < interaction.start_frame) {
    fail(`${path}.end_frame`, "must not be earlier than start_frame");
  }
  if (Object.hasOwn(object, "notes")) {
    if (typeof object.notes !== "string") fail(`${path}.notes`, "must be a string");
    interaction.notes = object.notes;
  }
  if (Object.hasOwn(object, "created_at")) {
    interaction.created_at = expectIsoTimestamp(object.created_at, `${path}.created_at`);
  }
  if (Object.hasOwn(object, "updated_at")) {
    interaction.updated_at = expectIsoTimestamp(object.updated_at, `${path}.updated_at`);
  }
  return interaction;
}

function parseSource(value: unknown, path: string): ProjectSource {
  const object = expectObject(value, path);
  assertExactKeys(object, [
    "video_path",
    "contour_path",
    "video_fps",
    "video_width",
    "video_height",
    "frame_count",
    "duration_seconds",
    "available_categories",
  ], ["video_file_size_bytes", "contour_file_size_bytes"], path);

  const source: ProjectSource = {
    video_path: expectString(object.video_path, `${path}.video_path`),
    contour_path: expectString(object.contour_path, `${path}.contour_path`),
    video_fps: expectPositiveNumber(object.video_fps, `${path}.video_fps`),
    video_width: expectPositiveInteger(object.video_width, `${path}.video_width`),
    video_height: expectPositiveInteger(object.video_height, `${path}.video_height`),
    frame_count: expectPositiveInteger(object.frame_count, `${path}.frame_count`),
    duration_seconds: expectPositiveNumber(object.duration_seconds, `${path}.duration_seconds`),
    available_categories: expectStringArray(
      object.available_categories,
      `${path}.available_categories`,
      true,
    ),
  };
  if (Object.hasOwn(object, "video_file_size_bytes")) {
    source.video_file_size_bytes = expectPositiveInteger(
      object.video_file_size_bytes,
      `${path}.video_file_size_bytes`,
    );
  }
  if (Object.hasOwn(object, "contour_file_size_bytes")) {
    source.contour_file_size_bytes = expectPositiveInteger(
      object.contour_file_size_bytes,
      `${path}.contour_file_size_bytes`,
    );
  }
  return source;
}

function parseRoi(value: unknown, path: string): RoiState {
  const object = expectObject(value, path);
  assertExactKeys(object, ["polygon", "blackout_enabled"], [], path);
  const polygon = object.polygon === null
    ? null
    : validateNormalizedRoiPolygon(object.polygon, `${path}.polygon`);
  const blackoutEnabled = expectBoolean(object.blackout_enabled, `${path}.blackout_enabled`);
  if (polygon === null && blackoutEnabled) {
    fail(`${path}.blackout_enabled`, "cannot be enabled without an ROI polygon");
  }
  return { polygon, blackout_enabled: blackoutEnabled };
}

function sortedAliases(aliases: IdAliases): IdAliases {
  const sorted: IdAliases = Object.create(null) as IdAliases;
  for (const key of Object.keys(aliases).sort((left, right) => NATURAL_COLLATOR.compare(left, right))) {
    sorted[key] = aliases[key];
  }
  return sorted;
}

/** Validate the complete alias graph and return a defensive, deterministically ordered copy. */
export function validateIdAliases(value: unknown, path = "id_aliases"): IdAliases {
  const object = expectObject(value, path);
  const aliases: IdAliases = Object.create(null) as IdAliases;
  for (const [rawAlias, rawTarget] of Object.entries(object)) {
    const alias = expectString(rawAlias, `${path} key`);
    const target = expectString(rawTarget, `${path}.${alias}`);
    aliases[alias] = target;
  }
  for (const alias of Object.keys(aliases)) resolveTrackId(alias, aliases, `${path}.${alias}`);
  return sortedAliases(aliases);
}

/** Resolve a possibly transitive alias to its canonical track ID. */
export function resolveTrackId(trackId: string, aliases: Readonly<IdAliases>, path = "track_id"): string {
  const start = expectString(trackId, path);
  const visited = new Set<string>();
  let current = start;
  while (Object.hasOwn(aliases, current)) {
    if (visited.has(current)) {
      fail(path, `alias cycle detected at ${JSON.stringify(current)}`);
    }
    visited.add(current);
    current = expectString(aliases[current], `${path} alias target`);
  }
  return current;
}

/** Add or replace an alias without mutating the supplied graph; cycles are rejected. */
export function withIdAlias(aliases: Readonly<IdAliases>, alias: string, target: string): IdAliases {
  const next: IdAliases = Object.create(null) as IdAliases;
  Object.assign(next, aliases);
  next[expectString(alias, "alias")] = expectString(target, "alias target");
  return validateIdAliases(next);
}

/** Delete an alias without mutating the supplied graph. */
export function withoutIdAlias(aliases: Readonly<IdAliases>, alias: string): IdAliases {
  const parsedAlias = expectString(alias, "alias");
  const next: IdAliases = Object.create(null) as IdAliases;
  for (const [key, value] of Object.entries(aliases)) {
    if (key !== parsedAlias) next[key] = value;
  }
  return validateIdAliases(next);
}

/** Return a copy whose participant IDs are resolved and de-duplicated in stable natural order. */
export function canonicalizeInteraction(
  interaction: Interaction,
  aliases: Readonly<IdAliases>,
): Interaction {
  const parsed = parseInteraction(interaction, "interaction");
  const checkedAliases = validateIdAliases(aliases);
  const objectIds = [...new Set(
    parsed.object_id_list.map((trackId) => resolveTrackId(trackId, checkedAliases)),
  )].sort((left, right) => NATURAL_COLLATOR.compare(left, right));
  return { ...parsed, object_id_list: objectIds };
}

/** Parse an already decoded JSON value using the versioned, closed project contract. */
export function parseAnnotationProject(value: unknown): AnnotationProject {
  const root = expectObject(value, "project");
  assertExactKeys(root, [
    "schema_version",
    "source",
    "interactions",
    "roi",
    "id_aliases",
    "created_at",
    "updated_at",
  ], [], "project");
  if (root.schema_version !== ANNOTATION_PROJECT_SCHEMA) {
    fail("project.schema_version", `must equal ${JSON.stringify(ANNOTATION_PROJECT_SCHEMA)}`);
  }
  if (!Array.isArray(root.interactions)) fail("project.interactions", "must be an array");

  const source = parseSource(root.source, "project.source");
  const interactions = root.interactions.map((interaction, index) => (
    parseInteraction(interaction, `project.interactions[${index}]`)
  ));
  const interactionIds = new Set<string>();
  interactions.forEach((interaction, index) => {
    if (interactionIds.has(interaction.interaction_id)) {
      fail(`project.interactions[${index}].interaction_id`, "must be unique within the project");
    }
    interactionIds.add(interaction.interaction_id);
    if (interaction.end_time > source.duration_seconds + EPSILON) {
      fail(`project.interactions[${index}].end_time`, "must not exceed the source duration");
    }
    if (interaction.start_frame !== undefined && interaction.start_frame >= source.frame_count) {
      fail(`project.interactions[${index}].start_frame`, "must be within the source frame range");
    }
    if (interaction.end_frame !== undefined && interaction.end_frame >= source.frame_count) {
      fail(`project.interactions[${index}].end_frame`, "must be within the source frame range");
    }
  });

  const createdAt = expectIsoTimestamp(root.created_at, "project.created_at");
  const updatedAt = expectIsoTimestamp(root.updated_at, "project.updated_at");
  if (Date.parse(updatedAt) < Date.parse(createdAt)) {
    fail("project.updated_at", "must not be earlier than created_at");
  }

  return {
    schema_version: ANNOTATION_PROJECT_SCHEMA,
    source,
    interactions,
    roi: parseRoi(root.roi, "project.roi"),
    id_aliases: validateIdAliases(root.id_aliases, "project.id_aliases"),
    created_at: createdAt,
    updated_at: updatedAt,
  };
}

/** Parse project JSON text and report syntax failures using the same validation error type. */
export function parseAnnotationProjectJson(json: string): AnnotationProject {
  if (typeof json !== "string") fail("project", "JSON input must be a string");
  let value: unknown;
  try {
    value = JSON.parse(json) as unknown;
  } catch (error) {
    const message = error instanceof Error ? error.message : "invalid JSON";
    fail("project", `invalid JSON (${message})`);
  }
  return parseAnnotationProject(value);
}

/** Validate and serialize a project to deterministic, human-readable JSON. */
export function serializeAnnotationProject(project: AnnotationProject): string {
  return `${JSON.stringify(parseAnnotationProject(project), null, 2)}\n`;
}

function normalizedTrackLabels(source: TrackLabelSource): Map<string, string> {
  const labels = new Map<string, string>();
  let entries: readonly (readonly [string, string])[];
  if (source instanceof Map) {
    entries = [...source.entries()];
  } else if (Array.isArray(source)) {
    entries = source.map((track) => [track.id, track.label] as const);
  } else {
    const object = expectObject(source, "track_labels");
    entries = Object.entries(object).map(([id, label]) => [id, expectString(label, `track_labels.${id}`)] as const);
  }

  entries.forEach(([rawId, rawLabel], index) => {
    const id = expectString(rawId, `track_labels[${index}].id`);
    const label = expectString(rawLabel, `track_labels[${index}].label`);
    if (labels.has(id)) fail(`track_labels[${index}].id`, `duplicates ${JSON.stringify(id)}`);
    labels.set(id, label);
  });
  return labels;
}

function labelTokens(label: string): string[] {
  return label.toLocaleLowerCase("en-US").split(/[^a-z0-9]+/u).filter(Boolean);
}

export function isPersonTrackLabel(label: string): boolean {
  const tokens = labelTokens(label);
  const personTokens = new Set([
    "person", "persons", "people", "customer", "customers", "staff", "employee", "employees",
    "worker", "workers", "waiter", "waiters", "waitress", "waitresses", "server", "servers",
  ]);
  if (tokens.some((token) => personTokens.has(token))) return true;
  return tokens.includes("cashier")
    && !tokens.some((token) => ["machine", "register", "station", "counter"].includes(token));
}

export function isTableTrackLabel(label: string): boolean {
  const tokens = labelTokens(label);
  return tokens.some((token) => token === "table" || token === "tables" || token.startsWith("table-"));
}

/**
 * Export the exact table-interaction array consumed by the training pipeline.
 * Every selected ID is canonicalized, then classified from its SAM3 track label.
 */
export function exportMinimalInteractions(
  interactions: readonly Interaction[],
  trackLabels: TrackLabelSource,
  aliases: Readonly<IdAliases> = {},
): MinimalInteraction[] {
  const checkedAliases = validateIdAliases(aliases);
  const labels = normalizedTrackLabels(trackLabels);

  return interactions.map((rawInteraction, interactionIndex) => {
    const path = `interactions[${interactionIndex}]`;
    const interaction = canonicalizeInteraction(rawInteraction, checkedAliases);
    const people: string[] = [];
    const tables: string[] = [];

    for (const rawTrackId of rawInteraction.object_id_list) {
      if (!labels.has(rawTrackId) && !Object.hasOwn(checkedAliases, rawTrackId)) {
        fail(`${path}.object_id_list`, `track ${JSON.stringify(rawTrackId)} is absent from contour labels and aliases`);
      }
    }
    for (const trackId of interaction.object_id_list) {
      const label = labels.get(trackId);
      if (label === undefined) {
        fail(`${path}.object_id_list`, `canonical track ${JSON.stringify(trackId)} has no contour label`);
      }
      if (isPersonTrackLabel(label)) people.push(trackId);
      else if (isTableTrackLabel(label)) tables.push(trackId);
      else {
        fail(
          `${path}.object_id_list`,
          `track ${JSON.stringify(trackId)} has unsupported label ${JSON.stringify(label)}; minimal table export accepts only people and one table`,
        );
      }
    }
    if (people.length === 0) fail(`${path}.object_id_list`, "must include at least one person-labeled track");
    if (tables.length !== 1) fail(`${path}.object_id_list`, "must include exactly one table-labeled track");

    people.sort((left, right) => NATURAL_COLLATOR.compare(left, right));
    return {
      event: interaction.interaction_type,
      person_id_list: people,
      table_id: tables[0],
      start_time: interaction.start_time,
      end_time: interaction.end_time,
    };
  });
}

export function serializeMinimalInteractions(
  interactions: readonly Interaction[],
  trackLabels: TrackLabelSource,
  aliases: Readonly<IdAliases> = {},
): string {
  return `${JSON.stringify(exportMinimalInteractions(interactions, trackLabels, aliases), null, 2)}\n`;
}

function expectFps(fps: number): number {
  return expectPositiveNumber(fps, "fps");
}

/** Convert a zero-based frame index to seconds using the actual video FPS. */
export function frameToTime(frameIndex: number, fps: number): number {
  return expectNonNegativeInteger(frameIndex, "frame_index") / expectFps(fps);
}

/** Convert seconds to a zero-based frame index, with an explicit rounding policy. */
export function timeToFrame(
  seconds: number,
  fps: number,
  rounding: "floor" | "nearest" | "ceil" = "nearest",
): number {
  const parsedSeconds = expectFiniteNumber(seconds, "seconds");
  if (parsedSeconds < 0) fail("seconds", "must not be negative");
  const rawFrame = parsedSeconds * expectFps(fps);
  const frameIndex = rounding === "floor"
    ? Math.floor(rawFrame + EPSILON)
    : rounding === "ceil"
      ? Math.ceil(rawFrame - EPSILON)
      : Math.round(rawFrame);
  return expectNonNegativeInteger(frameIndex, "frame_index");
}

/** Convert a normalized coordinate to video/canvas coordinates. */
export function normalizedToPixel(point: NormalizedPoint, width: number, height: number): [number, number] {
  const [x, y] = parseNormalizedPoint(point, "point");
  return [x * expectPositiveNumber(width, "width"), y * expectPositiveNumber(height, "height")];
}

/** Convert video/canvas coordinates to a normalized coordinate, rejecting out-of-frame points. */
export function pixelToNormalized(point: [number, number], width: number, height: number): NormalizedPoint {
  if (!Array.isArray(point) || point.length !== 2) fail("point", "must be a [x, y] pair");
  const parsedWidth = expectPositiveNumber(width, "width");
  const parsedHeight = expectPositiveNumber(height, "height");
  const normalized: NormalizedPoint = [
    expectFiniteNumber(point[0], "point[0]") / parsedWidth,
    expectFiniteNumber(point[1], "point[1]") / parsedHeight,
  ];
  return parseNormalizedPoint(normalized, "point");
}
