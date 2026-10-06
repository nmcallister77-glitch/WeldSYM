export type XY = [number, number];

export interface Pose {
  x: number;
  y: number;
  angle: number;
  height: number;
}

export interface ScanPoint {
  x: number;
  y: number;
  z: number;
  line: number;
}

export const TARGET_DURATION = 30;
export const FIELD = { x: 2.5, y: 2 };
export const SCAN_LINES = 41;
export const SCAN_POINTS = 81;
export const PART = { length: 2.4, width: 1.2 };
export const SEAM_HALF_LENGTH = 0.95;
export const BEAD_HALF_WIDTH = 0.16;
export const TOLERANCE_UM = 100;
export const NOMINAL: Pose = { x: 0, y: 0, angle: 0, height: 500 };
export const ACTUAL: Pose = {
  x: 0.42,
  y: -0.31,
  angle: (7 * Math.PI) / 180,
  height: 580,
};

export const TARGET_STEPS = [
  {
    start: 0,
    end: 10,
    title: "Scan the part",
    short: "Scan",
    eyebrow: "1 / Scan",
    heading: "Look before you weld",
    description:
      "With the processing laser off, the galvos sweep only the cyan LDD beam across the fixture in a raster. Every sample is a height reading through the same lens the weld will use.",
    focus: "LDD raster over the tab",
    mechanism: "Galvo raster → height map",
    detail:
      "The scan is a few thousand points over a 5 × 4 mm field. Each point is measured relative to the focal reference, so the map is already in scanner coordinates.",
  },
  {
    start: 10,
    end: 14,
    title: "Find the edges",
    short: "Find",
    eyebrow: "2 / Find",
    heading: "Where is it really?",
    description:
      "Wherever a scan line steps up onto the tab, that is an edge. The edges and the raised area give the part's actual center, rotation and top height.",
    focus: "Step edges in the height map",
    mechanism: "Height step → edge → pose",
    detail:
      "The tab is found by thresholding halfway between the base and the top, then taking the centroid and principal axis of the raised points. No camera or reflectivity contrast is needed.",
  },
  {
    start: 14,
    end: 19,
    title: "Correct the path",
    short: "Correct",
    eyebrow: "3 / Correct",
    heading: "Move the weld to the part",
    description:
      "The programmed seam came from CAD and the fixture. It gets shifted and rotated onto the tab the scan actually found, and the focus is moved to the measured top height.",
    focus: "Programmed path → found path",
    mechanism: "Translate + rotate + refocus",
    detail:
      "Because the scan used the same galvos and F-theta lens, there is no camera-to-laser calibration offset in between. The correction is applied directly as a scanner coordinate transform.",
  },
  {
    start: 19,
    end: 26,
    title: "Weld",
    short: "Weld",
    eyebrow: "4 / Weld",
    heading: "Hit the seam",
    description:
      "The laser fires along the corrected path. The cyan beam stays coaxial, so the same sensor can keep reading keyhole depth while the bead is laid down.",
    focus: "Corrected seam on the tab",
    mechanism: "Weld on the found seam",
    detail:
      "The graph compares how far each path is from the real seam. On a 1.2 mm wide tab, a few tenths of a millimeter of fixture error is most of the margin.",
  },
  {
    start: 26,
    end: 30,
    title: "Check the seam",
    short: "Check",
    eyebrow: "5 / Check",
    heading: "Scan it again",
    description:
      "A quick LDD pass across the finished bead shows the crown profile and where the bead landed relative to the tab edges.",
    focus: "Bead cross-section",
    mechanism: "Post-weld profile",
    detail:
      "The same scan that found the part now inspects it: bead position, crown height, and whether the bead stayed clear of the tab edge.",
  },
] as const;

export function clampTargetTime(time: number): number {
  return Math.max(
    0,
    Math.min(TARGET_DURATION, Number.isFinite(time) ? time : 0),
  );
}

export function targetStepAt(time: number): number {
  const index = TARGET_STEPS.findIndex(
    (step) => clampTargetTime(time) < step.end,
  );
  return index < 0 ? TARGET_STEPS.length - 1 : index;
}

function noise(index: number): number {
  let n = Math.imul(index + 7, 668265263);
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return (((n ^ (n >>> 16)) >>> 0) / 4294967295) * 2 - 1;
}

export function toWorld(pose: Pose, [u, v]: XY): XY {
  const c = Math.cos(pose.angle);
  const s = Math.sin(pose.angle);
  return [pose.x + u * c - v * s, pose.y + u * s + v * c];
}

export function toLocal(pose: Pose, [x, y]: XY): XY {
  const c = Math.cos(pose.angle);
  const s = Math.sin(pose.angle);
  const dx = x - pose.x;
  const dy = y - pose.y;
  return [dx * c + dy * s, -dx * s + dy * c];
}

export function outline(pose: Pose): XY[] {
  const u = PART.length / 2;
  const v = PART.width / 2;
  return (
    [
      [-u, -v],
      [u, -v],
      [u, v],
      [-u, v],
      [-u, -v],
    ] as XY[]
  ).map((p) => toWorld(pose, p));
}

export function seam(pose: Pose): [XY, XY] {
  return [
    toWorld(pose, [-SEAM_HALF_LENGTH, 0]),
    toWorld(pose, [SEAM_HALF_LENGTH, 0]),
  ];
}

export function partHeightAt(x: number, y: number, index = 0): number {
  const [u, v] = toLocal(ACTUAL, [x, y]);
  const on = Math.abs(u) <= PART.length / 2 && Math.abs(v) <= PART.width / 2;
  return Math.round((on ? ACTUAL.height : 0) + 6 * noise(index));
}

export const SCAN: ScanPoint[] = Array.from(
  { length: SCAN_LINES * SCAN_POINTS },
  (_, index) => {
    const line = Math.floor(index / SCAN_POINTS);
    const step = index % SCAN_POINTS;
    const column = line % 2 ? SCAN_POINTS - 1 - step : step;
    const x = -FIELD.x + (column / (SCAN_POINTS - 1)) * 2 * FIELD.x;
    const y = FIELD.y - (line / (SCAN_LINES - 1)) * 2 * FIELD.y;
    return { x, y, z: partHeightAt(x, y, index), line };
  },
);

export function scanLine(line: number, count = SCAN_POINTS): ScanPoint[] {
  return SCAN.slice(line * SCAN_POINTS, line * SCAN_POINTS + count).sort(
    (a, b) => a.x - b.x,
  );
}

const THRESHOLD = Math.max(...SCAN.map((p) => p.z)) / 2;

export function lineEdges(points: ScanPoint[]): number[] {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const edges: number[] = [];
  for (let i = 1; i < sorted.length; i += 1) {
    const a = sorted[i - 1];
    const b = sorted[i];
    if (a.z < THRESHOLD !== b.z < THRESHOLD)
      edges.push(a.x + ((THRESHOLD - a.z) / (b.z - a.z)) * (b.x - a.x));
  }
  return edges;
}

export const EDGES: XY[] = Array.from({ length: SCAN_LINES }, (_, line) => {
  const points = scanLine(line);
  return lineEdges(points).map((x) => [x, points[0].y] as XY);
}).flat();

export function measurePose(points: ScanPoint[]): Pose {
  const top = points.filter((p) => p.z >= THRESHOLD);
  const x = top.reduce((sum, p) => sum + p.x, 0) / top.length;
  const y = top.reduce((sum, p) => sum + p.y, 0) / top.length;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (const p of top) {
    sxx += (p.x - x) ** 2;
    syy += (p.y - y) ** 2;
    sxy += (p.x - x) * (p.y - y);
  }
  return {
    x,
    y,
    angle: 0.5 * Math.atan2(2 * sxy, sxx - syy),
    height: Math.round(top.reduce((sum, p) => sum + p.z, 0) / top.length),
  };
}

export const FOUND = measurePose(SCAN);

export function distanceToSeam(point: XY): number {
  return Math.abs(toLocal(ACTUAL, point)[1]);
}

export function blendPose(a: Pose, b: Pose, f: number): Pose {
  return {
    x: a.x + (b.x - a.x) * f,
    y: a.y + (b.y - a.y) * f,
    angle: a.angle + (b.angle - a.angle) * f,
    height: a.height + (b.height - a.height) * f,
  };
}

export interface OffsetSample {
  travel: number;
  nominal: number;
  corrected: number;
}

export const OFFSETS: OffsetSample[] = Array.from({ length: 61 }, (_, i) => {
  const f = i / 60;
  const u = -SEAM_HALF_LENGTH + f * 2 * SEAM_HALF_LENGTH;
  return {
    travel: Math.round(f * 2 * SEAM_HALF_LENGTH * 100) / 100,
    nominal: Math.round(distanceToSeam(toWorld(NOMINAL, [u, 0])) * 1000),
    corrected: Math.round(distanceToSeam(toWorld(FOUND, [u, 0])) * 1000),
  };
});

export interface ProfileSample {
  across: number;
  corrected: number;
  nominal: number;
}

function beadCenter(pose: Pose): number {
  return toLocal(ACTUAL, toWorld(pose, [0, 0]))[1];
}

function surfaceAcross(v: number, center: number, index: number): number {
  const onPart = Math.abs(v) <= PART.width / 2;
  const bead = 70 * Math.exp(-(((v - center) / BEAD_HALF_WIDTH) ** 2) * 2);
  const base = onPart ? ACTUAL.height : 0;
  return Math.round(base + (onPart ? bead : 0) + 5 * noise(index + 9000));
}

export const BEAD_PROFILE: ProfileSample[] = Array.from(
  { length: 61 },
  (_, i) => {
    const across = Math.round((-1 + i / 30) * 100) / 100;
    return {
      across,
      corrected: surfaceAcross(across, beadCenter(FOUND), i),
      nominal: surfaceAcross(across, beadCenter(NOMINAL), i + 61),
    };
  },
);

export const RESULT = {
  nominalMissUm: Math.max(...OFFSETS.map((s) => s.nominal)),
  correctedMissUm: Math.max(...OFFSETS.map((s) => s.corrected)),
  nominalEdgeUm: Math.round(
    (PART.width / 2 - Math.abs(beadCenter(NOMINAL)) - BEAD_HALF_WIDTH) * 1000,
  ),
  correctedEdgeUm: Math.round(
    (PART.width / 2 - Math.abs(beadCenter(FOUND)) - BEAD_HALF_WIDTH) * 1000,
  ),
  focusShiftUm: FOUND.height - NOMINAL.height,
};

function progress(time: number, start: number, end: number): number {
  return Math.max(
    0,
    Math.min(1, (clampTargetTime(time) - start) / (end - start)),
  );
}

export function targetingAt(time: number) {
  const t = clampTargetTime(time);
  const step = targetStepAt(t);
  const scanned = Math.min(
    SCAN.length,
    Math.floor(progress(t, 0, 10) * SCAN.length + 1e-9),
  );
  const line = Math.min(SCAN_LINES - 1, Math.floor(scanned / SCAN_POINTS));
  const lineCount = scanned - line * SCAN_POINTS;
  const centerLine = Math.round(
    ((FIELD.y - FOUND.y) / (2 * FIELD.y)) * (SCAN_LINES - 1),
  );
  const correction = progress(t, 14.5, 18.5);
  const weld = progress(t, 19.5, 25.5);
  const check = progress(t, 26.3, 29.5);
  const path = blendPose(NOMINAL, FOUND, correction);
  const [start, end] = seam(FOUND);
  const spot: XY =
    step <= 0 && scanned > 0
      ? [SCAN[scanned - 1].x, SCAN[scanned - 1].y]
      : step === 3
        ? [
            start[0] + (end[0] - start[0]) * weld,
            start[1] + (end[1] - start[1]) * weld,
          ]
        : step === 4
          ? toWorld(ACTUAL, [0, -1 + 2 * check])
          : toWorld(path, [0, 0]);
  return {
    step,
    scanned,
    line,
    profile:
      step === 0
        ? scanLine(line, lineCount || (scanned ? SCAN_POINTS : 0))
        : scanLine(centerLine),
    edges: step === 0 ? [] : lineEdges(scanLine(centerLine)),
    correction,
    path,
    weld,
    check,
    spot,
    offsets: OFFSETS.slice(0, step === 3 ? Math.floor(weld * 60) + 1 : 61),
    bead: BEAD_PROFILE.slice(0, Math.floor(check * 60) + 1),
  };
}

export const signed = (value: number, digits = 2) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
export const degrees = (FOUND.angle * 180) / Math.PI;

export function targetFacts(step: number): [string, string, string][] {
  return [
    [
      ["Scan field", `${FIELD.x * 2} × ${FIELD.y * 2}`, "mm"],
      ["Points", SCAN.length.toLocaleString(), `${SCAN_LINES} lines`],
      ["Processing laser", "Off", "LDD only"],
    ],
    [
      ["Center shift", `${signed(FOUND.x)} / ${signed(FOUND.y)}`, "mm"],
      ["Rotation", signed(degrees, 1), "°"],
      ["Top height", FOUND.height.toString(), "µm"],
    ],
    [
      ["Programmed miss", RESULT.nominalMissUm.toString(), "µm"],
      ["Corrected miss", RESULT.correctedMissUm.toString(), "µm"],
      ["Focus shift", signed(RESULT.focusShiftUm, 0), "µm"],
    ],
    [
      ["Seam length", (SEAM_HALF_LENGTH * 2).toFixed(1), "mm"],
      ["Tolerance", `±${TOLERANCE_UM}`, "µm"],
      ["Worst offset", RESULT.correctedMissUm.toString(), "µm"],
    ],
    [
      ["Bead to edge", RESULT.correctedEdgeUm.toString(), "µm"],
      ["Without targeting", RESULT.nominalEdgeUm.toString(), "µm"],
      ["Tab width", PART.width.toFixed(1), "mm"],
    ],
  ][step] as [string, string, string][];
}
