export type Point = [number, number, number];

export const DURATION = 32;
export const INTERACTION_START = 20;
export const SAMPLE_INTERVAL_MS = 0.5;
export const DEPTH_SCALE = 0.0017;
export const STEEL_N = 3.5;
export const STEEL_K = 4.0;
export const MOUTH_RADIUS_UM = 150;
export const KEYHOLE_ONSET_MS = 6;
export const PHASES = [
  {
    start: 0,
    end: 6,
    title: "Photon generation",
    short: "Pump",
    eyebrow: "1 / Pump",
    heading: "Start with the diodes",
    description:
      "Side-mounted diode banks deliver pump light into a fiber combiner. Many low-power inputs become one concentrated source of optical energy.",
    focus: "Pump diode array",
    mechanism: "Electrical energy → pump photons",
    detail:
      "The combiner guides 976 nm pump light into the inner cladding of the active fiber. The diode modules sit beside the fiber assembly; pumping enters through the combiner.",
    facts: [
      ["Pump wavelength", "976", "nm"],
      ["Diode banks", "3", "modules"],
      ["Energy transfer", "Optical", "pumping"],
    ],
  },
  {
    start: 6,
    end: 12,
    title: "Fiber amplification",
    short: "Amplify",
    eyebrow: "2 / Amplify",
    heading: "Build up the power",
    description:
      "Pump light excites ytterbium ions inside the active fiber. Stimulated emission amplifies a guided seed into a high-power processing beam.",
    focus: "Yb-doped active fiber",
    mechanism: "Pump absorption → stimulated emission",
    detail:
      "The double-clad fiber confines the seed in its doped core while pump light travels through the surrounding cladding. The orange output represents invisible infrared light.",
    facts: [
      ["Processing wavelength", "1,070", "nm"],
      ["Illustrative power", "4.0", "kW"],
      ["Active ion", "Yb³⁺", "ytterbium"],
    ],
  },
  {
    start: 12,
    end: 20,
    title: "Beam steering",
    short: "Steer",
    eyebrow: "3 / Steer",
    heading: "Bring the beams together",
    description:
      "A dichroic mirror combines the cyan measurement beam with the orange processing beam. X and Y galvanometers steer both through an F-theta lens.",
    focus: "2D scanner head",
    mechanism: "Combine → steer → focus",
    detail:
      "The dichroic transmits the processing wavelength and reflects the measurement wavelength. Two rotating mirrors steer a shared focal spot across the workpiece.",
    facts: [
      ["Measurement wavelength", "840", "nm"],
      ["Steering axes", "X + Y", "galvos"],
      ["Focusing optic", "F-theta", "lens"],
    ],
  },
  {
    start: 20,
    end: 32,
    title: "Depth measurement",
    short: "Measure",
    eyebrow: "4 / Measure",
    heading: "Watch the keyhole",
    description:
      "The processing beam opens a vapor-filled keyhole. Coaxial LDD light probes its floor and returns along the same optical path, revealing depth as it changes.",
    focus: "Keyhole & melt pool",
    mechanism: "Probe → reflect → interfere",
    detail:
      "Returned sample light interferes with a reference inside the OCT sensor. Optical path difference yields depth relative to the surface. The cyan pulses show the return, not a time-of-flight measurement.",
    facts: [
      ["Sampling rate", "2,000", "Hz"],
      ["Nominal depth", "1,200", "µm"],
      ["Measurement", "Coherent", "interference"],
    ],
  },
] as const;

export function clampTime(time: number): number {
  return Math.max(0, Math.min(DURATION, Number.isFinite(time) ? time : 0));
}

export function phaseAt(time: number): number {
  const index = PHASES.findIndex((phase) => clampTime(time) < phase.end);
  return index < 0 ? 3 : index;
}

function noise(index: number): number {
  let n = Math.imul(index + 1, 374761393);
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return (((n ^ (n >>> 16)) >>> 0) / 4294967295) * 2 - 1;
}

function complexSquareRoot(real: number, imaginary: number): [number, number] {
  const magnitude = Math.hypot(real, imaginary);
  const rootReal = Math.sqrt(Math.max(0, (magnitude + real) / 2));
  const rootImaginary = Math.sqrt(Math.max(0, (magnitude - real) / 2));
  return [rootReal, imaginary < 0 ? -rootImaginary : rootImaginary];
}

function complexDivide(
  numeratorReal: number,
  numeratorImaginary: number,
  denominatorReal: number,
  denominatorImaginary: number,
): [number, number] {
  const denominator = denominatorReal ** 2 + denominatorImaginary ** 2;
  return [
    (numeratorReal * denominatorReal +
      numeratorImaginary * denominatorImaginary) /
      denominator,
    (numeratorImaginary * denominatorReal -
      numeratorReal * denominatorImaginary) /
      denominator,
  ];
}

export function fresnelAbsorptivity(
  cosTheta: number,
  n = STEEL_N,
  k = STEEL_K,
): number {
  const cosine = Math.min(1, Math.max(0, cosTheta));
  if (cosine === 0) return 0;

  const sineSquared = 1 - cosine ** 2;
  const indexSquaredReal = n ** 2 - k ** 2;
  const indexSquaredImaginary = 2 * n * k;
  const [transmittedReal, transmittedImaginary] = complexSquareRoot(
    indexSquaredReal - sineSquared,
    indexSquaredImaginary,
  );

  const [sReal, sImaginary] = complexDivide(
    cosine - transmittedReal,
    -transmittedImaginary,
    cosine + transmittedReal,
    transmittedImaginary,
  );
  const [pReal, pImaginary] = complexDivide(
    indexSquaredReal * cosine - transmittedReal,
    indexSquaredImaginary * cosine - transmittedImaginary,
    indexSquaredReal * cosine + transmittedReal,
    indexSquaredImaginary * cosine + transmittedImaginary,
  );
  const reflectance =
    (sReal ** 2 + sImaginary ** 2 + pReal ** 2 + pImaginary ** 2) / 2;
  return Math.min(1, Math.max(0, 1 - reflectance));
}

export interface RayPath {
  points: [number, number][];
  power: number[];
}

interface TracedRay {
  absorbed: number;
  path: RayPath;
}

function coneIntersection(
  x: number,
  z: number,
  directionX: number,
  directionZ: number,
  mouthRadiusUm: number,
  depthUm: number,
  minimumDistance: number,
): number | undefined {
  const slope = mouthRadiusUm / depthUm;
  const localRadius = mouthRadiusUm * (1 - z / depthUm);
  const a = directionX ** 2 - (slope * directionZ) ** 2;
  const b = 2 * (x * directionX + slope * localRadius * directionZ);
  const c = x ** 2 - localRadius ** 2;
  let roots: number[];

  if (Math.abs(a) < 1e-14) {
    roots = Math.abs(b) < 1e-30 ? [] : [-c / b];
  } else {
    const discriminant = b ** 2 - 4 * a * c;
    if (discriminant < 0) return undefined;
    const root = Math.sqrt(discriminant);
    roots = [(-b - root) / (2 * a), (-b + root) / (2 * a)];
  }

  for (const distance of roots.sort((left, right) => left - right)) {
    const hitX = x + directionX * distance;
    const hitZ = z + directionZ * distance;
    const atApex =
      Math.abs(hitX) <= minimumDistance &&
      Math.abs(hitZ - depthUm) <= minimumDistance;
    if (
      distance > minimumDistance &&
      hitZ > 0 &&
      (hitZ < depthUm || atApex)
    ) {
      return distance;
    }
  }
  return undefined;
}

function traceKeyholeRay(
  entryX: number,
  depthUm: number,
  mouthRadiusUm: number,
): TracedRay {
  const path: RayPath = { points: [[entryX, 0]], power: [1] };
  if (depthUm <= 0) return { absorbed: fresnelAbsorptivity(1), path };

  const scale = Math.max(depthUm, mouthRadiusUm, 1e-12);
  const minimumDistance = 1e-9 * scale;
  const slope = mouthRadiusUm / depthUm;
  const normalScale = 1 / Math.sqrt(1 + slope ** 2);
  let x = entryX;
  let z = 0;
  let directionX = 0;
  let directionZ = 1;
  let remainingPower = 1;

  for (let reflection = 0; reflection < 50; reflection += 1) {
    const distance = coneIntersection(
      x,
      z,
      directionX,
      directionZ,
      mouthRadiusUm,
      depthUm,
      minimumDistance,
    );

    if (directionZ < 0) {
      const exitDistance = -z / directionZ;
      if (
        exitDistance > minimumDistance &&
        (distance === undefined || exitDistance <= distance)
      ) {
        path.points.push([x + directionX * exitDistance, 0]);
        path.power.push(remainingPower);
        break;
      }
    }
    if (distance === undefined) break;

    const hitX = x + directionX * distance;
    const hitZ = z + directionZ * distance;
    const normalX = (hitX < 0 ? -1 : 1) * normalScale;
    const normalZ = slope * normalScale;
    const incidenceCosine = Math.abs(
      directionX * normalX + directionZ * normalZ,
    );
    remainingPower *= 1 - fresnelAbsorptivity(incidenceCosine);
    path.points.push([hitX, hitZ]);
    path.power.push(remainingPower);
    if (remainingPower < 1e-6) break;

    const normalProjection = directionX * normalX + directionZ * normalZ;
    directionX -= 2 * normalProjection * normalX;
    directionZ -= 2 * normalProjection * normalZ;
    x = hitX + minimumDistance * directionX;
    z = hitZ + minimumDistance * directionZ;
  }

  return { absorbed: 1 - remainingPower, path };
}

function traceKeyholeAbsorptance(
  depthUm: number,
  mouthRadiusUm: number,
): number {
  const flat = fresnelAbsorptivity(1);
  if (depthUm <= 0) return flat;
  const radialStep = (2 * mouthRadiusUm) / 200;
  const rings = Array.from({ length: 200 }, (_, index) => {
    const radius = (index + 0.5) * radialStep;
    return {
      radius,
      weight:
        Math.exp(-2 * radius ** 2 / mouthRadiusUm ** 2) *
        radius *
        radialStep,
    };
  });
  const totalWeight = rings.reduce((sum, ring) => sum + ring.weight, 0);
  return rings.reduce((absorbed, ring) => {
    const weight = ring.weight / totalWeight;
    return (
      absorbed +
      weight *
        (ring.radius >= mouthRadiusUm
          ? flat
          : traceKeyholeRay(ring.radius, depthUm, mouthRadiusUm).absorbed)
    );
  }, 0);
}

const ABSORPTANCE_STEP_UM = 10;
const MAX_KEYHOLE_DEPTH_UM = 1600;

function createAbsorptanceTable(mouthRadiusUm: number): number[] {
  const table: number[] = [];
  for (
    let index = 0;
    index <= MAX_KEYHOLE_DEPTH_UM / ABSORPTANCE_STEP_UM;
    index += 1
  ) {
    const traced = traceKeyholeAbsorptance(
      index * ABSORPTANCE_STEP_UM,
      mouthRadiusUm,
    );
    table.push(Math.max(table[table.length - 1] ?? 0, traced));
  }
  return table;
}

const DEFAULT_ABSORPTANCE_TABLE = createAbsorptanceTable(MOUTH_RADIUS_UM);
const ABSORPTANCE_TABLES = new Map<number, number[]>([
  [MOUTH_RADIUS_UM, DEFAULT_ABSORPTANCE_TABLE],
]);

export function keyholeAbsorptance(
  depthUm: number,
  mouthRadiusUm = MOUTH_RADIUS_UM,
): number {
  if (depthUm <= 0) return fresnelAbsorptivity(1);
  if (!(mouthRadiusUm > 0)) {
    throw new RangeError("mouthRadiusUm must be positive");
  }

  const depth = Math.min(MAX_KEYHOLE_DEPTH_UM, depthUm);
  let table = ABSORPTANCE_TABLES.get(mouthRadiusUm);
  if (!table) {
    table = createAbsorptanceTable(mouthRadiusUm);
    ABSORPTANCE_TABLES.set(mouthRadiusUm, table);
  }
  const tablePosition = depth / ABSORPTANCE_STEP_UM;
  const lowerIndex = Math.floor(tablePosition);
  const upperIndex = Math.min(
    table.length - 1,
    lowerIndex + 1,
  );
  const fraction = tablePosition - lowerIndex;
  return (
    table[lowerIndex] * (1 - fraction) +
    table[upperIndex] * fraction
  );
}

export function keyholeRayPath(
  offset: number,
  depthUm: number,
  mouthRadiusUm = MOUTH_RADIUS_UM,
): RayPath {
  if (offset <= -1 || offset >= 1) {
    throw new RangeError("offset must be between -1 and 1");
  }
  return traceKeyholeRay(
    offset * mouthRadiusUm,
    depthUm,
    mouthRadiusUm,
  ).path;
}

export interface DepthSample {
  time: number;
  depth: number;
  mean: number;
  absorptance: number;
  mode: "conduction" | "keyhole";
}

export function depthAt(timeMs: number): number {
  const t = Number.isFinite(timeMs) ? Math.max(0, timeMs) : 0;
  const conductionDepth =
    70 * (1 - Math.exp(-KEYHOLE_ONSET_MS / 2));
  let depth: number;
  if (t < KEYHOLE_ONSET_MS) {
    depth = 70 * (1 - Math.exp(-t / 2));
  } else {
    const growth = 1 - Math.exp(-(t - KEYHOLE_ONSET_MS) / 4);
    const fluctuations =
      65 * Math.sin(t * 2 * Math.PI * 0.13) +
      42 * Math.sin(t * 2 * Math.PI * 0.37) +
      55 * noise(Math.round(t / SAMPLE_INTERVAL_MS));
    depth =
      conductionDepth +
      growth * (1200 - conductionDepth + fluctuations);
  }
  return Math.min(MAX_KEYHOLE_DEPTH_UM, Math.max(0, Math.round(depth)));
}

export function modeAt(timeMs: number): "conduction" | "keyhole" {
  const t = Number.isFinite(timeMs) ? Math.max(0, timeMs) : 0;
  return t < KEYHOLE_ONSET_MS ? "conduction" : "keyhole";
}

export const DEPTH_SAMPLES: DepthSample[] = Array.from(
  { length: 241 },
  (_, index) => {
    const time = index * SAMPLE_INTERVAL_MS;
    const depth = depthAt(time);
    const from = Math.max(0, index - 19);
    let sum = 0;
    for (let i = from; i <= index; i += 1)
      sum += depthAt(i * SAMPLE_INTERVAL_MS);
    return {
      time,
      depth,
      mean: Math.round(sum / (index - from + 1)),
      absorptance: Math.round(keyholeAbsorptance(depth) * 1000) / 1000,
      mode: modeAt(time),
    };
  },
);

export function sampleIndexAt(time: number): number {
  return Math.min(
    DEPTH_SAMPLES.length - 1,
    Math.max(
      0,
      Math.floor(
        ((clampTime(time) - INTERACTION_START) * 10) / SAMPLE_INTERVAL_MS +
          1e-9,
      ),
    ),
  );
}

export function signalAt(time: number) {
  const active = clampTime(time) >= INTERACTION_START;
  const index = sampleIndexAt(time);
  return {
    active,
    index,
    sample: DEPTH_SAMPLES[index],
    history: active ? DEPTH_SAMPLES.slice(0, index + 1) : [],
  };
}

export function measurementPath(time: number, depth: number): Point[] {
  const scan = Math.max(0, clampTime(time) - 12);
  const x = 4.35 + 0.25 * Math.sin(scan * 0.7);
  const z = 0.12 * Math.sin(scan * 1.1);
  return [
    [-1.62, 3.45, -2.6],
    [0.9, 3.45, -2.6],
    [0.1, 3.45, -1.8],
    [0.1, 3.45, 0],
    [2.15, 3.45, 0],
    [3.65, 4.15, 0.05 * Math.sin(scan * 0.7)],
    [4.35 + 0.12 * Math.sin(scan * 0.7), 2.5, 0.06 * Math.sin(scan * 1.1)],
    [x, 0.12, z],
    [x, 0.12 - depth * DEPTH_SCALE, z],
  ];
}

export function returnPath(path: Point[]): Point[] {
  return [...path].reverse();
}
