export type Point = [number, number, number];

export const DURATION = 32;
export const INTERACTION_START = 20;
export const SAMPLE_INTERVAL_MS = 0.5;
export const DEPTH_SCALE = 0.0017;
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

export interface DepthSample {
  time: number;
  depth: number;
  mean: number;
}

export function depthAt(timeMs: number): number {
  const t = Math.max(0, timeMs);
  const growth = 1 - Math.exp(-t / 5);
  const fluctuations =
    65 * Math.sin(t * 2 * Math.PI * 0.13) +
    42 * Math.sin(t * 2 * Math.PI * 0.37) +
    55 * noise(Math.round(t / SAMPLE_INTERVAL_MS));
  return Math.round(Math.max(0, growth * (1200 + fluctuations)));
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
    return { time, depth, mean: Math.round(sum / (index - from + 1)) };
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
    [0.1, 3.45, -2.5],
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
