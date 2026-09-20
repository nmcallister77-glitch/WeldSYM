# Laser sandbox

A standalone React, react-three-fiber, drei, and Recharts educational explainer for
inline coherent imaging (LDD / OCT) alongside a ytterbium fiber laser and 2D scanner.
It is independent of the Python weld prediction dashboard.

## Run

Requires Node.js 22.18+ (or 24+) and npm.

```sh
cd optical-explorer
npm ci
npm run dev
```

Open the URL Vite prints. To package **one self-contained, playable artifact**:

```sh
npm run build
```

Open `dist/index.html` directly in a WebGL-capable browser. JavaScript, CSS, and
all dependencies are bundled inline; the app needs no CDN, network, server,
external assets, or API at runtime. `npm run preview` also serves this build.
Generated output is not committed; CI uploads it as the `ldd-optical-explorer` artifact.

## Explore

- Play / Pause, Reset, 0.5× / 1× / 2× speed, and a 32-second scrubber.
- Click a phase to seek; scrubbing pauses playback.
- Left-drag to orbit, right-drag to pan, scroll to zoom (touch: one finger orbit,
  two fingers pan/pinch). Camera presets focus on the scanner or cutaway keyhole.
- Toggle labels; reset the camera independently of playback.
- On narrow screens labels focus on the current stage; camera presets bring the
  scanner or keyhole closer without covering the model with annotations.
- Space toggles playback and R resets when an input or button is not focused.
- Phase 04 plots depth, the trailing average, and an instantaneous 3D dimension.
  Save CSV exports the visible samples.
- A compact return-intensity map puts time left-to-right and depth downward,
  with a bright band at the same sampled depth. The colors are an illustrative
  intensity pattern, not a simulated OCT interferogram or measured reflectivity.

## Signal and optical model

The only animation clock is `time`, in seconds. Phases start at 0, 6, 12, and 20 s.
The final 12 s represent a slowed 120 ms acquisition at 2 kHz (0.5 ms/sample).
`signalAt(time)` selects one deterministic sample and the history up to that
sample. The chart, numeric reading, and keyhole geometry all consume it.
Scrubbing backwards removes future samples; restarting reproduces the same noise.
The trailing mean uses up to 20 samples, with a shorter window at startup.

Depth rises toward 1,200 µm and combines two high-frequency oscillations with
seeded hash noise. These are **synthetic educational data, not sensor readings
or a predictive welding model**. The cutaway depth is exaggerated.

The measurement beam runs from the OCT sensor through the dichroic, X and Y
galvos, and F-theta lens to the keyhole floor. Its return traverses the exact
same vertices in reverse. Animated mirror normals bisect incident and outgoing
directions; the optical layout and scan motion are schematic. Internal OCT
reference interference is explained in the sensor label and principle panel.
Pump diode banks feed the combiner from the side of the assembly; the active
fiber itself is cladding-pumped through the combiner.

Wavelengths and power are illustrative: 976 nm pump, 1,070 nm / 4 kW processing,
and 840 nm low-power measurement. Orange and cyan are false colors for invisible
infrared. Photon speed, material effects, lens behavior, and cavity shape are
visualizations, not ray-traced optics or fluid dynamics.

## Checks

```sh
npm run lint
npm run typecheck
npm test
npm run build
```

The tests cover phase boundaries, deterministic rewind, chart/geometry agreement
for every sample, exact return paths, and depth/average invariants. No credentials
are needed.
