import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import {
  distanceToSeam,
  FIELD,
  FOUND,
  OFFSETS,
  PART,
  RESULT,
  degrees,
  SCAN_LINES,
  SEAM_HALF_LENGTH,
  signed,
  TARGET_STEPS,
  targetingAt,
  TOLERANCE_UM,
  toWorld,
} from "./targeting";

const tick = { fill: "#7d9198", fontSize: 9 };
const grid = (
  <CartesianGrid stroke="#29383e" strokeDasharray="2 5" vertical={false} />
);

function blendedOffsets(time: number) {
  const { path } = targetingAt(time);
  return OFFSETS.map((sample, i) => ({
    ...sample,
    corrected: Math.round(
      distanceToSeam(
        toWorld(path, [-SEAM_HALF_LENGTH + (i / 60) * 2 * SEAM_HALF_LENGTH, 0]),
      ) * 1000,
    ),
  }));
}

export function TargetChart({
  time,
  playing,
}: {
  time: number;
  playing: boolean;
}) {
  const state = targetingAt(time);
  const { step } = state;
  const offsets = step === 2 ? blendedOffsets(time) : state.offsets;
  const current = offsets.at(-1);
  const live = playing && (step === 0 || step === 3 || step === 4);
  const heading = [
    "Height profile",
    "Edges found",
    "Distance off the real seam",
    "Distance off the real seam",
    "Bead cross-section",
  ][step];
  const big = [
    [state.scanned.toLocaleString(), "points"],
    [`${signed(FOUND.x)}, ${signed(FOUND.y)}`, "mm"],
    [Math.max(...offsets.map((s) => s.corrected)).toString(), "µm miss"],
    [(current?.corrected ?? 0).toString(), "µm off"],
    [RESULT.correctedEdgeUm.toString(), "µm to edge"],
  ][step];
  const side = [
    ["Scan line", `${Math.min(state.line + 1, SCAN_LINES)} / ${SCAN_LINES}`],
    ["Rotation · top", `${signed(degrees, 1)}° · ${FOUND.height} µm`],
    ["As programmed", `${RESULT.nominalMissUm} µm`],
    ["As programmed", `${current?.nominal ?? 0} µm`],
    ["Without targeting", `${RESULT.nominalEdgeUm} µm`],
  ][step];
  const profileChart = step <= 1;
  return (
    <section
      className="chart-card active target-card"
      aria-label="LDD targeting measurements"
    >
      <div className="chart-heading">
        <span className="eyebrow">{heading}</span>
        <span className={`signal-status ${live ? "live" : ""}`}>
          <i />
          {live
            ? ["Scanning", "", "", "Welding", "Checking"][step]
            : TARGET_STEPS[step].short}
        </span>
      </div>
      <div className="depth-values">
        <div>
          <span className="depth-number">{big[0]}</span>
          <span className="unit">{big[1]}</span>
        </div>
        <div className="mean-value">
          <span>{side[0]}</span>
          <strong>{side[1]}</strong>
        </div>
      </div>
      <div className="chart-axis-title">
        {profileChart || step === 4 ? "Height (µm)" : "Off seam (µm)"}
      </div>
      <div className="chart-plot">
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          {profileChart ? (
            <LineChart
              data={state.profile}
              margin={{ top: 9, right: 8, bottom: 0, left: -21 }}
              accessibilityLayer
            >
              {grid}
              <XAxis
                type="number"
                dataKey="x"
                domain={[-FIELD.x, FIELD.x]}
                ticks={[-2, -1, 0, 1, 2]}
                tickLine={false}
                axisLine={false}
                tick={tick}
              />
              <YAxis
                domain={[0, 800]}
                ticks={[0, 400, 800]}
                tickLine={false}
                axisLine={false}
                tick={tick}
              />
              {state.edges.map((x) => (
                <ReferenceLine
                  key={x}
                  x={x}
                  stroke="#d8f27a"
                  strokeWidth={1.4}
                />
              ))}
              <Line
                type="linear"
                dataKey="z"
                stroke="#57dfe6"
                strokeWidth={1.6}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          ) : step === 4 ? (
            <LineChart
              data={state.bead}
              margin={{ top: 9, right: 8, bottom: 0, left: -21 }}
              accessibilityLayer
            >
              {grid}
              <XAxis
                type="number"
                dataKey="across"
                domain={[-1, 1]}
                ticks={[-1, -0.5, 0, 0.5, 1]}
                tickLine={false}
                axisLine={false}
                tick={tick}
              />
              <YAxis
                domain={[0, 800]}
                ticks={[0, 400, 800]}
                tickLine={false}
                axisLine={false}
                tick={tick}
              />
              {[-PART.width / 2, PART.width / 2].map((x) => (
                <ReferenceLine
                  key={x}
                  x={x}
                  stroke="#8b9a9e"
                  strokeDasharray="3 4"
                />
              ))}
              <Line
                type="linear"
                dataKey="nominal"
                stroke="#9aa6a9"
                strokeDasharray="4 4"
                strokeWidth={1.2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="linear"
                dataKey="corrected"
                stroke="#57dfe6"
                strokeWidth={1.6}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          ) : (
            <LineChart
              data={offsets}
              margin={{ top: 9, right: 8, bottom: 0, left: -21 }}
              accessibilityLayer
            >
              {grid}
              <ReferenceArea
                y1={0}
                y2={TOLERANCE_UM}
                fill="#d8f27a"
                fillOpacity={0.08}
                stroke="none"
              />
              <XAxis
                type="number"
                dataKey="travel"
                domain={[0, SEAM_HALF_LENGTH * 2]}
                ticks={[0, 0.5, 1, 1.5]}
                tickLine={false}
                axisLine={false}
                tick={tick}
              />
              <YAxis
                domain={[0, 600]}
                ticks={[0, 300, 600]}
                tickLine={false}
                axisLine={false}
                tick={tick}
              />
              <Line
                type="linear"
                dataKey="nominal"
                stroke="#9aa6a9"
                strokeDasharray="4 4"
                strokeWidth={1.2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="linear"
                dataKey="corrected"
                stroke="#d8f27a"
                strokeWidth={1.8}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
        {step === 0 && state.scanned === 0 && (
          <div className="chart-empty">
            Press play to start the scan
            <span>The cyan beam sweeps the fixture first</span>
          </div>
        )}
      </div>
      <div className="chart-foot">
        <span>
          {profileChart ? (
            <>
              <i className="cyan-dot" />
              Height
              {step === 1 && (
                <>
                  <i className="lime-dot" />
                  Edges
                </>
              )}
            </>
          ) : step === 4 ? (
            <>
              <i className="cyan-dot" />
              Targeted <i className="ghost-dot" />
              Untargeted
            </>
          ) : (
            <>
              <i className="lime-dot" />
              Found path <i className="ghost-dot" />
              Programmed
            </>
          )}
        </span>
        <span>
          {profileChart
            ? "Position (mm)"
            : step === 4
              ? "Across seam (mm)"
              : "Along seam (mm)"}
        </span>
      </div>
    </section>
  );
}

export function TargetPrinciple() {
  const steps = [
    [
      "Same optics, no offset",
      "The scan goes through the same galvos and F-theta lens as the weld, so a point the sensor sees is a point the laser can hit. No separate camera calibration to drift.",
    ],
    [
      "Height, not contrast",
      "Nickel on copper or shiny same-color parts are tough for 2D vision. LDD reads height, so the tab edge is a step in the data, as long as enough light comes back.",
    ],
    [
      "Small parts, tiny margins",
      `A ${PART.width} mm tab with a 0.3 mm bead leaves about ${((PART.width - 0.32) / 2).toFixed(2)} mm per side. Normal fixture and placement error can use most of that.`,
    ],
    [
      "Fix Z and check after",
      "The measured top height sets a focus offset, and a second pass across the bead checks where it landed.",
    ],
  ];
  return (
    <>
      <span className="eyebrow">Weld targeting</span>
      <h2>
        Why scan small
        <br />
        parts first?
      </h2>
      <p className="phase-description">
        Use the cyan beam like a touch probe made of light: map the part, find
        it, then weld where it actually is.
      </p>
      {steps.map(([title, body], i) => (
        <div className="principle-step" key={title}>
          <span>{String(i + 1).padStart(2, "0")}</span>
          <div>
            <h3>{title}</h3>
            <p>{body}</p>
          </div>
        </div>
      ))}
      <div className="principle-note">
        <i className="cyan-dot" />
        <p>
          Part offset, scan density and timing here are made up for the picture.
          Real scans trade point count against cycle time.
        </p>
      </div>
    </>
  );
}
