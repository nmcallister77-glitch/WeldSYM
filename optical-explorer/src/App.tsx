import { Component, useEffect, useRef, useState } from "react";
import type { CSSProperties, ErrorInfo, ReactNode } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import OpticalScene from "./OpticalScene";
import type { CameraView } from "./OpticalScene";
import {
  clampTime,
  DURATION,
  INTERACTION_START,
  phaseAt,
  PHASES,
  signalAt,
} from "./simulation";

function Icon({
  name,
  size = 18,
}: {
  name:
    | "play"
    | "pause"
    | "reset"
    | "focus"
    | "labels"
    | "arrow"
    | "close"
    | "help"
    | "download"
    | "cube";
  size?: number;
}) {
  const paths = {
    play: <path d="m8 5 11 7-11 7z" />,
    pause: (
      <>
        <path d="M8 5v14M16 5v14" />
      </>
    ),
    reset: (
      <>
        <path d="M4 11a8 8 0 1 1 2 6M4 4v7h7" />
      </>
    ),
    focus: (
      <>
        <path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" />
        <circle cx="12" cy="12" r="3" />
      </>
    ),
    labels: (
      <>
        <path d="M3 5h12l6 7-6 7H3z" />
        <circle cx="7" cy="12" r="1" />
      </>
    ),
    arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    help: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M9.5 9a2.5 2.5 0 1 1 3.3 2.4c-.8.4-.8 1.1-.8 2.1m0 3h.01" />
      </>
    ),
    download: (
      <>
        <path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />
      </>
    ),
    cube: (
      <>
        <path d="m12 3 9 5v9l-9 5-9-5V8zm0 10v9M3 8l9 5 9-5M8 5l9 5v4" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}

class SceneBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("3D scene unavailable", error, info.componentStack);
  }
  render() {
    return this.state.failed ? (
      <div className="canvas-fallback">
        <Icon name="cube" size={32} />
        <h2>The 3D view could not start.</h2>
        <p>
          Enable WebGL or hardware acceleration, then reload. The phase guide
          and depth data remain available.
        </p>
      </div>
    ) : (
      this.props.children
    );
  }
}

function ReturnMap({ time }: { time: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    const image = context.createImageData(241, 48);
    signalAt(time).history.forEach((sample, x) => {
      for (let y = 0; y < 48; y += 1) {
        const depth = (y / 47) * 1600;
        const floor = Math.exp(-(((depth - sample.depth) / 80) ** 2));
        const surface = 0.45 * Math.exp(-((depth / 50) ** 2));
        const texture = 0.65 + 0.35 * Math.sin(x * 7.3 + y * 5.1) ** 2;
        const intensity = Math.min(1, (floor + surface) * texture);
        const pixel = (y * 241 + x) * 4;
        image.data[pixel] = 60 + intensity * 195;
        image.data[pixel + 1] = 12 + intensity ** 2 * 220;
        image.data[pixel + 2] = 60 - intensity * 35;
        image.data[pixel + 3] = 255;
      }
    });
    context.putImageData(image, 0, 0);
  }, [time]);
  return (
    <div className="return-map">
      <span>Return intensity · illustrative</span>
      <canvas
        ref={canvas}
        width={241}
        height={48}
        role="img"
        aria-label="Synthetic return intensity: time runs left to right, depth increases downward from 0 to 1600 micrometers. The bright band tracks the same keyhole depth."
      />
      <small>0 → 120 ms · depth increases downward</small>
    </div>
  );
}

function DepthChart({ time, playing }: { time: number; playing: boolean }) {
  const signal = signalAt(time);
  const { sample, active, history } = signal;
  return (
    <section
      className={`chart-card ${active ? "active" : ""}`}
      aria-label="Live LDD depth measurements"
    >
      <div className="chart-heading">
        <span className="eyebrow">Live depth</span>
        <span className={`signal-status ${active && playing ? "live" : ""}`}>
          <i />
          {active ? (playing ? "Reading" : "Paused") : "Waiting for the weld"}
        </span>
      </div>
      <div className="depth-values">
        <div>
          <span className="depth-number">
            {active ? sample.depth.toLocaleString() : "—"}
          </span>
          <span className="unit">µm</span>
        </div>
        <div className="mean-value">
          <span>10 ms average</span>
          <strong>
            {active ? sample.mean.toLocaleString() : "—"} <small>µm</small>
          </strong>
        </div>
      </div>
      <div className="chart-axis-title">Depth (µm)</div>
      <div className="chart-plot">
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <LineChart
            data={history}
            margin={{ top: 9, right: 8, bottom: 0, left: -21 }}
            accessibilityLayer
          >
            <CartesianGrid
              stroke="#29383e"
              strokeDasharray="2 5"
              vertical={false}
            />
            <XAxis
              type="number"
              dataKey="time"
              domain={[0, 120]}
              ticks={[0, 30, 60, 90, 120]}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "#7d9198", fontSize: 9 }}
            />
            <YAxis
              domain={[0, 1600]}
              ticks={[0, 800, 1600]}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "#7d9198", fontSize: 9 }}
            />
            <Tooltip
              contentStyle={{
                background: "#172328",
                border: "1px solid #3b525b",
                borderRadius: 6,
                fontSize: 11,
              }}
              labelFormatter={(value) => `${value} ms`}
              formatter={(value, name) => [
                `${value} µm`,
                name === "depth" ? "Measured depth" : "10 ms average",
              ]}
              isAnimationActive={false}
            />
            {active && (
              <ReferenceLine y={1200} stroke="#829078" strokeDasharray="3 5" />
            )}
            <Line
              type="linear"
              dataKey="depth"
              stroke="#57dfe6"
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="linear"
              dataKey="mean"
              stroke="#d5e68b"
              strokeWidth={1.3}
              dot={false}
              isAnimationActive={false}
            />
            {active && (
              <ReferenceDot
                x={sample.time}
                y={sample.depth}
                r={3}
                fill="#a1f5f7"
                stroke="#152a30"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
        {!active && (
          <div className="chart-empty">
            Signal begins at material interaction
            <span>Jump to phase 04 to explore</span>
          </div>
        )}
      </div>
      <div className="chart-foot">
        <span>
          <i className="cyan-dot" />
          Depth <i className="lime-dot" />
          Average
        </span>
        <span>Time (ms)</span>
      </div>
      <ReturnMap time={time} />
    </section>
  );
}

function Guide({ onClose }: { onClose: () => void }) {
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const previous = document.activeElement;
    close.current?.focus();
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab") {
        event.preventDefault();
        close.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => {
      window.removeEventListener("keydown", handler);
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, [onClose]);
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section
        className="guide-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="guide-title"
        onClick={(event) => event.stopPropagation()}
      >
        <button
          ref={close}
          className="icon-button modal-close"
          onClick={onClose}
          aria-label="Close guide"
        >
          <Icon name="close" />
        </button>
        <span className="eyebrow">A QUICK FIELD GUIDE</span>
        <h2 id="guide-title">Follow the light.</h2>
        <p>
          Explore how a fiber laser and inline coherent imaging (ICI / LDD)
          share an optical train to process and measure a weld at the same time.
        </p>
        <div className="guide-controls">
          <p>
            <strong>Orbit</strong>
            <span>Left-drag or one finger</span>
          </p>
          <p>
            <strong>Pan</strong>
            <span>Right-drag or two fingers</span>
          </p>
          <p>
            <strong>Zoom</strong>
            <span>Scroll or pinch</span>
          </p>
          <p>
            <strong>Playback</strong>
            <span>Space to play / pause · R to reset</span>
          </p>
        </div>
        <h3>One signal, two views</h3>
        <p>
          The depth at the chart cursor and the 3D keyhole floor use the same
          deterministic sample. Scrubbing reproduces the same result. A 10 ms
          trailing average separates the nominal depth from rapid fluctuations.
        </p>
        <h3>Reading the model</h3>
        <p>
          Colors make infrared beams visible. The optical layout, photon motion,
          and cutaway depth are illustrative, with a slowed 120 ms acquisition
          over phase 04. Depth noise is synthetic; this is an educational model,
          not a process prediction or sensor feed.
        </p>
        <p className="guide-note">
          LDD uses low-coherence interferometry: the returned sample is compared
          with a reference inside the sensor. It does not infer depth from the
          animated pulse travel time.
        </p>
      </section>
    </div>
  );
}

export default function App() {
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [labels, setLabels] = useState(true);
  const [view, setView] = useState<CameraView>("overview");
  const [resetKey, setResetKey] = useState(0);
  const [guide, setGuide] = useState(false);
  const [tab, setTab] = useState<"journey" | "principle">("journey");
  const timeRef = useRef(0);
  const phase = phaseAt(time);
  const current = PHASES[phase];
  const signal = signalAt(time);

  const seek = (next: number) => {
    const clamped = clampTime(next);
    timeRef.current = clamped;
    setTime(clamped);
  };
  const reset = () => {
    seek(0);
    setPlaying(false);
  };
  const togglePlay = () => {
    if (timeRef.current >= DURATION) seek(0);
    setPlaying((value) => !value);
  };

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let previous: number | undefined;
    let lastDraw = 0;
    const animate = (now: number) => {
      if (previous !== undefined)
        timeRef.current = clampTime(
          timeRef.current + Math.min((now - previous) / 1000, 0.1) * speed,
        );
      previous = now;
      if (now - lastDraw >= 1000 / 30 || timeRef.current >= DURATION) {
        setTime(timeRef.current);
        lastDraw = now;
      }
      if (timeRef.current >= DURATION) {
        setPlaying(false);
        return;
      }
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [playing, speed]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const tag = (event.target as HTMLElement).tagName;
      if (guide || ["INPUT", "SELECT", "BUTTON", "TEXTAREA"].includes(tag))
        return;
      if (event.code === "Space") {
        event.preventDefault();
        if (timeRef.current >= DURATION) {
          timeRef.current = 0;
          setTime(0);
        }
        setPlaying((value) => !value);
      }
      if (event.key.toLowerCase() === "r") {
        timeRef.current = 0;
        setTime(0);
        setPlaying(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [guide]);

  function downloadData() {
    const rows = [
      "time_ms,depth_um,trailing_10ms_mean_um",
      ...signal.history.map(
        (sample) => `${sample.time},${sample.depth},${sample.mean}`,
      ),
    ];
    const url = URL.createObjectURL(
      new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "ldd-synthetic-depth.csv";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <a
          className="brand"
          href="#"
          aria-label="Laser sandbox home"
          onClick={(event) => {
            event.preventDefault();
            reset();
            setView("overview");
            setResetKey((key) => key + 1);
          }}
        >
          <strong>Laser sandbox</strong>
          <span className="brand-name">let’s see what’s going on inside</span>
        </a>
        <button className="guide-button" onClick={() => setGuide(true)}>
          <Icon name="help" size={16} />
          <span>Controls &amp; a few notes</span>
        </button>
      </header>

      <main className="workspace">
        <section
          className="viewport"
          aria-label="Interactive 3D optical system"
        >
          <div className="viewport-heading">
            <h1>
              {view === "overview"
                ? "Follow the beams."
                : view === "scanner"
                  ? "Inside the scanner."
                  : "A slice through the weld."}
            </h1>
            <p>Grab the model. Move around. Scrub back and forth.</p>
          </div>
          <div className="scene-container">
            <SceneBoundary>
              <OpticalScene
                time={time}
                labels={labels}
                view={view}
                resetKey={resetKey}
              />
            </SceneBoundary>
          </div>
          <div className="view-tools">
            <button
              className={`icon-button ${labels ? "selected" : ""}`}
              onClick={() => setLabels((value) => !value)}
              title="Toggle component labels"
              aria-label="Toggle component labels"
              aria-pressed={labels}
            >
              <Icon name="labels" />
            </button>
            <button
              className="icon-button"
              onClick={() => {
                setView("overview");
                setResetKey((key) => key + 1);
              }}
              title="Reset camera"
              aria-label="Reset camera"
            >
              <Icon name="focus" />
            </button>
          </div>
          <div className="camera-presets" role="group" aria-label="Camera view">
            {(["overview", "scanner", "keyhole"] as const).map((preset) => (
              <button
                key={preset}
                aria-pressed={view === preset}
                className={view === preset ? "selected" : ""}
                onClick={() => {
                  setView(preset);
                  setResetKey((key) => key + 1);
                }}
              >
                {preset === "overview" && <Icon name="cube" size={13} />}
                {preset}
              </button>
            ))}
          </div>
          <div className="beam-legend">
            <span>
              <i className="pump-line" />
              Pump <small>976 nm</small>
            </span>
            <span>
              <i className="process-line" />
              Processing <small>1,070 nm</small>
            </span>
            <span>
              <i className="measure-line" />
              LDD <small>840 nm</small>
            </span>
          </div>
          <div className="interaction-hint">
            <span>↔</span> Drag to orbit <b>·</b> scroll to zoom
          </div>
          <DepthChart time={time} playing={playing} />
          <div className="viewport-corner">
            Shapes and depth exaggerated to make things visible
          </div>
        </section>

        <aside className="inspector">
          <div
            className="inspector-tabs"
            role="tablist"
            aria-label="Explainer content"
          >
            <button
              role="tab"
              id="journey-tab"
              aria-controls="inspector-panel"
              aria-selected={tab === "journey"}
              className={tab === "journey" ? "active" : ""}
              onClick={() => setTab("journey")}
            >
              What’s happening
            </button>
            <button
              role="tab"
              id="principle-tab"
              aria-controls="inspector-panel"
              aria-selected={tab === "principle"}
              className={tab === "principle" ? "active" : ""}
              onClick={() => setTab("principle")}
            >
              How LDD works
            </button>
          </div>
          <div
            className="inspector-body"
            id="inspector-panel"
            role="tabpanel"
            aria-labelledby={`${tab}-tab`}
          >
            {tab === "journey" ? (
              <>
                <div className="phase-eyebrow">
                  <span className="eyebrow">{current.eyebrow}</span>
                  <span className="phase-counter">
                    {String(phase + 1).padStart(2, "0")}
                    <small> / 04</small>
                  </span>
                </div>
                <h2 key={phase}>{current.heading}</h2>
                <p className="phase-description">{current.description}</p>
                <div className="component-focus">
                  <span className="focus-icon">
                    <Icon name="focus" size={21} />
                  </span>
                  <div>
                    <span className="eyebrow">Look here</span>
                    <strong>{current.focus}</strong>
                  </div>
                  <span className="focus-pulse" />
                </div>
                <div className="spec-list">
                  {current.facts.map(([label, value, unit]) => (
                    <div className="spec-row" key={label}>
                      <span>{label}</span>
                      <strong>
                        {value} <small>{unit}</small>
                      </strong>
                    </div>
                  ))}
                </div>
                <div className="mechanism">
                  <span className="eyebrow">A closer look</span>
                  <h3>{current.mechanism}</h3>
                  <p>{current.detail}</p>
                </div>
                <button
                  className="next-phase"
                  onClick={() => {
                    seek(phase < 3 ? PHASES[phase + 1].start : 0);
                    setPlaying(false);
                  }}
                >
                  <span>
                    {phase < 3 ? (
                      <>
                        Next<strong>{PHASES[phase + 1].title}</strong>
                      </>
                    ) : (
                      <>
                        Again?<strong>Back to the diodes</strong>
                      </>
                    )}
                  </span>
                  <Icon name="arrow" size={21} />
                </button>
              </>
            ) : (
              <>
                <span className="eyebrow">The cyan beam</span>
                <h2>
                  How do we know<br />how deep it is?
                </h2>
                <p className="phase-description">
                  Inline coherent imaging measures a keyhole through the very
                  optics used to make it.
                </p>
                <div className="principle-step">
                  <span>01</span>
                  <div>
                    <h3>Share the optical path</h3>
                    <p>
                      A wavelength-selective dichroic aligns low-power
                      measurement light with the high-power laser.
                    </p>
                  </div>
                </div>
                <div className="principle-step">
                  <span>02</span>
                  <div>
                    <h3>Read the reflection</h3>
                    <p>
                      The keyhole floor reflects a small part of the measurement
                      light back through the lens and both galvos.
                    </p>
                  </div>
                </div>
                <div className="principle-step">
                  <span>03</span>
                  <div>
                    <h3>Resolve the depth</h3>
                    <p>
                      Inside the OCT sensor, returned light interferes with a
                      reference. The optical path difference locates the floor
                      relative to the surface.
                    </p>
                  </div>
                </div>
                <div className="principle-note">
                  <i className="cyan-dot" />
                  <p>
                    Cyan and orange are visual guides. Both real beams are
                    infrared and invisible to the eye.
                  </p>
                </div>
                <button
                  className="next-phase"
                  onClick={() => {
                    seek(INTERACTION_START);
                    setPlaying(true);
                    setTab("journey");
                  }}
                >
                  <span>
                    Try it<strong>Watch the depth change</strong>
                  </span>
                  <Icon name="arrow" />
                </button>
              </>
            )}
          </div>
          <div className="inspector-footer">
            Synthetic signal · just for exploring the idea
          </div>
        </aside>
      </main>

      <footer className="playback">
        <div className="timeline-top">
          <div className="transport">
            <button
              className={`play-button ${playing ? "is-playing" : ""}`}
              onClick={togglePlay}
              aria-label={playing ? "Pause animation" : "Play animation"}
            >
              <Icon name={playing ? "pause" : "play"} size={17} />
              {playing ? "Pause" : "Play"}
            </button>
            <button className="reset-button" onClick={reset}>
              <Icon name="reset" size={16} />
              <span>Reset</span>
            </button>
            <div className="transport-separator" />
            <label className="speed-label">
              <span className="sr-only">Playback speed</span>
              <select
                value={speed}
                onChange={(event) => setSpeed(Number(event.target.value))}
              >
                <option value={0.5}>0.5×</option>
                <option value={1}>1×</option>
                <option value={2}>2×</option>
              </select>
            </label>
            <span className="time-display">
              {time.toFixed(1).padStart(4, "0")}
              <span> / {DURATION}.0 s</span>
            </span>
          </div>
          <div className="timeline-caption">
            <span>Jump to a stage, or drag the timeline.</span>
          </div>
          <button
            className="export-button"
            onClick={downloadData}
            disabled={!signal.active}
            title={
              signal.active
                ? "Download visible synthetic depth data"
                : "Available during phase 04"
            }
          >
            <Icon name="download" size={15} />
            <span>Save CSV</span>
          </button>
        </div>
        <div className="scrubber-wrap">
          <div className="timeline-track">
            <div style={{ width: `${(time / DURATION) * 100}%` }} />
          </div>
          {PHASES.slice(1).map((p) => (
            <i
              key={p.start}
              className="timeline-marker"
              style={{ left: `${(p.start / DURATION) * 100}%` }}
            />
          ))}
          <input
            className="scrubber"
            type="range"
            min={0}
            max={DURATION}
            step={0.01}
            value={time}
            aria-label="Simulation timeline"
            aria-valuetext={`${time.toFixed(1)} seconds, phase ${phase + 1}: ${current.title}`}
            onChange={(event) => {
              seek(Number(event.target.value));
              setPlaying(false);
            }}
            style={
              { "--progress": `${(time / DURATION) * 100}%` } as CSSProperties
            }
          />
        </div>
        <div className="phase-navigation">
          {PHASES.map((p, index) => (
            <button
              key={p.start}
              className={`phase-button ${phase === index ? "active" : ""} ${phase > index ? "complete" : ""}`}
              style={{ flex: p.end - p.start }}
              aria-current={phase === index ? "step" : undefined}
              onClick={() => {
                seek(p.start);
                setPlaying(false);
              }}
            >
              <span className="phase-number">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span className="phase-title">
                {p.title}
                <small>
                  {p.start.toString().padStart(2, "0")} —{" "}
                  {p.end.toString().padStart(2, "0")} SEC
                </small>
              </span>
            </button>
          ))}
        </div>
        <div className="bottom-note">
          <span>Orange makes the weld. Cyan measures it.</span>
          <span>
            Depth signal: procedural · Acquisition: 2 kHz · Display slowed for
            clarity
          </span>
        </div>
      </footer>
      {guide && <Guide onClose={() => setGuide(false)} />}
    </div>
  );
}
