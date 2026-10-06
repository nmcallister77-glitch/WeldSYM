import { useEffect, useMemo, useRef } from "react";
import type { ComponentRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, Line, OrbitControls, RoundedBox } from "@react-three/drei";
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DoubleSide,
  Mesh,
} from "three";
import { Label, Segment } from "./OpticalScene";
import type { Point } from "./simulation";
import {
  ACTUAL,
  BEAD_PROFILE,
  EDGES,
  FOUND,
  NOMINAL,
  outline,
  PART,
  SCAN,
  seam,
  targetingAt,
} from "./targeting";
import type { Pose, XY } from "./targeting";

const ORANGE = "#ff722e";
const CYAN = "#53e4ec";
const LIME = "#d8f27a";
const LENS: Point = [0, 3.55, 0];

export type TargetView = "angle" | "top" | "close";

const at = ([x, y]: XY, height: number): Point => [x, height / 1000, y];

function Outline({
  pose,
  color,
  dashed = false,
  opacity = 1,
}: {
  pose: Pose;
  color: string;
  dashed?: boolean;
  opacity?: number;
}) {
  return (
    <Line
      points={outline(pose).map((p) => at(p, pose.height + 12))}
      color={color}
      lineWidth={dashed ? 1.4 : 2.2}
      dashed={dashed}
      dashSize={0.12}
      gapSize={0.08}
      transparent
      opacity={opacity}
    />
  );
}

function ScanCloud({ count }: { count: number }) {
  const geometry = useMemo(() => {
    const g = new BufferGeometry();
    const positions = new Float32Array(SCAN.length * 3);
    const colors = new Float32Array(SCAN.length * 3);
    const low = new Color("#2f7d84");
    const high = new Color("#9ff8fc");
    SCAN.forEach((p, i) => {
      positions.set([p.x, p.z / 1000 + 0.02, p.y], i * 3);
      const c = low.clone().lerp(high, Math.min(1, p.z / ACTUAL.height));
      colors.set([c.r, c.g, c.b], i * 3);
    });
    g.setAttribute("position", new BufferAttribute(positions, 3));
    g.setAttribute("color", new BufferAttribute(colors, 3));
    return g;
  }, []);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useEffect(() => geometry.setDrawRange(0, count), [geometry, count]);
  return (
    <points geometry={geometry}>
      <pointsMaterial size={0.045} vertexColors transparent opacity={0.9} />
    </points>
  );
}

function HotSpot({ position, time }: { position: Point; time: number }) {
  const ring = useRef<Mesh>(null);
  useFrame(() => {
    const pulse = 1 + Math.sin(time * 25) * 0.1;
    ring.current?.scale.set(pulse, pulse, 1);
  });
  return (
    <group position={position}>
      <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]}>
        <ringGeometry args={[0.04, 0.19, 32]} />
        <meshBasicMaterial color="#ffc23d" side={DoubleSide} />
      </mesh>
      <mesh position={[0, 0.03, 0]}>
        <sphereGeometry args={[0.07, 16, 16]} />
        <meshBasicMaterial color="#fff1b8" />
      </mesh>
      <pointLight color="#ff7b2c" intensity={2.5} distance={2.5} />
    </group>
  );
}

function CameraRig({ view, resetKey }: { view: TargetView; resetKey: number }) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const compact = useThree((state) => state.size.width < 600);
  useEffect(() => {
    if (!controls.current) return;
    const positions = {
      angle: compact ? [6.5, 6.5, 9.5] : [5.3, 4.6, 7.3],
      top: [0, compact ? 13 : 9.5, 0.01],
      close: compact ? [3.4, 2.8, 4.4] : [2.6, 2, 3.1],
    } as const;
    const targets = {
      angle: [0.15, 1.55, 0],
      top: [0, 0, 0],
      close: [ACTUAL.x, 0.45, ACTUAL.y],
    } as const;
    controls.current.object.position.fromArray(positions[view]);
    controls.current.target.fromArray(targets[view]);
    controls.current.update();
  }, [view, resetKey, compact]);
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={1.2}
      maxDistance={30}
      maxPolarAngle={Math.PI * 0.9}
    />
  );
}

function Fixture({
  time,
  labels,
  view,
}: {
  time: number;
  labels: boolean;
  view: TargetView;
}) {
  const state = targetingAt(time);
  const { step, scanned, spot, path, weld } = state;
  const compact = useThree((s) => s.size.width < 600);
  const top = ACTUAL.height;
  const [start, end] = seam(FOUND);
  const sample = SCAN[Math.max(0, scanned - 1)];
  const checkSample =
    BEAD_PROFILE[Math.min(60, Math.floor(state.check * 60))].corrected;
  const spotHeight =
    step === 0
      ? sample.z
      : step === 4
        ? checkSample
        : step === 3
          ? top
          : path.height;
  const target = at(spot, spotHeight);
  const scanning = step === 0 && scanned > 0 && scanned < SCAN.length;
  const checking = step === 4 && state.check > 0 && state.check < 1;
  const welding = step === 3 && weld > 0 && weld < 1;
  const weldEnd: XY = [
    start[0] + (end[0] - start[0]) * (step >= 4 ? 1 : weld),
    start[1] + (end[1] - start[1]) * (step >= 4 ? 1 : weld),
  ];
  const hotStart: XY = [
    weldEnd[0] - (end[0] - start[0]) * Math.min(weld, 0.12),
    weldEnd[1] - (end[1] - start[1]) * Math.min(weld, 0.12),
  ];
  const show = labels && !(compact && view === "close");
  return (
    <group>
      <gridHelper
        args={[16, 32, "#383d40", "#303538"]}
        position={[0, -0.8, 0]}
      />
      <RoundedBox
        args={[6.4, 0.5, 5.2]}
        radius={0.04}
        smoothness={2}
        position={[0, -0.25, 0]}
      >
        <meshStandardMaterial
          color="#b9785a"
          metalness={0.75}
          roughness={0.32}
        />
      </RoundedBox>
      {[-2.7, 2.7].map((x) =>
        [-2.1, 2.1].map((z) => (
          <mesh key={`${x}${z}`} position={[x, 0.01, z]}>
            <cylinderGeometry args={[0.16, 0.16, 0.04, 20]} />
            <meshStandardMaterial
              color="#4d5558"
              metalness={0.8}
              roughness={0.35}
            />
          </mesh>
        )),
      )}
      <mesh
        position={[ACTUAL.x, top / 2000, ACTUAL.y]}
        rotation={[0, -ACTUAL.angle, 0]}
      >
        <boxGeometry args={[PART.length, top / 1000, PART.width]} />
        <meshStandardMaterial
          color="#c9cfd1"
          metalness={0.65}
          roughness={0.28}
        />
      </mesh>
      {view !== "top" && (
        <>
          <RoundedBox
            args={[2.4, 1.1, 1.9]}
            radius={0.06}
            smoothness={2}
            position={[0, 4.4, 0]}
          >
            <meshStandardMaterial
              color="#8f989c"
              metalness={0.55}
              roughness={0.35}
            />
          </RoundedBox>
          <mesh position={[0, 3.72, 0]}>
            <cylinderGeometry args={[0.6, 0.68, 0.3, 40]} />
            <meshStandardMaterial
              color="#59656c"
              metalness={0.8}
              roughness={0.3}
            />
          </mesh>
          <mesh position={[0, 3.56, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[0.5, 40]} />
            <meshStandardMaterial
              color="#7fc6cf"
              metalness={0.3}
              roughness={0.08}
              transparent
              opacity={0.7}
              side={DoubleSide}
            />
          </mesh>
        </>
      )}

      <ScanCloud count={scanned} />
      <Outline
        pose={NOMINAL}
        color="#c3cbcd"
        dashed
        opacity={step >= 3 ? 0.35 : 0.8}
      />
      {step >= 1 && <Outline pose={FOUND} color={LIME} />}
      {step >= 1 &&
        EDGES.map((p, i) => (
          <mesh key={i} position={at(p, top / 2)}>
            <sphereGeometry args={[0.035, 8, 8]} />
            <meshBasicMaterial color={LIME} />
          </mesh>
        ))}
      {step >= 2 && (
        <>
          <Line
            points={seam(NOMINAL).map((p) => at(p, top + 14))}
            color="#e7ecec"
            lineWidth={1.4}
            dashed
            dashSize={0.1}
            gapSize={0.07}
            transparent
            opacity={step >= 3 ? 0.45 : 0.9}
          />
          <Line
            points={seam(path).map((p) => at(p, top + 16))}
            color={LIME}
            lineWidth={2.6}
          />
        </>
      )}
      {step === 2 && (
        <Html
          position={at(seam(path)[1], top + 450)}
          center
          zIndexRange={[25, 0]}
        >
          <span className="depth-tag target-tag">
            Δx {(FOUND.x * state.correction).toFixed(2)} · Δy{" "}
            {(FOUND.y * state.correction).toFixed(2)} mm ·{" "}
            {(((FOUND.angle * 180) / Math.PI) * state.correction).toFixed(1)}°
          </span>
        </Html>
      )}
      {step >= 3 && weld > 0 && (
        <>
          <Segment
            from={at(start, top + 20)}
            to={at(weldEnd, top + 20)}
            color="#d9a85c"
            radius={0.13}
            opacity={0.95}
          />
          {welding && (
            <Segment
              from={at(hotStart, top + 30)}
              to={at(weldEnd, top + 30)}
              color="#ff8a1f"
              radius={0.145}
            />
          )}
        </>
      )}
      {welding && <HotSpot position={target} time={time} />}
      {welding && (
        <Segment
          from={LENS}
          to={target}
          color={ORANGE}
          radius={0.05}
          opacity={0.85}
        />
      )}
      {(scanning || welding || checking) && (
        <>
          <Segment from={LENS} to={target} color={CYAN} radius={0.016} />
          <mesh position={target}>
            <sphereGeometry args={[0.045, 12, 12]} />
            <meshBasicMaterial color="#b4fbff" />
          </mesh>
        </>
      )}

      <Label
        position={[2.3, 4.5, 0]}
        title="SCANNER HEAD"
        subtitle="Same galvos + F-theta for scan and weld"
        active={show && view === "angle"}
      />
      <Label
        position={[-3.1, 1.6, -1.6]}
        title="LDD RASTER"
        subtitle="Laser off · height only"
        active={show && step === 0}
        accent
      />
      <Label
        position={[-2.6, -0.1, 2.9]}
        title="COPPER BUSBAR"
        subtitle="Base part, fixtured"
        active={show && view !== "close" && !compact}
      />
      <Label
        position={[ACTUAL.x + 0.6, 1.25, ACTUAL.y + 1.2]}
        title="NICKEL TAB"
        subtitle="2.4 × 1.2 mm · placed by hand"
        active={show && step === 0}
      />
      <Label
        position={[-1.9, 1.15, -1.2]}
        title="WHERE CAD SAYS"
        subtitle="Programmed position"
        active={show && step >= 1 && step <= 2}
      />
      <Label
        position={[ACTUAL.x + 1.9, 1.05, ACTUAL.y + 0.9]}
        title="WHERE LDD FOUND IT"
        subtitle={`${FOUND.height} µm tall · ${((FOUND.angle * 180) / Math.PI).toFixed(1)}°`}
        active={show && step === 1}
        accent
      />
      <Label
        position={[target[0], target[1] + 1.1, target[2]]}
        title={step === 4 ? "CHECK PASS" : "WELD ON THE FOUND SEAM"}
        subtitle={
          step === 4 ? "Profile across the bead" : "Cyan rides along coaxially"
        }
        active={show && (welding || checking)}
        accent={step === 4}
      />
    </group>
  );
}

export default function TargetingScene({
  time,
  labels,
  view,
  resetKey,
}: {
  time: number;
  labels: boolean;
  view: TargetView;
  resetKey: number;
}) {
  return (
    <Canvas
      camera={{ position: [5.3, 4.6, 7.3], fov: 43, near: 0.05, far: 100 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, alpha: true }}
      fallback={
        <div className="canvas-fallback">
          WebGL is unavailable. Enable hardware acceleration to explore the 3D
          fixture.
        </div>
      }
    >
      <ambientLight intensity={1.5} />
      <directionalLight position={[2, 10, 6]} intensity={3} color="#fff2e0" />
      <directionalLight
        position={[-5, 4, -5]}
        intensity={1.8}
        color="#acccdf"
      />
      <directionalLight position={[4, 2, 8]} intensity={1.1} color="#e5eff4" />
      <Fixture time={time} labels={labels} view={view} />
      <CameraRig view={view} resetKey={resetKey} />
    </Canvas>
  );
}
