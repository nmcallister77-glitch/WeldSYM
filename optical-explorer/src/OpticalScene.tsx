import { useEffect, useMemo, useRef } from "react";
import type { ComponentRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Edges, Html, Line, OrbitControls } from "@react-three/drei";
import {
  CatmullRomCurve3,
  DoubleSide,
  Group,
  Mesh,
  Quaternion,
  Vector3,
} from "three";
import {
  DEPTH_SCALE,
  measurementPath,
  phaseAt,
  returnPath,
  signalAt,
} from "./simulation";
import type { Point } from "./simulation";

const ORANGE = "#ff722e";
const CYAN = "#53e4ec";
const PUMP = "#e9e77c";
const UP = new Vector3(0, 1, 0);
const COIL: Point[] = Array.from({ length: 180 }, (_, i) => {
  const f = i / 179;
  return [
    -4.65 + f * 2.6,
    2.7 + Math.sin(f * Math.PI * 10) * 0.58,
    0.58 * Math.cos(f * Math.PI * 10),
  ];
});

function Segment({
  from,
  to,
  color,
  radius = 0.026,
  opacity = 1,
}: {
  from: Point;
  to: Point;
  color: string;
  radius?: number;
  opacity?: number;
}) {
  const a = new Vector3(...from);
  const b = new Vector3(...to);
  const distance = a.distanceTo(b);
  if (distance < 0.00001) return null;
  const rotation = new Quaternion().setFromUnitVectors(
    UP,
    b.clone().sub(a).normalize(),
  );
  return (
    <mesh position={a.add(b).multiplyScalar(0.5)} quaternion={rotation}>
      <cylinderGeometry args={[radius, radius, distance, 8]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        depthWrite={false}
      />
    </mesh>
  );
}

function Beam({
  points,
  color,
  time,
  reverse = false,
  radius = 0.022,
  count = 9,
}: {
  points: Point[];
  color: string;
  time: number;
  reverse?: boolean;
  radius?: number;
  count?: number;
}) {
  const particles = useRef<Group>(null);
  const path = reverse ? returnPath(points) : points;
  const vectors = path.map((p) => new Vector3(...p));
  const lengths = vectors.slice(1).map((v, i) => v.distanceTo(vectors[i]));
  const total = lengths.reduce((a, b) => a + b, 0);
  useFrame(() => {
    particles.current?.children.forEach((particle, index) => {
      let d = ((time * 0.45 + index / count) % 1) * total;
      let segment = 0;
      while (segment < lengths.length - 1 && d > lengths[segment])
        d -= lengths[segment++];
      particle.position
        .copy(vectors[segment])
        .lerp(
          vectors[segment + 1],
          lengths[segment] ? d / lengths[segment] : 0,
        );
    });
  });
  return (
    <group>
      {points.length > 20 ? (
        <Line
          points={points}
          color={color}
          lineWidth={2}
          transparent
          opacity={0.8}
        />
      ) : (
        points
          .slice(1)
          .map((point, i) => (
            <Segment
              key={i}
              from={points[i]}
              to={point}
              color={color}
              radius={radius}
              opacity={0.7}
            />
          ))
      )}
      <group ref={particles}>
        {Array.from({ length: count }, (_, i) => (
          <mesh key={i}>
            <sphereGeometry args={[reverse ? 0.067 : 0.048, 8, 8]} />
            <meshBasicMaterial color={reverse ? "#e2ffff" : color} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

function Label({
  position,
  title,
  subtitle,
  active,
  accent = false,
}: {
  position: Point;
  title: string;
  subtitle: string;
  active: boolean;
  accent?: boolean;
}) {
  return (
    <Html
      position={position}
      center
      distanceFactor={19}
      zIndexRange={[30, 0]}
      style={{ pointerEvents: "none" }}
    >
      <div
        className={`scene-label ${active ? "visible" : ""} ${accent ? "cyan" : ""}`}
      >
        <span>{title}</span>
        <small>{subtitle}</small>
      </div>
    </Html>
  );
}

function Housing({
  position,
  size,
  color = "#253039",
}: {
  position: Point;
  size: Point;
  color?: string;
}) {
  return (
    <mesh position={position}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} metalness={0.65} roughness={0.38} />
      <Edges color="#62717a" threshold={15} />
    </mesh>
  );
}

function Mirror({
  previous,
  position,
  next,
  color = "#97b7be",
}: {
  previous: Point;
  position: Point;
  next: Point;
  color?: string;
}) {
  const incoming = new Vector3(...position)
    .sub(new Vector3(...previous))
    .normalize();
  const outgoing = new Vector3(...next)
    .sub(new Vector3(...position))
    .normalize();
  const normal = incoming.sub(outgoing).normalize();
  const quaternion = new Quaternion().setFromUnitVectors(
    new Vector3(0, 0, 1),
    normal,
  );
  return (
    <group position={position} quaternion={quaternion}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.44, 0.44, 0.1, 32]} />
        <meshStandardMaterial color="#46525b" metalness={0.9} roughness={0.3} />
      </mesh>
      <mesh position={[0, 0, 0.056]}>
        <circleGeometry args={[0.36, 32]} />
        <meshStandardMaterial
          color={color}
          metalness={0.8}
          roughness={0.15}
          side={DoubleSide}
        />
      </mesh>
    </group>
  );
}

function Fiber({ active, time }: { active: boolean; time: number }) {
  const curve = useMemo(
    () => new CatmullRomCurve3(COIL.map((p) => new Vector3(...p))),
    [],
  );
  return (
    <group>
      <Housing position={[-3.3, 1.7, 0]} size={[3.3, 0.24, 2.2]} />
      {[-4.5, -2.2].map((x) => (
        <Housing key={x} position={[x, 2.1, 0]} size={[0.15, 0.6, 1.6]} />
      ))}
      <mesh>
        <tubeGeometry args={[curve, 200, 0.095, 8, false]} />
        <meshStandardMaterial
          color={active ? "#edb444" : "#5e6962"}
          emissive={PUMP}
          emissiveIntensity={active ? 0.6 : 0}
          transparent
          opacity={0.68}
          roughness={0.3}
        />
      </mesh>
      {active && (
        <Beam
          points={COIL}
          color={ORANGE}
          time={time}
          radius={0.018}
          count={18}
        />
      )}
    </group>
  );
}

function Workpiece({
  time,
  active,
  depth,
  tip,
}: {
  time: number;
  active: boolean;
  depth: number;
  tip: Point;
}) {
  const pool = useRef<Mesh>(null);
  const plume = useRef<Group>(null);
  const d = depth * DEPTH_SCALE;
  useFrame(() => {
    if (pool.current) {
      const pulse = 1 + Math.sin(time * 21) * 0.07;
      pool.current.scale.set(pulse, pulse, 1);
    }
    plume.current?.children.forEach((p, i) => {
      const f = (time * 0.65 + i / 12) % 1;
      p.position.set(
        tip[0] + Math.sin(i * 17 + time) * f * 0.48,
        0.15 + f * 1.25,
        tip[2] + Math.cos(i * 7) * f * 0.4,
      );
      p.scale.setScalar((1 - f) * 0.045);
    });
  });
  return (
    <group>
      <Housing
        position={[4.4, -2.83, 0]}
        size={[3.35, 0.2, 2.7]}
        color="#313c43"
      />
      <Housing
        position={[4.4, -1.33, -0.86]}
        size={[3.35, 2.8, 0.98]}
        color="#3c474c"
      />
      <Housing
        position={[3.01, -1.33, 0.45]}
        size={[0.57, 2.8, 1.65]}
        color="#404c51"
      />
      <Housing
        position={[5.8, -1.33, 0.45]}
        size={[0.55, 2.8, 1.65]}
        color="#404c51"
      />
      {Array.from({ length: 7 }, (_, i) => (
        <Segment
          key={i}
          from={[2.73, -0.15 - i * 0.38, 1.29]}
          to={[3.29, -0.15 - i * 0.38, 1.29]}
          color="#74838b"
          radius={0.008}
          opacity={0.6}
        />
      ))}
      {active && (
        <group>
          <mesh
            ref={pool}
            position={[tip[0], 0.125, tip[2]]}
            rotation={[-Math.PI / 2, 0, 0]}
          >
            <ringGeometry args={[0.2, 0.65, 48]} />
            <meshStandardMaterial
              color="#ffb02e"
              emissive="#ff5900"
              emissiveIntensity={1.8}
              side={DoubleSide}
            />
          </mesh>
          {d > 0.001 && (
            <mesh position={[tip[0], 0.12 - d / 2, tip[2]]}>
              <cylinderGeometry
                args={[0.22, 0.045, d, 32, 1, true, Math.PI / 2, Math.PI]}
              />
              <meshStandardMaterial
                color="#dc6928"
                emissive="#ff4600"
                emissiveIntensity={0.75}
                side={DoubleSide}
              />
            </mesh>
          )}
          <mesh position={tip}>
            <sphereGeometry args={[0.06, 16, 16]} />
            <meshBasicMaterial color="#aefaff" />
          </mesh>
          <Segment
            from={[tip[0] + 0.82, 0.12, 0.25]}
            to={[tip[0] + 0.82, tip[1], 0.25]}
            color={CYAN}
            radius={0.008}
          />
          {[0.12, tip[1]].map((y, i) => (
            <Segment
              key={i}
              from={[tip[0] + 0.73, y, 0.25]}
              to={[tip[0] + 0.92, y, 0.25]}
              color={CYAN}
              radius={0.008}
            />
          ))}
          <Html
            position={[tip[0] + 1, 0.12 - d / 2, 0.25]}
            center
            zIndexRange={[25, 0]}
          >
            <span className="depth-tag">{depth.toLocaleString()} µm</span>
          </Html>
          <group ref={plume}>
            {Array.from({ length: 12 }, (_, i) => (
              <mesh key={i}>
                <sphereGeometry args={[1, 6, 6]} />
                <meshBasicMaterial
                  color={i % 3 ? "#ffa346" : "#cdd1cf"}
                  transparent
                  opacity={0.5}
                />
              </mesh>
            ))}
          </group>
          <pointLight
            position={[tip[0], 0.7, tip[2]]}
            color="#ff702a"
            intensity={3}
            distance={4}
          />
        </group>
      )}
    </group>
  );
}

export type CameraView = "overview" | "scanner" | "keyhole";

function CameraRig({ view, resetKey }: { view: CameraView; resetKey: number }) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  useEffect(() => {
    if (!controls.current) return;
    const positions = {
      overview: [12.5, 10, 18.5],
      scanner: [9, 8, 11],
      keyhole: [8, 3.2, 7],
    } as const;
    const targets = {
      overview: [-0.65, 0.9, 0],
      scanner: [2.2, 2.8, 0],
      keyhole: [4.4, -0.7, 0],
    } as const;
    controls.current.object.position.fromArray(positions[view]);
    controls.current.target.fromArray(targets[view]);
    controls.current.update();
  }, [view, resetKey]);
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={2}
      maxDistance={45}
      maxPolarAngle={Math.PI * 0.9}
    />
  );
}

function Assembly({ time, labels }: { time: number; labels: boolean }) {
  const phase = phaseAt(time);
  const signal = signalAt(time);
  const path = measurementPath(time, signal.active ? signal.sample.depth : 0);
  const [sensor, dichroic, xMirror, yMirror, lens, surface, tip] = path;
  const output: Point[] = [COIL[COIL.length - 1], [-1.5, 3.45, 0], dichroic];
  return (
    <group>
      <gridHelper args={[30, 60, "#28383c", "#1b282d"]} position={[0, -3, 0]} />
      <Housing
        position={[-3.7, 1.35, 0]}
        size={[8.4, 0.15, 3.35]}
        color="#1e282e"
      />
      {[1.9, 2.7, 3.5].map((y, i) => (
        <group key={i}>
          <Housing position={[-7, y, 0]} size={[1.15, 0.5, 0.9]} />
          <mesh position={[-6.405, y, 0]} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.13, 0.13, 0.08, 16]} />
            <meshBasicMaterial color={PUMP} />
          </mesh>
          {Array.from({ length: 5 }, (_, j) => (
            <Housing
              key={j}
              position={[-7.4 + j * 0.2, y + 0.27, 0]}
              size={[0.05, 0.12, 0.9]}
              color="#46535a"
            />
          ))}
          <Beam
            points={[
              [-6.37, y, 0],
              [-5.2, 2.7, 0],
            ]}
            color={PUMP}
            time={time}
            count={5}
          />
        </group>
      ))}
      <mesh position={[-5.15, 2.7, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[0.18, 0.5, 0.75, 24]} />
        <meshStandardMaterial
          color="#64737a"
          metalness={0.8}
          roughness={0.25}
        />
      </mesh>
      <Beam
        points={[[-4.78, 2.7, 0], COIL[0]]}
        color={PUMP}
        time={time}
        count={3}
      />
      <Fiber active={phase >= 1} time={time} />
      {phase >= 1 && (
        <Beam
          points={output}
          color={ORANGE}
          time={time}
          radius={0.045}
          count={6}
        />
      )}
      <Housing
        position={[2.4, 2.13, 0]}
        size={[5.25, 0.18, 2.1]}
        color="#253038"
      />
      <Housing
        position={[2.4, 3.25, -1.05]}
        size={[5.25, 2.4, 0.12]}
        color="#243039"
      />
      <mesh position={[2.4, 3.4, 0]}>
        <boxGeometry args={[5.3, 2.55, 2.15]} />
        <meshBasicMaterial
          color="#789aab"
          transparent
          opacity={0.025}
          depthWrite={false}
        />
        <Edges color="#52666f" />
      </mesh>
      <Housing
        position={[0.1, 3.45, -2.85]}
        size={[1.1, 0.75, 0.7]}
        color="#23414a"
      />
      <Mirror
        previous={sensor}
        position={dichroic}
        next={xMirror}
        color="#83cfcb"
      />
      <Mirror previous={dichroic} position={xMirror} next={yMirror} />
      <Mirror previous={xMirror} position={[3.65, 4.15, 0]} next={lens} />
      {[xMirror, yMirror].map((p, i) => (
        <Segment
          key={i}
          from={[p[0], p[1], -0.1]}
          to={[p[0], p[1], -1]}
          color="#65747e"
          radius={0.12}
        />
      ))}
      <mesh position={[4.35, 2.5, 0]}>
        <cylinderGeometry args={[0.52, 0.52, 0.18, 40]} />
        <meshStandardMaterial
          color="#84c5cc"
          transparent
          opacity={0.35}
          metalness={0.3}
          roughness={0.1}
          depthWrite={false}
        />
        <Edges color="#accdd2" />
      </mesh>
      <mesh position={[4.35, 2.6, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.55, 0.09, 8, 40]} />
        <meshStandardMaterial color="#5d6b75" metalness={0.8} roughness={0.3} />
      </mesh>
      {phase >= 2 && (
        <group>
          <Beam
            points={[
              dichroic,
              xMirror,
              yMirror,
              lens,
              ...(signal.active ? [surface, tip] : []),
            ]}
            color={ORANGE}
            time={time}
            radius={0.06}
            count={10}
          />
          <Beam
            points={signal.active ? path : path.slice(0, 5)}
            color={CYAN}
            time={time}
            radius={0.016}
            count={10}
          />
          {signal.active && (
            <Beam
              points={path}
              color={CYAN}
              time={time}
              reverse
              radius={0.009}
              count={7}
            />
          )}
        </group>
      )}
      <Workpiece
        time={time}
        active={signal.active}
        depth={signal.sample.depth}
        tip={tip}
      />
      <Label
        position={[-7, 4.7, 0]}
        title="PUMP DIODES"
        subtitle="976 nm · optical pumping"
        active={labels}
      />
      <Label
        position={[-5.1, 1, 1.2]}
        title="FIBER COMBINER"
        subtitle="Multiple inputs → one fiber"
        active={labels}
      />
      <Label
        position={[-3.1, 4.1, 0]}
        title="Yb-DOPED FIBER"
        subtitle="Double-clad gain medium"
        active={labels && phase >= 1}
      />
      <Label
        position={[-0.3, 4.9, -2.3]}
        title="LDD / OCT SENSOR"
        subtitle="Sample + reference interference"
        active={labels && phase >= 2}
        accent
      />
      <Label
        position={[0, 2, 1.6]}
        title="DICHROIC"
        subtitle="Collinear beam combination"
        active={labels && phase >= 2}
      />
      <Label
        position={[2.1, 4.3, 0.1]}
        title="X GALVO"
        subtitle="First steering axis"
        active={labels && phase >= 2}
      />
      <Label
        position={[4, 5.25, 0]}
        title="Y GALVO"
        subtitle="Second steering axis"
        active={labels && phase >= 2}
      />
      <Label
        position={[5.75, 2.5, 0]}
        title="F-THETA LENS"
        subtitle="Focus onto the workpiece"
        active={labels && phase >= 2}
      />
      <Label
        position={[4.3, -3.35, 1]}
        title="METAL WORKPIECE"
        subtitle="Cutaway · depth exaggerated"
        active={labels && phase >= 3}
      />
    </group>
  );
}

export default function OpticalScene({
  time,
  labels,
  view,
  resetKey,
}: {
  time: number;
  labels: boolean;
  view: CameraView;
  resetKey: number;
}) {
  return (
    <Canvas
      camera={{ position: [12.5, 10, 18.5], fov: 43, near: 0.1, far: 150 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, alpha: true }}
      fallback={
        <div className="canvas-fallback">
          WebGL is unavailable. Enable hardware acceleration to explore the 3D
          optical train.
        </div>
      }
    >
      <ambientLight intensity={1.2} />
      <directionalLight position={[0, 10, 8]} intensity={3} color="#e0e7e8" />
      <directionalLight position={[-5, 4, -5]} intensity={2} color="#7194aa" />
      <Assembly time={time} labels={labels} />
      <CameraRig view={view} resetKey={resetKey} />
    </Canvas>
  );
}
