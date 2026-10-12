import { useEffect, useMemo, useRef } from "react";
import type { ComponentRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  Edges,
  Html,
  Line,
  OrbitControls,
  RoundedBox,
} from "@react-three/drei";
import {
  CatmullRomCurve3,
  DoubleSide,
  Group,
  Matrix4,
  Mesh,
  Quaternion,
  Shape,
  Vector3,
} from "three";
import {
  DEPTH_SCALE,
  MOUTH_RADIUS_UM,
  keyholeRayPath,
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
const MOUTH_HALF_WIDTH = 0.32;
const KEYHOLE_RAY_OFFSETS = [-0.72, -0.28, 0.45];
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
      if (reverse)
        particle.quaternion.setFromUnitVectors(
          UP,
          vectors[segment + 1].clone().sub(vectors[segment]).normalize(),
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
            {reverse ? (
              <coneGeometry args={[0.075, 0.2, 8]} />
            ) : (
              <sphereGeometry args={[0.048, 8, 8]} />
            )}
            <meshBasicMaterial color={reverse ? "#8efaff" : color} />
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
  color = "#90999c",
}: {
  position: Point;
  size: Point;
  color?: string;
}) {
  return (
    <RoundedBox position={position} args={size} radius={0.025} smoothness={2}>
      <meshStandardMaterial color={color} metalness={0.3} roughness={0.43} />
    </RoundedBox>
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

function Galvo({
  previous,
  position,
  next,
  axis,
  scale = 1,
}: {
  previous: Point;
  position: Point;
  next: Point;
  axis: Point;
  scale?: number;
}) {
  const incoming = new Vector3(...position)
    .sub(new Vector3(...previous))
    .normalize();
  const outgoing = new Vector3(...next)
    .sub(new Vector3(...position))
    .normalize();
  const normal = incoming.sub(outgoing).normalize();
  const shaft = new Vector3(...axis);
  shaft.sub(normal.clone().multiplyScalar(normal.dot(shaft))).normalize();
  const side = new Vector3().crossVectors(shaft, normal);
  const quaternion = new Quaternion().setFromRotationMatrix(
    new Matrix4().makeBasis(side, shaft, normal),
  );
  const s = scale;
  return (
    <group position={position} quaternion={quaternion}>
      <mesh>
        <boxGeometry args={[0.5 * s, 0.62 * s, 0.035 * s]} />
        <meshStandardMaterial color="#2f3a40" metalness={0.9} roughness={0.3} />
      </mesh>
      {[1, -1].map((face) => (
        <mesh
          key={face}
          position={[0, 0, face * 0.019 * s]}
          rotation={[0, face > 0 ? 0 : Math.PI, 0]}
        >
          <planeGeometry args={[0.46 * s, 0.58 * s]} />
          <meshStandardMaterial
            color="#c3dde2"
            metalness={0.95}
            roughness={0.06}
          />
        </mesh>
      ))}
      <mesh position={[0, 0.41 * s, 0]}>
        <cylinderGeometry args={[0.04 * s, 0.04 * s, 0.2 * s, 12]} />
        <meshStandardMaterial color="#c9cfd1" metalness={0.9} roughness={0.2} />
      </mesh>
      <mesh position={[0, 0.58 * s, 0]}>
        <boxGeometry args={[0.44 * s, 0.16 * s, 0.44 * s]} />
        <meshStandardMaterial
          color="#5f6b70"
          metalness={0.7}
          roughness={0.35}
        />
      </mesh>
      <mesh position={[0, 1.03 * s, 0]}>
        <cylinderGeometry args={[0.19 * s, 0.19 * s, 0.75 * s, 28]} />
        <meshStandardMaterial color="#1f262a" metalness={0.5} roughness={0.4} />
      </mesh>
      <mesh position={[0, 0.78 * s, 0]}>
        <cylinderGeometry args={[0.195 * s, 0.195 * s, 0.05 * s, 28]} />
        <meshStandardMaterial color="#b9c3c6" metalness={0.9} roughness={0.2} />
      </mesh>
      <mesh position={[0, 1.47 * s, 0]}>
        <cylinderGeometry args={[0.15 * s, 0.17 * s, 0.13 * s, 24]} />
        <meshStandardMaterial
          color="#3c474c"
          metalness={0.6}
          roughness={0.35}
        />
      </mesh>
    </group>
  );
}

function ClearShell({
  position,
  size,
  frame = true,
}: {
  position: Point;
  size: Point;
  frame?: boolean;
}) {
  const [w, h, d] = size;
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={size} />
        <meshPhysicalMaterial
          color="#d7eef2"
          transparent
          opacity={0.1}
          roughness={0.05}
          metalness={0}
          clearcoat={1}
          side={DoubleSide}
          depthWrite={false}
        />
        <Edges color="#b9cacc" />
      </mesh>
      {frame &&
        [-1, 1].map((sx) =>
          [-1, 1].map((sz) => (
            <Housing
              key={`${sx}${sz}`}
              position={[(sx * w) / 2, 0, (sz * d) / 2]}
              size={[0.07, h, 0.07]}
              color="#7b858a"
            />
          )),
        )}
      {frame &&
        [-1, 1].map((sz) => (
          <Housing
            key={`top-${sz}`}
            position={[0, h / 2, (sz * d) / 2]}
            size={[w, 0.07, 0.07]}
            color="#7b858a"
          />
        ))}
      {frame &&
        [-1, 1].map((sx) => (
          <Housing
            key={`side-${sx}`}
            position={[(sx * w) / 2, h / 2, 0]}
            size={[0.07, 0.07, d]}
            color="#7b858a"
          />
        ))}
    </group>
  );
}

function FocusStage({ carriage }: { carriage: number }) {
  const z = -2.6;
  return (
    <group>
      <Housing
        position={[-1.72, 2.94, z]}
        size={[0.36, 0.36, 0.36]}
        color="#2a3236"
      />
      <mesh position={[-1.5, 2.94, z]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.07, 0.07, 0.1, 16]} />
        <meshStandardMaterial color="#9aa5a9" metalness={0.8} roughness={0.3} />
      </mesh>
      <mesh position={[-0.5, 2.94, z]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.035, 0.035, 1.9, 12]} />
        <meshStandardMaterial
          color="#d0d6d8"
          metalness={0.95}
          roughness={0.25}
        />
      </mesh>
      {Array.from({ length: 24 }, (_, i) => (
        <mesh
          key={i}
          position={[-1.42 + i * 0.08, 2.94, z]}
          rotation={[0, Math.PI / 2, 0]}
        >
          <torusGeometry args={[0.037, 0.008, 6, 14]} />
          <meshStandardMaterial
            color="#8e999d"
            metalness={0.9}
            roughness={0.3}
          />
        </mesh>
      ))}
      <Housing
        position={[0.48, 2.94, z]}
        size={[0.12, 0.3, 0.32]}
        color="#4f5a5f"
      />
      {[-0.27, 0.27].map((dz) => (
        <Housing
          key={dz}
          position={[-0.5, 2.8, z + dz]}
          size={[2.0, 0.05, 0.07]}
          color="#9aa4a8"
        />
      ))}
      <group position={[carriage, 0, 0]}>
        <Housing
          position={[0, 3.0, z]}
          size={[0.32, 0.14, 0.66]}
          color="#c07a3a"
        />
        <Housing
          position={[0, 3.18, z]}
          size={[0.1, 0.24, 0.1]}
          color="#59656a"
        />
        <mesh position={[0, 3.45, z]} rotation={[0, Math.PI / 2, 0]}>
          <torusGeometry args={[0.19, 0.045, 10, 32]} />
          <meshStandardMaterial
            color="#59656a"
            metalness={0.8}
            roughness={0.3}
          />
        </mesh>
        <mesh position={[0, 3.45, z]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.18, 0.18, 0.05, 28]} />
          <meshStandardMaterial
            color="#84c5cc"
            transparent
            opacity={0.45}
            roughness={0.05}
            depthWrite={false}
          />
        </mesh>
      </group>
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
  const offset = tip[0] - 4.4;
  const section = useMemo(() => {
    const shape = new Shape();
    shape.moveTo(-1.9, -2.8);
    shape.lineTo(1.9, -2.8);
    shape.lineTo(1.9, 0);
    if (d > 0) {
      shape.lineTo(offset + MOUTH_HALF_WIDTH, 0);
      shape.lineTo(offset, -d);
      shape.lineTo(offset - MOUTH_HALF_WIDTH, 0);
    }
    shape.lineTo(-1.9, 0);
    shape.closePath();
    return shape;
  }, [d, offset]);
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
  const rays = useMemo(
    () =>
      d > 0
        ? KEYHOLE_RAY_OFFSETS.map((rayOffset) => {
            const ray = keyholeRayPath(rayOffset, depth);
            const points = ray.points.map(([x, z]): Point => [
              tip[0] + (x * MOUTH_HALF_WIDTH) / MOUTH_RADIUS_UM,
              0.12 - z * DEPTH_SCALE,
              0.37,
            ]);
            return { points, power: ray.power };
          })
        : [],
    [d, depth, tip],
  );
  return (
    <group>
      {active &&
        rays.map((ray, r) => (
          <group key={r}>
            <Segment
              from={[ray.points[0][0], 0.95, 0.37]}
              to={ray.points[0]}
              color={ORANGE}
              radius={0.007}
              opacity={0.95}
            />
            {ray.points.slice(1).map((point, i) => (
              <Segment
                key={i}
                from={ray.points[i]}
                to={point}
                color={ORANGE}
                radius={0.007}
                opacity={0.2 + 0.75 * ray.power[i]}
              />
            ))}
          </group>
        ))}
      <mesh position={[4.4, 0.12, -0.9]}>
        <extrudeGeometry
          args={[
            section,
            { depth: 1.25, bevelEnabled: false, curveSegments: 12 },
          ]}
        />
        <meshStandardMaterial color="#9dabad" metalness={0.3} roughness={0.5} />
      </mesh>
      {Array.from({ length: 23 }, (_, i) => (
        <group key={i}>
          <Segment
            from={[2.51, -0.04 - i * 0.113, 0.36]}
            to={[3.92, -0.04 - i * 0.113, 0.36]}
            color="#d4d7d7"
            radius={0.002}
            opacity={0.3}
          />
          <Segment
            from={[4.92, -0.04 - i * 0.113, 0.36]}
            to={[6.29, -0.04 - i * 0.113, 0.36]}
            color="#d4d7d7"
            radius={0.002}
            opacity={0.3}
          />
        </group>
      ))}
      {active && (
        <group>
          <mesh
            ref={pool}
            position={[tip[0], 0.125, tip[2]]}
            rotation={[-Math.PI / 2, 0, 0]}
          >
            <ringGeometry args={[0.2, 0.56, 48]} />
            <meshBasicMaterial color="#ffbd32" side={DoubleSide} />
          </mesh>
          {[0, 1, 2].map((i) => (
            <mesh
              key={i}
              position={[tip[0], 0.124 - i * 0.001, tip[2]]}
              rotation={[-Math.PI / 2, 0, 0]}
            >
              <ringGeometry args={[0.5 + i * 0.04, 0.55 + i * 0.05, 48]} />
              <meshBasicMaterial
                color={i === 0 ? "#ff7a16" : "#ff3c00"}
                transparent
                opacity={0.45 - i * 0.13}
                side={DoubleSide}
                depthWrite={false}
              />
            </mesh>
          ))}
          {d > 0.001 && (
            <mesh position={[tip[0], 0.12 - d / 2, tip[2]]}>
              <cylinderGeometry
                args={[0.22, 0.045, d, 32, 1, true, Math.PI / 2, Math.PI]}
              />
              <meshStandardMaterial
                color="#ffbc28"
                emissive="#ff6a00"
                emissiveIntensity={1.2}
                side={DoubleSide}
              />
            </mesh>
          )}
          <mesh position={tip}>
            <sphereGeometry args={[0.06, 16, 16]} />
            <meshBasicMaterial color="#aefaff" />
          </mesh>
          <Segment
            from={[tip[0] + 0.7, 0.12, 0.42]}
            to={[tip[0] + 0.7, tip[1], 0.42]}
            color={CYAN}
            radius={0.008}
          />
          {[0.12, tip[1]].map((y, i) => (
            <Segment
              key={i}
              from={[tip[0] + 0.61, y, 0.42]}
              to={[tip[0] + 0.85, y, 0.42]}
              color={CYAN}
              radius={0.008}
            />
          ))}
          <Html
            position={[tip[0] + 1.1, 0.12 - d / 2, 0.43]}
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
  const compact = useThree((state) => state.size.width < 600);
  useEffect(() => {
    if (!controls.current) return;
    const positions = {
      overview: compact ? [8, 7, 14] : [12.5, 10, 18.5],
      scanner: compact ? [6, 6, 8] : [8.5, 8.5, 10.5],
      keyhole: [5.6, 1.6, 8.5],
    } as const;
    const targets = {
      overview: [-0.65, 0.9, 0],
      scanner: [1.3, 3.1, -0.9],
      keyhole: [5.5, -0.3, 0],
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
      minDistance={2}
      maxDistance={45}
      maxPolarAngle={Math.PI * 0.9}
    />
  );
}

function Assembly({
  time,
  labels,
  view,
}: {
  time: number;
  labels: boolean;
  view: CameraView;
}) {
  const phase = phaseAt(time);
  const compact = useThree((state) => state.size.width < 600);
  const signal = signalAt(time);
  const path = measurementPath(time, signal.active ? signal.sample.depth : 0);
  const [
    sensor,
    align1,
    align2,
    dichroic,
    xMirror,
    yMirror,
    lens,
    surface,
    tip,
  ] = path;
  const recent = signal.history.slice(-20);
  const searchDepth = recent.length
    ? recent.reduce((sum, sample) => sum + sample.depth, 0) / recent.length
    : 0;
  const carriage = -1.15 + 1.2 * Math.min(1, searchDepth / 1600);
  const output: Point[] = [COIL[COIL.length - 1], [-1.5, 3.45, 0], dichroic];
  const sourceLabels = labels && view === "overview";
  const scannerLabels =
    labels &&
    view !== "keyhole" &&
    phase >= 2 &&
    (!compact || view === "scanner" || phase === 2);
  return (
    <group>
      <gridHelper args={[30, 30, "#383d40", "#303538"]} position={[0, -3, 0]} />
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
        color="#7b858a"
      />
      <ClearShell position={[2.4, 3.4, 0]} size={[5.3, 2.55, 2.15]} />
      <mesh position={[-0.78, 3.45, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.2, 0.2, 1.05, 28]} />
        <meshStandardMaterial color="#5d6b75" metalness={0.8} roughness={0.3} />
      </mesh>
      <mesh position={[-0.25, 3.45, 0]} rotation={[0, Math.PI / 2, 0]}>
        <torusGeometry args={[0.24, 0.05, 10, 32]} />
        <meshStandardMaterial color="#3a454b" metalness={0.8} roughness={0.3} />
      </mesh>
      <mesh position={[0.1, 3.45, -1.1]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.17, 0.17, 0.14, 24, 1, true]} />
        <meshStandardMaterial
          color="#3a454b"
          metalness={0.8}
          roughness={0.3}
          side={DoubleSide}
        />
      </mesh>
      <Housing
        position={[-0.33, 2.72, -2.15]}
        size={[3.35, 0.08, 2.0]}
        color="#1e282e"
      />
      <ClearShell position={[-0.33, 3.38, -2.15]} size={[3.35, 1.24, 2.0]} />
      <Housing
        position={[-2.65, 3.45, -2.6]}
        size={[1.1, 0.75, 0.7]}
        color="#23414a"
      />
      <mesh position={[-1.86, 3.45, -2.6]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.12, 0.12, 0.48, 20]} />
        <meshStandardMaterial color="#5d6b75" metalness={0.8} roughness={0.3} />
      </mesh>
      <FocusStage carriage={carriage} />
      <Galvo
        previous={sensor}
        position={align1}
        next={align2}
        axis={[0, -1, 0]}
        scale={0.5}
      />
      <Galvo
        previous={align1}
        position={align2}
        next={dichroic}
        axis={[0, -1, 0]}
        scale={0.5}
      />
      <Mirror
        previous={align2}
        position={dichroic}
        next={xMirror}
        color="#83cfcb"
      />
      <Galvo
        previous={dichroic}
        position={xMirror}
        next={yMirror}
        axis={[0, 0, -1]}
      />
      <Galvo
        previous={xMirror}
        position={[3.65, 4.15, 0]}
        next={lens}
        axis={[-0.933, 0.359, 0]}
      />
      <mesh position={[4.35, 1.745, 0]}>
        <cylinderGeometry args={[0.6, 0.6, 0.59, 40, 1, true]} />
        <meshStandardMaterial
          color="#3d474d"
          metalness={0.75}
          roughness={0.35}
          side={DoubleSide}
        />
      </mesh>
      {[1.6, 1.8, 1.97].map((y) => (
        <mesh key={y} position={[4.35, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[0.6, 0.025, 6, 40]} />
          <meshStandardMaterial
            color="#6f7b81"
            metalness={0.8}
            roughness={0.3}
          />
        </mesh>
      ))}
      {[1.82, 1.48].map((y) => (
        <mesh key={y} position={[4.35, y, 0]}>
          <cylinderGeometry args={[0.56, 0.56, y > 1.6 ? 0.12 : 0.03, 40]} />
          <meshStandardMaterial
            color="#9fd3d8"
            transparent
            opacity={0.3}
            roughness={0.05}
            depthWrite={false}
          />
        </mesh>
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
            points={signal.active ? path : path.slice(0, 7)}
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
        active={sourceLabels && (!compact || phase === 0)}
      />
      <Label
        position={[-5.1, 1, 1.2]}
        title="FIBER COMBINER"
        subtitle="Multiple inputs → one fiber"
        active={sourceLabels && (!compact || phase === 0)}
      />
      <Label
        position={[-3.1, 4.1, 0]}
        title="Yb-DOPED FIBER"
        subtitle="Double-clad gain medium"
        active={sourceLabels && phase >= 1 && (!compact || phase === 1)}
      />
      <Label
        position={[-3.0, 4.15, -2.6]}
        title="LDD / OCT SENSOR"
        subtitle="Sample + reference interference"
        active={scannerLabels}
        accent
      />
      <Label
        position={[-0.7, 2.25, -3.2]}
        title="FOCUS STAGE"
        subtitle="Ball screw · sets search depth"
        active={scannerLabels && !compact}
        accent
      />
      <Label
        position={[1.9, 4.05, -2.4]}
        title="ALIGNMENT GALVOS"
        subtitle="Overlay 840 nm on 1,070 nm"
        active={scannerLabels && !compact}
        accent
      />
      <Label
        position={[0, 2, 1.6]}
        title="DICHROIC"
        subtitle="Collinear beam combination"
        active={scannerLabels}
      />
      <Label
        position={[2.15, 3.0, 1.3]}
        title="X GALVO"
        subtitle="First steering axis"
        active={scannerLabels && !compact}
      />
      <Label
        position={[3.3, 5.45, 0]}
        title="Y GALVO"
        subtitle="Second steering axis"
        active={scannerLabels && !compact}
      />
      <Label
        position={[5.85, 1.75, 0]}
        title="F-THETA LENS"
        subtitle="Focus onto the workpiece"
        active={scannerLabels}
      />
      <Label
        position={[4.3, -3.35, 1]}
        title="METAL WORKPIECE"
        subtitle="Cutaway · depth exaggerated"
        active={labels && phase >= 3 && !compact}
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
      <ambientLight intensity={1.6} />
      <directionalLight position={[0, 10, 8]} intensity={3} color="#fff2e0" />
      <directionalLight position={[-5, 4, -5]} intensity={2} color="#acccdf" />
      <directionalLight position={[4, 1, 8]} intensity={1.2} color="#e5eff4" />
      <Assembly time={time} labels={labels} view={view} />
      <CameraRig view={view} resetKey={resetKey} />
    </Canvas>
  );
}
