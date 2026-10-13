import assert from "node:assert/strict";
import { test } from "node:test";
import {
  clampTime,
  DEPTH_SAMPLES,
  DEPTH_SCALE,
  DURATION,
  fresnelAbsorptivity,
  INTERACTION_START,
  KEYHOLE_ONSET_MS,
  keyholeAbsorptance,
  keyholeRayPath,
  measurementPath,
  modeAt,
  MOUTH_RADIUS_UM,
  phaseAt,
  returnPath,
  signalAt,
  STEEL_K,
  STEEL_N,
} from "./simulation.ts";

test("phase boundaries are exact and the end stays in the interaction phase", () => {
  const cases = [
    [-1, 0],
    [0, 0],
    [5.999, 0],
    [6, 1],
    [11.999, 1],
    [12, 2],
    [19.999, 2],
    [20, 3],
    [32, 3],
    [40, 3],
  ];
  for (const [time, phase] of cases) assert.equal(phaseAt(time), phase);
  assert.equal(clampTime(Number.NaN), 0);
  assert.equal(clampTime(Number.POSITIVE_INFINITY), 0);
});

test("the graph is inactive until interaction and rewinding discards future points", () => {
  assert.deepEqual(signalAt(INTERACTION_START - 0.001).history, []);
  assert.equal(signalAt(INTERACTION_START).sample.depth, 0);
  assert.equal(signalAt(DURATION).history.length, 241);
  const before = signalAt(23.42);
  signalAt(DURATION);
  assert.deepEqual(signalAt(23.42), before);
  assert.equal(before.history.at(-1), before.sample);
  assert(before.history.every((sample) => sample.time <= 34.2));
  assert.deepEqual(signalAt(0).history, []);
});

test("every visible chart endpoint directly controls the keyhole floor", () => {
  for (let index = 0; index < DEPTH_SAMPLES.length; index += 1) {
    const time = INTERACTION_START + index * 0.05 + 0.000001;
    const { sample, history } = signalAt(time);
    const path = measurementPath(time, sample.depth);
    const surface = path.at(-2)!;
    const floor = path.at(-1)!;
    assert.equal(sample, history.at(-1));
    assert(
      Math.abs((surface[1] - floor[1]) / DEPTH_SCALE - sample.depth) < 1e-9,
    );
    assert.equal(surface[0], floor[0]);
    assert.equal(surface[2], floor[2]);
  }
});

test("exact sample boundaries do not slip backwards due to floating point rounding", () => {
  DEPTH_SAMPLES.forEach((sample, index) => {
    assert.equal(signalAt(INTERACTION_START + index * 0.05).sample, sample);
  });
});

test("the reflected measurement retraces all optical vertices without mutating the outbound path", () => {
  for (const time of [20, 24, 28, 32]) {
    const path = measurementPath(time, signalAt(time).sample.depth);
    const original = structuredClone(path);
    const reflected = returnPath(path);
    assert.deepEqual(path, original);
    assert.equal(reflected.length, path.length);
    reflected.forEach((point, index) =>
      assert.deepEqual(point, path[path.length - index - 1]),
    );
  }
});

test("the synthetic steady signal fluctuates around its nominal depth with a true trailing average", () => {
  const steady = DEPTH_SAMPLES.filter((sample) => sample.time >= 30);
  const mean =
    steady.reduce((sum, sample) => sum + sample.depth, 0) / steady.length;
  assert(Math.abs(mean - 1200) < 25);
  assert(
    Math.max(...steady.map((sample) => sample.depth)) -
      Math.min(...steady.map((sample) => sample.depth)) >
      150,
  );
  DEPTH_SAMPLES.forEach((sample, index) => {
    const window = DEPTH_SAMPLES.slice(Math.max(0, index - 19), index + 1);
    assert.equal(
      sample.mean,
      Math.round(
        window.reduce((sum, point) => sum + point.depth, 0) / window.length,
      ),
    );
    assert(sample.depth >= 0 && sample.depth <= 1600);
  });
});

test("Fresnel absorptivity matches normal incidence and tends to zero at grazing incidence", () => {
  const expected =
    1 -
    ((STEEL_N - 1) ** 2 + STEEL_K ** 2) /
      ((STEEL_N + 1) ** 2 + STEEL_K ** 2);
  assert.ok(Math.abs(fresnelAbsorptivity(1) - expected) < 1e-12);
  assert.equal(fresnelAbsorptivity(0), 0);
  assert(fresnelAbsorptivity(0.2) > fresnelAbsorptivity(1));
});

test("keyhole absorptance agrees with WeldSYM's Python ray trace", () => {
  // References from .venv/bin/python calling weldsim.optics.keyhole_absorptance(3.5, 4.0, depth_um * 1e-6, 150e-6).
  const references = [
    [0, 0.386206896552],
    [100, 0.420276405494],
    [400, 0.807917083735],
    [1200, 0.915412091977],
  ] as const;
  for (const [depthUm, expected] of references) {
    assert.ok(
      Math.abs(keyholeAbsorptance(depthUm) - expected) < 0.02,
      `depth ${depthUm} µm: ${keyholeAbsorptance(depthUm)} vs ${expected}`,
    );
  }
});

test("keyhole absorptance is monotonic and bounded over the table range", () => {
  const absorptances = Array.from({ length: 161 }, (_, index) =>
    keyholeAbsorptance(index * 10),
  );
  absorptances.forEach((absorptance, index) => {
    assert(absorptance >= 0 && absorptance <= 1);
    if (index > 0) assert(absorptance >= absorptances[index - 1] - 1e-12);
  });
});

test("depth samples switch from conduction to a more absorptive keyhole", () => {
  const conductionSamples = DEPTH_SAMPLES.filter(
    (sample) => sample.time < KEYHOLE_ONSET_MS,
  );
  assert(conductionSamples.length > 0);
  for (const sample of conductionSamples) {
    assert.equal(sample.mode, "conduction");
    assert(sample.depth < 100);
  }
  assert.equal(modeAt(KEYHOLE_ONSET_MS - 0.001), "conduction");
  assert.equal(modeAt(KEYHOLE_ONSET_MS), "keyhole");

  const flat = fresnelAbsorptivity(1);
  const formedSamples = DEPTH_SAMPLES.filter(
    (sample) => sample.time >= KEYHOLE_ONSET_MS + 20,
  );
  assert(formedSamples.length > 0);
  for (const sample of formedSamples) {
    assert.equal(sample.mode, "keyhole");
    assert(sample.absorptance > flat);
    assert.equal(sample.absorptance, Math.round(sample.absorptance * 1000) / 1000);
  }
});

test("keyhole ray paths preserve power and follow the conical wall", () => {
  const path = keyholeRayPath(0.5, 1200);
  assert.deepEqual(path.points[0], [75, 0]);
  assert.equal(path.power[0], 1);
  assert.equal(path.points.length, path.power.length);
  for (let index = 1; index < path.power.length; index += 1) {
    assert(path.power[index] <= path.power[index - 1]);
  }
  const lastIndex = path.points.length - 1;
  assert(path.points[lastIndex][1] <= 0 || path.power[lastIndex] < 1e-6);

  for (const [x, z] of path.points.slice(1, -1)) {
    const wallRadius = MOUTH_RADIUS_UM * (1 - z / 1200);
    assert.ok(Math.abs(Math.abs(x) - wallRadius) <= 1e-6);
  }
});
