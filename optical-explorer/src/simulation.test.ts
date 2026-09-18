import assert from "node:assert/strict";
import { test } from "node:test";
import {
  clampTime,
  DEPTH_SAMPLES,
  DEPTH_SCALE,
  DURATION,
  INTERACTION_START,
  measurementPath,
  phaseAt,
  returnPath,
  signalAt,
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
