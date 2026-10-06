import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ACTUAL,
  BEAD_HALF_WIDTH,
  BEAD_PROFILE,
  EDGES,
  FOUND,
  NOMINAL,
  OFFSETS,
  PART,
  RESULT,
  SCAN,
  SCAN_LINES,
  SCAN_POINTS,
  TARGET_DURATION,
  TARGET_STEPS,
  TOLERANCE_UM,
  targetingAt,
  targetStepAt,
  toLocal,
  toWorld,
} from "./targeting.ts";

test("steps tile the timeline and the end stays on the last step", () => {
  TARGET_STEPS.forEach((step, i) => {
    assert.equal(targetStepAt(step.start), i);
    if (i) assert.equal(TARGET_STEPS[i - 1].end, step.start);
  });
  assert.equal(TARGET_STEPS.at(-1)!.end, TARGET_DURATION);
  assert.equal(targetStepAt(TARGET_DURATION), TARGET_STEPS.length - 1);
  assert.equal(targetStepAt(-5), 0);
  assert.equal(targetStepAt(Number.NaN), 0);
});

test("the raster is serpentine and the tab shows up as a height step", () => {
  assert.equal(SCAN.length, SCAN_LINES * SCAN_POINTS);
  assert(SCAN[SCAN_POINTS - 1].x > SCAN[SCAN_POINTS].x - 1e-9);
  assert(SCAN[SCAN_POINTS].x > SCAN[SCAN_POINTS + 1].x);
  for (const p of SCAN) {
    const [u, v] = toLocal(ACTUAL, [p.x, p.y]);
    const inside =
      Math.abs(u) < PART.length / 2 - 0.05 &&
      Math.abs(v) < PART.width / 2 - 0.05;
    const outside =
      Math.abs(u) > PART.length / 2 + 0.05 ||
      Math.abs(v) > PART.width / 2 + 0.05;
    if (inside) assert(Math.abs(p.z - ACTUAL.height) <= 6);
    if (outside) assert(Math.abs(p.z) <= 6);
  }
});

test("found edges lie on the real tab outline", () => {
  assert(EDGES.length >= 20);
  for (const edge of EDGES) {
    const [u, v] = toLocal(ACTUAL, edge);
    const toEdge = Math.min(
      Math.abs(Math.abs(u) - PART.length / 2),
      Math.abs(Math.abs(v) - PART.width / 2),
    );
    assert(toEdge < 0.07, `edge ${edge} is ${toEdge} mm from the outline`);
  }
});

test("the pose is recovered from the scan, not copied from the truth", () => {
  assert.notDeepEqual(FOUND, ACTUAL);
  assert(Math.abs(FOUND.x - ACTUAL.x) < 0.03);
  assert(Math.abs(FOUND.y - ACTUAL.y) < 0.03);
  assert(Math.abs(FOUND.angle - ACTUAL.angle) < (1 * Math.PI) / 180);
  assert(Math.abs(FOUND.height - ACTUAL.height) <= 3);
  const [x, y] = toWorld(FOUND, toLocal(FOUND, [1.23, -0.45]));
  assert(Math.abs(x - 1.23) < 1e-12 && Math.abs(y + 0.45) < 1e-12);
});

test("targeting brings the seam inside tolerance and the bead clear of the edge", () => {
  assert(RESULT.nominalMissUm > 3 * TOLERANCE_UM);
  assert(RESULT.correctedMissUm < TOLERANCE_UM / 4);
  assert(OFFSETS.every((s) => s.corrected <= TOLERANCE_UM));
  assert(RESULT.correctedEdgeUm > 3 * RESULT.nominalEdgeUm);
  assert.equal(RESULT.focusShiftUm, FOUND.height - NOMINAL.height);
  const crest = BEAD_PROFILE.reduce((a, b) =>
    b.corrected > a.corrected ? b : a,
  );
  assert(Math.abs(crest.across) < BEAD_HALF_WIDTH);
});

test("scrubbing is deterministic and the scan only reveals what has been measured", () => {
  assert.equal(targetingAt(0).scanned, 0);
  assert.equal(targetingAt(10).scanned, SCAN.length);
  const before = targetingAt(4.37);
  targetingAt(TARGET_DURATION);
  assert.deepEqual(targetingAt(4.37), before);
  assert(before.profile.every((p) => p.line === before.line));
  assert.equal(targetingAt(14).correction, 0);
  assert.equal(targetingAt(19).correction, 1);
  assert.deepEqual(targetingAt(19).path, FOUND);
  assert.equal(targetingAt(19.5).weld, 0);
  assert.equal(targetingAt(26).weld, 1);
  assert.equal(targetingAt(22.5).offsets.length, 31);
  assert.equal(targetingAt(30).bead.length, 61);
});
