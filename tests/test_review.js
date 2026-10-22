import test from "node:test";
import assert from "node:assert/strict";
import { MAX_SOURCE_BYTES, reviewSource } from "../public/review.js";

test("detects eval and exec with source line numbers", () => {
  const findings = reviewSource("safe = 1\nvalue = eval(user_input)\nexec(payload)");
  assert.equal(findings.length, 2);
  assert.deepEqual(findings.map(item => item.line), [2, 3]);
  assert.equal(findings[0].severity, "high");
});

test("detects bare except", () => {
  const findings = reviewSource("try:\n    work()\nexcept:\n    pass");
  assert.equal(findings.length, 1);
  assert.equal(findings[0].severity, "medium");
  assert.equal(findings[0].line, 3);
});

test("detects shell=True in a subprocess call", () => {
  const findings = reviewSource("subprocess.run(command, shell=True)");
  assert.equal(findings.length, 1);
  assert.equal(findings[0].line, 1);
});

test("does not flag ordinary code or similar text in comments", () => {
  assert.deepEqual(reviewSource("# eval(input)\nprint('safe')"), []);
});

test("documents the local source size limit", () => {
  assert.equal(MAX_SOURCE_BYTES, 25_000);
});
