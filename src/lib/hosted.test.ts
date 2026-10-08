import { describe, expect, it } from "vitest";
import { generationJobSchema, siftProjectV2Schema } from "./contracts";
import { applyTaskUpdate, buildHostedArtifact, completeJob, describeJob, failJob, isActiveJob, markRateLimited, newHostedJob, nextPollDelayMs, timeoutIfStale, userCancel } from "./hosted";
import { createWorkflowProject, projectSignature } from "./projects";
import { evaluateGraph } from "./workflow";

const T0 = "2026-10-09T10:00:00.000Z";
const T1 = "2026-10-09T10:01:00.000Z";
const job = () => newHostedJob({ id: "j", nodeId: "n", taskId: "task-123456", now: T0 });

describe("hosted job state machine", () => {
  it("starts queued, active, and schema-valid", () => {
    expect(generationJobSchema.safeParse(job()).success).toBe(true);
    expect(job()).toMatchObject({ provider: "meshy", status: "queued", providerTaskId: "task-123456" });
    expect(isActiveJob(job())).toBe(true);
  });
  it("walks queued → running → completed, with monotonic progress", () => {
    let j = applyTaskUpdate(job(), { providerTaskId: "t", status: "running", progress: 40 }, T1);
    expect(j).toMatchObject({ status: "running", progress: 40 });
    j = applyTaskUpdate(j, { providerTaskId: "t", status: "running", progress: 25 }, T1);
    expect(j.progress).toBe(40);
    j = completeJob(j, "art", T1);
    expect(j).toMatchObject({ status: "completed", progress: 100, resultArtifactId: "art" });
    expect(isActiveJob(j)).toBe(false);
  });
  it("records provider failures with their error and never reopens finished jobs", () => {
    const failed = applyTaskUpdate(job(), { providerTaskId: "t", status: "failed", error: { code: "provider-failed", message: "Prompt rejected", retryable: true } }, T1);
    expect(failed).toMatchObject({ status: "failed", error: { message: "Prompt rejected" } });
    expect(applyTaskUpdate(failed, { providerTaskId: "t", status: "running", progress: 90 }, T1)).toBe(failed);
    const done = completeJob(job(), "a", T1);
    expect(applyTaskUpdate(done, { providerTaskId: "t", status: "failed" }, T1)).toBe(done);
    expect(failJob(done, { code: "x", message: "y", retryable: false }, T1)).toBe(done);
  });
  it("maps cancelled and rate-limited states", () => {
    expect(applyTaskUpdate(job(), { providerTaskId: "t", status: "cancelled" }, T1)).toMatchObject({ status: "cancelled" });
    const limited = markRateLimited(job(), T1);
    expect(limited.status).toBe("rate-limited");
    expect(isActiveJob(limited)).toBe(true); // keeps polling
    expect(applyTaskUpdate(limited, { providerTaskId: "t", status: "running" }, T1).status).toBe("running");
    expect(userCancel(job(), T1).status).toBe("cancelled");
  });
  it("times out stale jobs only, preserving the explanation", () => {
    const start = Date.parse(T0);
    expect(timeoutIfStale(job(), start + 19 * 60_000)).toEqual(job());
    const timed = timeoutIfStale(job(), start + 21 * 60_000);
    expect(timed).toMatchObject({ status: "timed-out", error: { code: "timeout", retryable: true } });
    expect(timeoutIfStale(completeJob(job(), "a", T1), start + 99 * 60_000).status).toBe("completed");
    expect(timeoutIfStale({ ...job(), createdAt: undefined }, start + 99 * 60_000).status).toBe("queued"); // legacy jobs without timestamps are never auto-timed-out
  });
  it("backs off polling and honours Retry-After", () => {
    expect([0, 1, 2, 5, 50].map((n) => nextPollDelayMs(n))).toEqual([3000, 4500, 6000, 10500, 15000]);
    expect(nextPollDelayMs(0, 30)).toBe(30_000);
    expect(nextPollDelayMs(0, 9999)).toBe(120_000);
  });
  it("describes every status for the UI", () => {
    (["queued", "running", "completed", "failed", "cancelled", "timed-out", "rate-limited"] as const).forEach((status) => expect(describeJob({ ...job(), status }).length).toBeGreaterThan(3));
  });
});

describe("hosted artifacts and reload safety", () => {
  it("builds an unverified model-glb artifact pointing at a local asset", () => {
    const a = buildHostedArtifact({ artifactId: "glb-1", job: { ...job(), outputExpiresAt: "2026-10-12T10:00:00.000Z" }, bytes: 2048, now: T1 });
    expect(a).toMatchObject({ kind: "model-glb", sourceNodeId: "n", storageKey: "asset:glb-1", metadata: { origin: "meshy", providerTaskId: "task-123456", bytes: 2048, verified: false, providerExpiresAt: "2026-10-12T10:00:00.000Z" } });
  });
  it("persists in-flight jobs and hosted results in the project record and round-trips through JSON", () => {
    const base = createWorkflowProject({ id: "p", name: "Hosted", now: T0, prompt: "A tower", refinement: "" });
    const art = buildHostedArtifact({ artifactId: "glb-1", job: job(), bytes: 10, now: T1 });
    const done = completeJob(job(), "glb-1", T1);
    const project = { ...base, artifacts: { ...base.artifacts, "glb-1": art }, jobs: { ...base.jobs, j: { ...done, nodeId: "p-generation" } }, graph: { ...base.graph, nodes: base.graph.nodes.map((n) => (n.id === "p-generation" ? { ...n, params: { hostedArtifactId: "glb-1" } } : n)) } };
    const reloaded = siftProjectV2Schema.parse(JSON.parse(JSON.stringify(project)));
    expect(reloaded.jobs.j.status).toBe("completed");
    const pending = siftProjectV2Schema.parse(JSON.parse(JSON.stringify({ ...base, jobs: { j: { ...job(), nodeId: "p-generation" } } })));
    expect(isActiveJob(pending.jobs.j)).toBe(true); // an unfinished task is still active after reload, so polling can resume
  });
  it("changes the autosave signature when a job changes state", () => {
    const base = createWorkflowProject({ id: "p", name: "Hosted", now: T0, prompt: "A tower", refinement: "" });
    const withJob = (j: ReturnType<typeof job>) => projectSignature({ ...base, jobs: { ...base.jobs, j: { ...j, nodeId: "p-generation" } } });
    expect(withJob(job())).not.toBe(withJob({ ...job(), status: "running" }));
    expect(withJob(job())).toBe(withJob({ ...job(), progress: 50 })); // progress ticks do not trigger a save each poll
  });
  it("explains why a hosted-only generation node has no editable geometry", () => {
    const base = createWorkflowProject({ id: "p", name: "H", now: T0, prompt: "A tower", refinement: "" });
    const nodes = base.graph.nodes.map((n) => (n.id === "p-generation" ? { ...n, artifactId: undefined, params: { hostedArtifactId: "glb-1" } } : n));
    const r = evaluateGraph({ nodes, edges: base.graph.edges }, {})["p-generation"];
    expect(r).toMatchObject({ status: "blocked", message: expect.stringMatching(/no editable geometry/) });
  });
});
