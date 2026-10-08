import type { Artifact, GenerationJob } from "./contracts";

/** Pure state logic for provider-neutral hosted generation jobs (no network, no React). */

export type JobStatus = GenerationJob["status"];
export type HostedTask = { providerTaskId: string; status: JobStatus; progress?: number; hasModel?: boolean; expiresAt?: string; error?: { code: string; message: string; retryable: boolean } };

const TERMINAL: JobStatus[] = ["completed", "failed", "cancelled", "timed-out"];
export const JOB_TIMEOUT_MS = 20 * 60_000;

export const isActiveJob = (job: Pick<GenerationJob, "status">) => !TERMINAL.includes(job.status);
export const isHostedJob = (job: Pick<GenerationJob, "provider" | "providerTaskId">) => job.provider !== "procedural" && Boolean(job.providerTaskId);

export function newHostedJob(input: { id: string; nodeId: string; taskId: string; now: string }): GenerationJob {
  return { id: input.id, nodeId: input.nodeId, provider: "meshy", providerTaskId: input.taskId, status: "queued", progress: 0, createdAt: input.now, updatedAt: input.now };
}

/** Folds a provider status report into a job. Finished jobs are never reopened, and progress never moves backwards. */
export function applyTaskUpdate(job: GenerationJob, task: HostedTask, now: string): GenerationJob {
  if (!isActiveJob(job)) return job;
  const progress = task.progress === undefined ? job.progress : Math.max(job.progress ?? 0, task.progress);
  const next: GenerationJob = { ...job, status: task.status, ...(progress !== undefined && { progress }), updatedAt: now, ...(task.expiresAt && { outputExpiresAt: task.expiresAt }) };
  if (task.status === "failed" || task.status === "cancelled") next.error = task.error ?? (task.status === "failed" ? { code: "provider-failed", message: "The provider could not generate this model.", retryable: true } : undefined);
  if (!next.error) delete next.error;
  return next;
}

export function failJob(job: GenerationJob, error: { code: string; message: string; retryable: boolean }, now: string, status: JobStatus = "failed"): GenerationJob {
  return isActiveJob(job) ? { ...job, status, error, updatedAt: now } : job;
}

export function markRateLimited(job: GenerationJob, now: string): GenerationJob {
  return isActiveJob(job) ? { ...job, status: "rate-limited", updatedAt: now } : job;
}

/** Jobs that have been active for longer than `limitMs` become `timed-out` (the provider may still finish, but we stop waiting). */
export function timeoutIfStale(job: GenerationJob, nowMs: number, limitMs = JOB_TIMEOUT_MS): GenerationJob {
  if (!isActiveJob(job) || !job.createdAt || nowMs - Date.parse(job.createdAt) < limitMs) return job;
  return { ...job, status: "timed-out", updatedAt: new Date(nowMs).toISOString(), error: { code: "timeout", message: "Meshy took too long, so this app stopped waiting. The task may still finish on the Meshy side.", retryable: true } };
}

export function userCancel(job: GenerationJob, now: string): GenerationJob {
  return isActiveJob(job) ? { ...job, status: "cancelled", updatedAt: now } : job;
}

/** Poll interval: gentle backoff from 3 s to 15 s; a provider Retry-After always wins. */
export function nextPollDelayMs(attempt: number, retryAfterSeconds?: number) {
  if (retryAfterSeconds) return Math.min(120, Math.max(1, retryAfterSeconds)) * 1000;
  return Math.min(15_000, 3000 + attempt * 1500);
}

export function buildHostedArtifact(input: { artifactId: string; job: GenerationJob; bytes: number; now: string }): Artifact {
  return {
    id: input.artifactId,
    kind: "model-glb",
    sourceNodeId: input.job.nodeId,
    createdAt: input.now,
    storageKey: `asset:${input.artifactId}`,
    metadata: { origin: "meshy", providerTaskId: input.job.providerTaskId ?? null, bytes: input.bytes, verified: false, ...(input.job.outputExpiresAt && { providerExpiresAt: input.job.outputExpiresAt }) },
  };
}

export function completeJob(job: GenerationJob, artifactId: string, now: string): GenerationJob {
  const { error: _error, ...rest } = job;
  void _error;
  return { ...rest, status: "completed", progress: 100, resultArtifactId: artifactId, updatedAt: now };
}

export function describeJob(job: GenerationJob): string {
  switch (job.status) {
    case "queued": return "Queued at Meshy";
    case "running": return job.progress && job.progress >= 100 ? "Downloading the model" : `Generating${job.progress ? ` · ${job.progress}%` : ""}`;
    case "rate-limited": return "Meshy is rate limiting; retrying shortly";
    case "completed": return "Hosted model saved in this browser";
    case "cancelled": return "Cancelled";
    case "timed-out": return job.error?.message ?? "Timed out";
    default: return job.error?.message ?? "Failed";
  }
}
