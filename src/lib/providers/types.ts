/** Provider-neutral server-side contract for hosted 3D generation. Provider payloads never leave src/lib/providers. */

export const HOSTED_PROVIDER_IDS = ["tripo"] as const;
export type HostedProviderId = (typeof HOSTED_PROVIDER_IDS)[number];

export const REQUEST_TIMEOUT_MS = 15_000;
export const DOWNLOAD_TIMEOUT_MS = 60_000;
export const MAX_GLB_BYTES = 100 * 1024 * 1024;

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "timed-out" | "rate-limited";
export type TaskError = { code: string; message: string; retryable: boolean };
export type NormalizedTask = { providerTaskId: string; status: JobStatus; progress?: number; glbUrl?: string; expiresAt?: string; error?: TaskError };
export type Fetch = typeof fetch;

export class ProviderError extends Error {
  constructor(readonly code: string, message: string, readonly httpStatus: number, readonly retryable: boolean, readonly retryAfterSeconds?: number) {
    super(message);
  }
}

export type ProviderConfig = { configured: boolean; enabled: boolean; hasKey: boolean; accessCodeRequired: boolean; verified: false };

export interface HostedProvider {
  readonly id: HostedProviderId;
  /** Vendor name shown in the UI and the paid-confirmation dialog. */
  readonly label: string;
  /** Approximate cost per generation. An estimate from public pricing pages; never a quote. */
  readonly costLabel: string;
  /** Whether the vendor API documents a way to cancel a task (best effort; running tasks may still refuse). */
  readonly supportsCancel: boolean;
  /** Hostnames (exact or `*.` suffix) the server may download GLBs from for this provider. */
  readonly assetHosts: readonly string[];
  readonly maxGlbBytes: number;
  config(env: Record<string, string | undefined>): ProviderConfig;
  create(prompt: string, refinement: string, fetchImpl?: Fetch): Promise<string>;
  status(taskId: string, fetchImpl?: Fetch): Promise<NormalizedTask>;
  cancel(taskId: string, fetchImpl?: Fetch): Promise<void>;
}
