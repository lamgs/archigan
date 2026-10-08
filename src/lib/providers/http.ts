import { DOWNLOAD_TIMEOUT_MS, ProviderError, REQUEST_TIMEOUT_MS, type Fetch, type HostedProvider } from "./types";

/** Shared HTTP error mapping. `vendor` is only used in user-facing messages. */
export function mapHttpError(status: number, retryAfter?: string | null, vendor = "Meshy"): ProviderError {
  const seconds = retryAfter && Number.isFinite(Number(retryAfter)) ? Math.max(1, Math.round(Number(retryAfter))) : undefined;
  if (status === 401 || status === 403) return new ProviderError("auth", `${vendor} rejected the server's API key.`, status, false);
  if (status === 402) return new ProviderError("insufficient-credits", `The ${vendor} account has no credits left for this request.`, status, false);
  if (status === 404) return new ProviderError("not-found", `${vendor} no longer has this task (results are only retained for a limited time).`, status, false);
  if (status === 409) return new ProviderError("running", `${vendor} cannot cancel a task that is already running.`, status, false);
  if (status === 429) return new ProviderError("rate-limited", `${vendor} is rate limiting requests. Try again shortly.`, status, true, seconds ?? 10);
  if (status === 400 || status === 422) return new ProviderError("rejected", `${vendor} rejected this request.`, status, false);
  if (status >= 500) return new ProviderError("provider-unavailable", `${vendor} is temporarily unavailable.`, status, true, seconds);
  return new ProviderError("unexpected", `${vendor} responded with status ${status}.`, status, false);
}

/** fetch with timeout and network/timeout error mapping. Does not check `response.ok`. */
export async function timedFetch(vendor: string, url: string, init: RequestInit, fetchImpl: Fetch): Promise<Response> {
  try {
    return await fetchImpl(url, { ...init, cache: "no-store", signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
  } catch (error) {
    const timeout = error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError");
    throw new ProviderError(timeout ? "timeout" : "network", timeout ? `${vendor} did not respond in time.` : `Could not reach ${vendor}.`, 504, true);
  }
}

/** Only HTTPS URLs on the provider's own hosts may be fetched on the user's behalf (prevents the server being an open proxy). */
export function isAllowedHost(value: string, hosts: readonly string[]): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && !url.username && !url.password && hosts.some((host) => url.hostname === host || url.hostname.endsWith(`.${host}`));
  } catch {
    return false;
  }
}

/** Downloads a GLB from a (signed) provider URL with host allowlist, size cap and a magic-number check. */
export async function downloadGlbFor(provider: Pick<HostedProvider, "label" | "assetHosts" | "maxGlbBytes">, url: string, fetchImpl: Fetch = fetch): Promise<ArrayBuffer> {
  const vendor = provider.label;
  if (!isAllowedHost(url, provider.assetHosts)) throw new ProviderError("bad-asset-url", `${vendor} returned a model URL this app will not fetch.`, 502, false);
  let response: Response;
  try {
    // Redirects are refused: a signed URL on an allowlisted host must not bounce the server to another host.
    response = await fetchImpl(url, { cache: "no-store", redirect: "error", signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS) });
  } catch {
    throw new ProviderError("network", `Could not download the model from ${vendor}.`, 504, true);
  }
  if (response.status === 403 || response.status === 410) throw new ProviderError("asset-expired", "The signed download link has expired. Re-check the task for a fresh link.", response.status, true);
  if (!response.ok) throw mapHttpError(response.status, null, vendor);
  const declared = Number(response.headers.get("content-length") ?? 0);
  if (declared > provider.maxGlbBytes) throw new ProviderError("too-large", "The model file is too large to import.", 502, false);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > provider.maxGlbBytes) throw new ProviderError("too-large", "The model file is too large to import.", 502, false);
  if (bytes.byteLength < 12 || new TextDecoder().decode(new Uint8Array(bytes, 0, 4)) !== "glTF") throw new ProviderError("not-glb", "The downloaded file is not a valid GLB.", 502, false);
  return bytes;
}

export const clampProgress = (value: number | undefined) => (value === undefined || !Number.isFinite(value) ? undefined : Math.max(0, Math.min(100, Math.round(value))));

/** Shared fail-closed config: enabled flag + own key + a shared access code (SIFT_ACCESS_CODE, falling back to MESHY_ACCESS_CODE). */
export function hostedConfig(env: Record<string, string | undefined>, enabledVar: string, keyVars: string[]) {
  const enabled = env[enabledVar] === "true";
  const hasKey = keyVars.every((name) => Boolean(env[name]));
  const accessCodeRequired = Boolean(sharedAccessCode(env));
  return { configured: enabled && hasKey && accessCodeRequired, enabled, hasKey, accessCodeRequired, verified: false as const };
}

export const sharedAccessCode = (env: Record<string, string | undefined>) => env.SIFT_ACCESS_CODE || env.MESHY_ACCESS_CODE || undefined;
