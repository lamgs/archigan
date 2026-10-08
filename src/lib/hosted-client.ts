import type { Provider } from "./contracts";
import type { HostedTask } from "./hosted";

/** Browser-side wrapper for the `/api/generate` hosted routes. Never sees the provider key; sends the user's access code. */

export type HostedError = { code: string; message: string; retryable: boolean; retryAfterSeconds?: number };
export type Result<T> = { ok: true; value: T } | { ok: false; error: HostedError };

async function failure(response: Response): Promise<{ ok: false; error: HostedError }> {
  const body = (await response.json().catch(() => null)) as { error?: string; code?: string; retryable?: boolean } | null;
  const retryAfter = Number(response.headers.get("retry-after"));
  return { ok: false, error: { code: body?.code ?? `http-${response.status}`, message: body?.error ?? `The server responded with status ${response.status}.`, retryable: body?.retryable ?? response.status >= 500, ...(Number.isFinite(retryAfter) && retryAfter > 0 && { retryAfterSeconds: retryAfter }) } };
}

const networkError = (): { ok: false; error: HostedError } => ({ ok: false, error: { code: "network", message: "Could not reach the server.", retryable: true } });
const headers = (code: string) => ({ "x-sift-access-code": code });

export async function createHostedTask(input: { prompt: string; refinement: string; code: string; provider: Provider }): Promise<Result<{ taskId: string }>> {
  try {
    const response = await fetch("/api/generate", { method: "POST", headers: { "content-type": "application/json", ...headers(input.code) }, body: JSON.stringify({ prompt: input.prompt, refinement: input.refinement, provider: input.provider, confirmSpend: true }) });
    if (!response.ok) return await failure(response);
    const data = (await response.json()) as { taskId?: string };
    return data.taskId ? { ok: true, value: { taskId: data.taskId } } : { ok: false, error: { code: "bad-response", message: "The server did not return a task id.", retryable: false } };
  } catch {
    return networkError();
  }
}

export async function fetchHostedTask(provider: Provider, taskId: string, code: string): Promise<Result<HostedTask>> {
  try {
    const response = await fetch(`/api/generate/${encodeURIComponent(taskId)}?provider=${encodeURIComponent(provider)}`, { headers: headers(code), cache: "no-store" });
    if (!response.ok) return await failure(response);
    return { ok: true, value: ((await response.json()) as { task: HostedTask }).task };
  } catch {
    return networkError();
  }
}

export async function cancelHostedTask(provider: Provider, taskId: string, code: string): Promise<Result<true>> {
  try {
    const response = await fetch(`/api/generate/${encodeURIComponent(taskId)}?provider=${encodeURIComponent(provider)}`, { method: "DELETE", headers: headers(code) });
    return response.ok ? { ok: true, value: true } : await failure(response);
  } catch {
    return networkError();
  }
}

export async function downloadHostedModel(provider: Provider, taskId: string, code: string): Promise<Result<Blob>> {
  try {
    const response = await fetch(`/api/generate/${encodeURIComponent(taskId)}/model?provider=${encodeURIComponent(provider)}`, { headers: headers(code), cache: "no-store" });
    return response.ok ? { ok: true, value: await response.blob() } : await failure(response);
  } catch {
    return networkError();
  }
}
