import { NextResponse } from "next/server";
import { authorize, clientIp, limitsFromEnv, SpendLimiter, TASK_ID_PATTERN, type GuardResult } from "./guard";
import { MeshyError } from "./meshy";

/** Module-level so counters persist across requests within one server instance. */
let limiter: SpendLimiter | undefined;
export const spendLimiter = () => (limiter ??= new SpendLimiter(limitsFromEnv(process.env)));
export const resetSpendLimiterForTests = () => { limiter = undefined; };

export function guardResponse(result: Extract<GuardResult, { ok: false }>) {
  return NextResponse.json({ error: result.message, code: result.code }, { status: result.status, headers: result.retryAfterSeconds ? { "Retry-After": String(result.retryAfterSeconds) } : undefined });
}

export function providerErrorResponse(error: unknown) {
  if (error instanceof MeshyError) {
    return NextResponse.json({ error: error.message, code: error.code, retryable: error.retryable }, { status: error.httpStatus >= 400 && error.httpStatus < 600 ? error.httpStatus : 502, headers: error.retryAfterSeconds ? { "Retry-After": String(error.retryAfterSeconds) } : undefined });
  }
  return NextResponse.json({ error: "Hosted generation failed.", code: "unexpected", retryable: false }, { status: 502 });
}

/** Common checks for the per-task routes (status, cancel, model download). Returns a Response to send, or the validated task id. */
export async function taskRequest(request: Request, params: Promise<{ taskId: string }>): Promise<{ response: Response } | { taskId: string }> {
  const auth = authorize(request.headers, process.env);
  if (!auth.ok) return { response: guardResponse(auth) };
  const { taskId } = await params;
  if (!TASK_ID_PATTERN.test(taskId)) return { response: NextResponse.json({ error: "Invalid task id.", code: "bad-task-id" }, { status: 400 }) };
  return { taskId };
}

export { authorize, clientIp };
