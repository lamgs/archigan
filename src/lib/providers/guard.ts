import { timingSafeEqual } from "node:crypto";

export type SpendLimits = { perIpWindowMs: number; perIpMax: number; dailyMax: number };
export type GuardResult = { ok: true } | { ok: false; status: number; code: string; message: string; retryAfterSeconds?: number };

export const DEFAULT_LIMITS: SpendLimits = { perIpWindowMs: 10 * 60_000, perIpMax: 3, dailyMax: 20 };
const DAY_MS = 24 * 60 * 60_000;

/**
 * Best-effort in-memory limiter for paid requests. On serverless hosts each warm instance keeps its own counters, so
 * this reduces accidents and casual abuse; the hard control is the access code plus the provider-side account budget.
 */
export class SpendLimiter {
  private perIp = new Map<string, number[]>();
  private day: number[] = [];
  constructor(private limits: SpendLimits = DEFAULT_LIMITS) {}

  check(ip: string, now: number): GuardResult {
    this.day = this.day.filter((t) => now - t < DAY_MS);
    const recent = (this.perIp.get(ip) ?? []).filter((t) => now - t < this.limits.perIpWindowMs);
    this.perIp.set(ip, recent);
    if (this.day.length >= this.limits.dailyMax) return { ok: false, status: 429, code: "daily-limit", message: "The daily hosted-generation limit for this deployment has been reached.", retryAfterSeconds: Math.ceil((DAY_MS - (now - this.day[0])) / 1000) };
    if (recent.length >= this.limits.perIpMax) return { ok: false, status: 429, code: "rate-limited", message: "Too many hosted generations in a short time. Please wait before trying again.", retryAfterSeconds: Math.ceil((this.limits.perIpWindowMs - (now - recent[0])) / 1000) };
    return { ok: true };
  }

  /** Call only after Meshy accepted the request, so failed attempts do not consume the budget. */
  record(ip: string, now: number) {
    this.day.push(now);
    this.perIp.set(ip, [...(this.perIp.get(ip) ?? []), now]);
  }
}

export function limitsFromEnv(env: Record<string, string | undefined>): SpendLimits {
  const daily = Number(env.MESHY_DAILY_LIMIT);
  return { ...DEFAULT_LIMITS, dailyMax: Number.isInteger(daily) && daily > 0 ? daily : DEFAULT_LIMITS.dailyMax };
}

export function codeMatches(provided: string | null | undefined, expected: string | undefined): boolean {
  if (!provided || !expected) return false;
  const a = Buffer.from(provided);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

/** Checks the shared access code (required for every hosted-provider call, including status polling). */
export function authorize(headers: Headers, env: Record<string, string | undefined>): GuardResult {
  if (env.MESHY_ENABLED !== "true" || !env.MESHY_API_KEY || !env.MESHY_ACCESS_CODE) return { ok: false, status: 503, code: "not-configured", message: "Hosted generation is not configured on this deployment." };
  if (!codeMatches(headers.get("x-sift-access-code"), env.MESHY_ACCESS_CODE)) return { ok: false, status: 401, code: "access-denied", message: "A valid access code is required for hosted generation." };
  return { ok: true };
}

export function clientIp(headers: Headers): string {
  return headers.get("x-forwarded-for")?.split(",")[0]?.trim() || headers.get("x-real-ip") || "unknown";
}

export const TASK_ID_PATTERN = /^[A-Za-z0-9_-]{6,80}$/;
