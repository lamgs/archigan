#!/usr/bin/env node
// Offline preflight for hosted generation. Makes NO vendor calls and prints NO secret values.
// Usage:  node --env-file=.env.local scripts/check-hosted-config.mjs [--url https://your-deployment]
// --url additionally does one read-only GET of your own app's /api/providers (never a vendor API).

const PROVIDERS = [
  { id: "hunyuan3d-rapid/pro", label: "Hunyuan3D via fal.ai", flag: "HUNYUAN_ENABLED", keys: ["FAL_KEY"] },
  { id: "tripo", label: "Tripo", flag: "TRIPO_ENABLED", keys: ["TRIPO_API_KEY"] },
  { id: "meshy", label: "Meshy", flag: "MESHY_ENABLED", keys: ["MESHY_API_KEY"] },
];

/** Pure: evaluates an env object. Returns per-provider status plus global warnings. */
export function evaluate(env) {
  const code = env.SIFT_ACCESS_CODE || env.MESHY_ACCESS_CODE || "";
  const warnings = [];
  if (!code) warnings.push("No access code: set SIFT_ACCESS_CODE. Without it NO hosted provider can be enabled (fail closed).");
  else if (code.length < 12) warnings.push("The access code is shorter than 12 characters; anyone who has it can spend your credits. Use a long random value.");
  if (!env.SIFT_ACCESS_CODE && env.MESHY_ACCESS_CODE) warnings.push("Using the legacy MESHY_ACCESS_CODE fallback; prefer SIFT_ACCESS_CODE.");
  const leaked = Object.keys(env).filter((name) => name.startsWith("NEXT_PUBLIC_") && /KEY|SECRET|TOKEN|ACCESS_CODE|FAL|TRIPO|MESHY/i.test(name));
  leaked.forEach((name) => warnings.push(`${name} is NEXT_PUBLIC_ and would ship to the browser. Rename it without the prefix.`));
  const limit = Number(env.SIFT_DAILY_LIMIT || env.MESHY_DAILY_LIMIT || 20);
  if (!Number.isInteger(limit) || limit < 1) warnings.push("Daily limit is not a positive integer; the default of 20 will be used.");
  const providers = PROVIDERS.map((p) => {
    const enabled = env[p.flag] === "true";
    const missing = p.keys.filter((k) => !env[k]);
    const problems = [];
    if (env[p.flag] && env[p.flag] !== "true" && env[p.flag] !== "false") problems.push(`${p.flag} must be exactly "true" or "false" (got a different value, so the provider is OFF).`);
    if (enabled && missing.length) problems.push(`Enabled but missing ${missing.join(", ")}.`);
    if (enabled && !code) problems.push("Enabled but no access code.");
    if (!enabled && missing.length === 0) problems.push(`Key is set but ${p.flag} is not "true", so the provider is OFF.`);
    return { ...p, enabled, keySet: missing.length === 0, configured: enabled && missing.length === 0 && Boolean(code), problems };
  });
  return { providers, warnings, accessCodeSet: Boolean(code) };
}

async function main() {
  const result = evaluate(process.env);
  console.log("Hosted generation preflight (offline; no vendor calls; secret values never printed)\n");
  for (const p of result.providers) {
    console.log(`${p.configured ? "READY      " : "NOT READY  "} ${p.label.padEnd(24)} flag ${p.enabled ? "on " : "off"} | key ${p.keySet ? "set" : "missing"} | access code ${result.accessCodeSet ? "set" : "missing"}   (UNVERIFIED live)`);
    p.problems.forEach((m) => console.log(`             - ${m}`));
  }
  if (result.warnings.length) console.log("\nWarnings:"), result.warnings.forEach((m) => console.log(`  - ${m}`));
  const urlIndex = process.argv.indexOf("--url");
  if (urlIndex > 0 && process.argv[urlIndex + 1]) {
    const url = new URL("/api/providers", process.argv[urlIndex + 1]).toString();
    try {
      const response = await fetch(url, { redirect: "manual" });
      if (response.status >= 300 && response.status < 400) console.log(`\n${url}: redirected (${response.status}) — likely Vercel Deployment Protection; cannot inspect.`);
      else {
        const body = await response.json();
        console.log(`\nDeployed catalog (${url}):`);
        for (const [id, v] of Object.entries(body)) console.log(`  ${id.padEnd(16)} configured=${v.configured} verified=${v.verified}`);
      }
    } catch (error) {
      console.log(`\nCould not read ${url}: ${error.message}`);
    }
  }
  const broken = result.providers.some((p) => p.problems.some((m) => /Enabled but|must be exactly/.test(m)));
  process.exitCode = broken ? 1 : 0;
}

import { fileURLToPath } from "node:url";
if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
