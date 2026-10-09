import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";

const walk = (dir: string): string[] => readdirSync(dir).flatMap((name) => { const full = join(dir, name); return statSync(full).isDirectory() ? walk(full) : [full]; });

test("no provider secrets, URLs, auth headers, or server-env reads reach the browser bundle", async ({ page }) => {
  const clientFiles = walk(".next/static").filter((file) => /\.(js|css|html|json)$/.test(file));
  expect(clientFiles.length).toBeGreaterThan(3);
  const offenders: string[] = [];
  for (const file of clientFiles) {
    const text = readFileSync(file, "utf8");
    // Variable *names* may appear in setup guidance shown to users; what must never ship is the provider URL, auth headers, or any read of the server environment.
    for (const needle of ["tsk_", "Bearer ", "process.env.TRIPO", "env.TRIPO_API_KEY", "api.meshy.ai", "msy_"]) if (text.includes(needle)) offenders.push(`${file}: ${needle}`);
  }
  expect(offenders).toEqual([]);

  // Runtime: the page, its scripts, and the public API reveal configuration status only.
  const seen: string[] = [];
  page.on("response", async (response) => { if (/\.js|\/api\//.test(response.url())) seen.push(await response.text().catch(() => "")); });
  await page.goto("/");
  await page.locator(".dashboard__featured").click();
  await expect(page.locator(".preview-panel__caption")).toBeVisible();
  const providers = await (await page.request.get("/api/providers")).json();
  expect(Object.keys(providers.tripo).sort()).toEqual(["accessCodeRequired", "configured", "costLabel", "enabled", "hasKey", "label", "supportsCancel", "verified"]);
  expect(JSON.stringify(providers)).not.toMatch(/key"\s*:\s*"/i);
  expect(seen.join("\n")).not.toMatch(/tsk_|api\.meshy\.ai|msy_|Bearer /);
});

test("hosted endpoints reject unauthenticated and malformed requests without calling a provider", async ({ page }) => {
  const post = (headers: Record<string, string>, data: unknown) => page.request.post("/api/generate", { headers, data });
  expect((await post({}, { prompt: "A tower", refinement: "", provider: "tripo", confirmSpend: true })).status()).toBe(503); // not configured → fails closed
  expect((await post({}, { prompt: "A tower", refinement: "", provider: "meshy", confirmSpend: true })).status()).toBe(400); // removed providers are unknown to the server
  expect((await post({}, { prompt: "x", provider: "browser-key" })).status()).toBe(400);
  expect((await page.request.post("/api/generate", { data: "not json", headers: { "content-type": "application/json" } })).status()).toBe(400);
  for (const path of ["/api/generate/../../etc/passwd", "/api/generate/%2e%2e%2fsecrets", "/api/generate/task!!bad/model"]) {
    const response = await page.request.get(path, { headers: { "x-sift-access-code": "x" } });
    expect([400, 404, 503]).toContain(response.status());
    expect(await response.text()).not.toMatch(/root:|secret|Bearer/i);
  }
});
