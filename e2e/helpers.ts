import { expect, type BrowserContext, type Page } from "@playwright/test";

/** Evidence screenshots go to a git-ignored folder by default (so test runs leave no churn); `E2E_EVIDENCE=1` refreshes the committed set in docs/evidence. */
export const EVIDENCE = process.env.E2E_EVIDENCE === "1" ? "docs/evidence" : "test-results/evidence";

/** Opens the app, loads an example brief, runs the Generation node, and fits the board. */
export async function openStudio(page: Page, example = "Terraced tower") {
  await page.goto("/");
  await page.locator(`.chip-row button:has-text("${example}")`).click();
  await page.locator(".node-run").click();
  await expect(page.locator(".preview-panel__caption")).toBeVisible();
  await fitView(page);
}

export async function fitView(page: Page) {
  await page.locator(".react-flow__controls-fitview").click();
  await page.waitForTimeout(450);
}

export async function selectNode(page: Page, name: string, nth = 0) {
  await page.locator(`article[aria-label="${name} node"]`).nth(nth).click({ position: { x: 20, y: 14 } });
  await page.waitForTimeout(250);
}

export const caption = async (page: Page) => (await page.locator(".preview-panel__caption").innerText()).replace(/\s+/g, " ").trim();

/** A cheap fingerprint of what the viewer canvas currently shows (the canvas keeps its drawing buffer). */
export async function canvasFingerprint(page: Page) {
  return page.locator(".preview-panel canvas").evaluate((canvas: HTMLCanvasElement) => {
    const data = canvas.toDataURL("image/png");
    let h = 2166136261;
    for (let i = 0; i < data.length; i += 7) { h ^= data.charCodeAt(i); h = Math.imul(h, 16777619); }
    return `${data.length}:${h >>> 0}`;
  });
}

export const viewportTransform = (page: Page) => page.locator(".react-flow__viewport").getAttribute("style");

/** Reads the persisted project records straight from IndexedDB (what a reload would see). */
export async function storedProjects(page: Page) {
  return page.evaluate(async () => {
    const db: IDBDatabase = await new Promise((resolve, reject) => { const request = indexedDB.open("sift-projects"); request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
    return await new Promise<any[]>((resolve) => { const request = db.transaction("projects").objectStore("projects").get("projects-v2"); request.onsuccess = () => resolve(request.result ?? []); });
  });
}

/** Opens the most recent saved project from the dashboard (waits for the list to load so a sample card is never clicked by mistake). */
export async function openSavedProject(page: Page) {
  await expect(page.getByText("Loading projects…")).toHaveCount(0);
  await expect(page.locator("h2.dashboard__section").first()).toHaveText("Saved in this browser");
  await page.locator(".project-grid").first().locator(".project-card__open").first().click();
}

export async function reopenFirstProject(page: Page) {
  await page.reload();
  await openSavedProject(page);
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await fitView(page);
}

/** Waits for autosave to settle. */
export async function savedBadge(page: Page) {
  try {
    await expect(page.locator(".save-badge")).toContainText("Saved", { timeout: 10_000 });
  } catch (error) {
    const state = await page.evaluate(() => ({ badge: document.querySelector(".save-badge")?.textContent, notice: document.querySelector(".save-state")?.textContent, banner: document.querySelector(".banner")?.textContent?.slice(0, 80), url: location.href }));
    throw new Error(`Autosave did not settle: ${JSON.stringify(state)}\n${(error as Error).message}`);
  }
}

/** Decodes PNG bytes in the browser: real dimensions plus how much of the image differs from its corner colour. */
export async function inspectPng(page: Page, bytes: Buffer) {
  return page.evaluate(async (base64) => {
    const blob = await (await fetch(`data:image/png;base64,${base64}`)).blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d")!; ctx.drawImage(bitmap, 0, 0);
    const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let different = 0, transparent = 0, minX = canvas.width, maxX = 0, minY = canvas.height, maxY = 0;
    for (let y = 0; y < canvas.height; y += 3) for (let x = 0; x < canvas.width; x += 3) {
      const i = (y * canvas.width + x) * 4;
      if (d[i + 3] === 0) transparent++;
      if (Math.abs(d[i] - d[0]) + Math.abs(d[i + 1] - d[1]) + Math.abs(d[i + 2] - d[2]) > 24 || d[i + 3] !== d[3]) { different++; minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
    }
    return { width: bitmap.width, height: bitmap.height, different, transparent, bbox: [minX, minY, maxX, maxY] };
  }, bytes.toString("base64"));
}

/** A minimal valid GLB (a 2×4×2 box) used as the fake hosted model. */
export function boxGlb(): Buffer {
  const pos = new Float32Array([-1,0,-1, 1,0,-1, 1,0,1, -1,0,1, -1,4,-1, 1,4,-1, 1,4,1, -1,4,1]);
  const idx = new Uint16Array([0,1,2,0,2,3, 4,6,5,4,7,6, 0,4,5,0,5,1, 1,5,6,1,6,2, 2,6,7,2,7,3, 3,7,4,3,4,0]);
  const bin = Buffer.concat([Buffer.from(pos.buffer), Buffer.from(idx.buffer)]);
  const json = { asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }], meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 8, type: "VEC3", min: [-1,0,-1], max: [1,4,1] }, { bufferView: 1, componentType: 5123, count: 36, type: "SCALAR" }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: pos.byteLength }, { buffer: 0, byteOffset: pos.byteLength, byteLength: idx.byteLength }], buffers: [{ byteLength: bin.length }] };
  let j = Buffer.from(JSON.stringify(json)); while (j.length % 4) j = Buffer.concat([j, Buffer.from(" ")]);
  let b = bin; while (b.length % 4) b = Buffer.concat([b, Buffer.from([0])]);
  const head = Buffer.alloc(12); head.write("glTF", 0); head.writeUInt32LE(2, 4); head.writeUInt32LE(12 + 8 + j.length + 8 + b.length, 8);
  const c1 = Buffer.alloc(8); c1.writeUInt32LE(j.length, 0); c1.write("JSON", 4);
  const c2 = Buffer.alloc(8); c2.writeUInt32LE(b.length, 0); c2.write("BIN\0", 4);
  return Buffer.concat([head, c1, j, c2, b]);
}

export type FakeProviderId = "tripo";
export const FAKE_PROVIDERS: Record<FakeProviderId, { label: string; costLabel: string; supportsCancel: boolean }> = {
  tripo: { label: "Tripo", costLabel: "≈ $0.30 per model (estimate)", supportsCancel: false },
};

export type HostedFake = { posts: any[]; codes: (string | undefined)[]; deletes: number; polls: number; mode: "ok" | "no-credits" | "hold"; pollProviders: (string | null)[]; deleteProviders: (string | null)[]; modelProviders: (string | null)[] };

/** Fakes the app's own hosted API at the browser boundary (documented-contract mock; never a live vendor call). Every provider is configured. */
export async function fakeHostedApi(context: BrowserContext, provider: FakeProviderId = "tripo"): Promise<HostedFake> {
  const fake: HostedFake = { posts: [], codes: [], deletes: 0, polls: 0, mode: "ok", pollProviders: [], deleteProviders: [], modelProviders: [] };
  const catalog = Object.fromEntries(Object.entries(FAKE_PROVIDERS).map(([id, meta]) => [id, { ...meta, configured: true, enabled: true, hasKey: true, accessCodeRequired: true, verified: false }]));
  await context.route("**/api/providers", (route) => route.fulfill({ json: { procedural: { configured: true, verified: true }, ...catalog } }));
  await context.route("**/api/generate**", async (route) => {
    const request = route.request();
    if (request.method() === "POST") {
      fake.posts.push(JSON.parse(request.postData() ?? "{}")); fake.codes.push(request.headers()["x-sift-access-code"]);
      if (fake.mode === "no-credits") return route.fulfill({ status: 402, json: { error: `The ${FAKE_PROVIDERS[provider].label} account has no credits left for this request.`, code: "insufficient-credits", retryable: false } });
      return route.fulfill({ status: 202, json: { kind: "task", taskId: "task-fake-0001", status: "queued", verified: false } });
    }
    const queryProvider = new URL(request.url()).searchParams.get("provider");
    if (request.method() === "DELETE") { fake.deletes += 1; fake.deleteProviders.push(queryProvider); return route.fulfill({ json: { ok: true, status: "cancelled" } }); }
    if (new URL(request.url()).pathname.endsWith("/model")) { fake.modelProviders.push(queryProvider); return route.fulfill({ status: 200, contentType: "model/gltf-binary", body: boxGlb() }); }
    fake.polls += 1; fake.pollProviders.push(queryProvider);
    if (fake.mode === "hold") return route.fulfill({ json: { task: { providerTaskId: "task-fake-0001", status: "queued", progress: 0 } } });
    const n = fake.polls;
    if (n === 2) return route.fulfill({ status: 429, headers: { "retry-after": "1" }, json: { error: "limited", code: "rate-limited", retryable: true } });
    const task = n <= 1 ? { status: "queued", progress: 0 } : n <= 3 ? { status: "running", progress: 35 } : n <= 4 ? { status: "running", progress: 80 } : { status: "completed", progress: 100, hasModel: true };
    return route.fulfill({ json: { task: { providerTaskId: "task-fake-0001", ...task } } });
  });
  return fake;
}

/** Collects uncaught page errors and unexpected console errors (404s for the missing favicon are ignored). */
export function watchErrors(page: Page, ignore: RegExp[] = []) {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error" && ![/status of 404/, ...ignore].some((pattern) => pattern.test(message.text()))) errors.push(`console: ${message.text()}`); });
  return errors;
}

/** Mean RGB of the central part of a PNG (default: middle 30%), decoded in the browser. */
export async function centreColour(page: Page, bytes: Buffer, fraction = 0.3) {
  return page.evaluate(async ({ base64, fraction }) => {
    const bitmap = await createImageBitmap(await (await fetch(`data:image/png;base64,${base64}`)).blob());
    const canvas = document.createElement("canvas"); canvas.width = bitmap.width; canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d")!; ctx.drawImage(bitmap, 0, 0);
    const w = Math.round(bitmap.width * fraction), h = Math.round(bitmap.height * fraction);
    const d = ctx.getImageData(Math.round((bitmap.width - w) / 2), Math.round((bitmap.height - h) / 2), w, h).data;
    let r = 0, g = 0, b = 0; const n = d.length / 4;
    for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
    return [r / n, g / n, b / n];
  }, { base64: bytes.toString("base64"), fraction });
}

/** Mean RGB of the middle of the live 3D viewer canvas. */
export async function viewerCentreColour(page: Page, fraction = 0.3) {
  const dataUrl = await page.locator(".preview-panel canvas").evaluate((canvas: HTMLCanvasElement) => canvas.toDataURL("image/png"));
  return centreColour(page, Buffer.from(dataUrl.split(",")[1], "base64"), fraction);
}
