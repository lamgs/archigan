import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { validateBytes } from "gltf-validator";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Box3, Mesh, Vector3, type Object3D } from "three";
import { computeLayout } from "../src/lib/geometry";
import { layoutComplexity } from "../src/lib/limits";
import { deriveBuildingSpec } from "../src/lib/typologies";
import { sceneMetrics } from "../src/lib/viewer";
import { EVIDENCE, caption, canvasFingerprint, centreColour, viewerCentreColour, fakeHostedApi, FAKE_PROVIDERS, fitView, inspectPng, openSavedProject, openStudio, reopenFirstProject, savedBadge, selectNode, storedProjects, viewportTransform, watchErrors } from "./helpers";
import type { FakeProviderId } from "./helpers";

const volumeField = (page: Page, label: string) => page.locator(`.inspector fieldset:has(legend:text("Volume")) label:has(span:text-is("${label}")) input`);
const levelsOf = (text: string) => Number(/(\d+) levels/i.exec(text)?.[1]);
async function pickVolume(page: Page, id: string) { await page.locator('.inspector fieldset:has(legend:text("Volume")) select').selectOption(id); }
async function setNumber(locator: ReturnType<Page["locator"]>, value: string) { await locator.fill(value); await locator.press("Enter"); }

// ---------------------------------------------------------------------------------------------------------------------
test("1. a procedural prompt creates a generation artifact with parameter-corresponding geometry", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/");
  await page.screenshot({ path: `${EVIDENCE}/01-first-run-dashboard.png` });
  await page.locator('.chip-row button:has-text("Twin towers")').click();
  await expect(page.locator(".preview-panel")).toContainText("No model yet"); // nothing exists until the node is run
  await page.locator(".node-run").click();
  await expect(page.locator(".preview-panel__caption")).toContainText("3 volumes");
  await savedBadge(page);

  const [project] = await storedProjects(page);
  const artifacts = Object.values<any>(project.artifacts);
  expect(artifacts.every((a) => a.kind === "building-spec")).toBe(true);
  const generated = artifacts.find((a) => a.metadata.origin === "procedural"); // (the example's refinement also snapshots a variation artifact)
  expect(generated.metadata.spec.volumes.map((v: any) => v.role)).toEqual(["podium", "tower", "tower"]);
  expect(Object.values<any>(project.jobs).map((j) => j.status)).toEqual(["completed"]);
  const twin = await canvasFingerprint(page);

  await fitView(page);
  await selectNode(page, "Prompt");
  await page.locator('article[aria-label="Prompt node"] textarea').fill("A cylindrical glass residential tower");
  await page.locator(".node-run").click();
  await expect(page.locator(".preview-panel__caption")).toContainText("⌀");
  expect(await canvasFingerprint(page)).not.toBe(twin); // different prompt → different geometry on screen
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
test("2. orbit, pan, zoom, reset and camera presets work without moving the outer canvas", async ({ page }) => {
  const errors = watchErrors(page);
  await openStudio(page);
  const outer = await viewportTransform(page);
  const canvas = page.locator(".preview-panel__canvas");
  const box = (await canvas.boundingBox())!;
  const baseline = await canvasFingerprint(page);

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2 + 40, { steps: 8 }); await page.mouse.up(); // orbit
  const orbited = await canvasFingerprint(page); expect(orbited).not.toBe(baseline);
  await page.mouse.move(box.x + 200, box.y + 200); await page.mouse.down({ button: "right" }); await page.mouse.move(box.x + 260, box.y + 230, { steps: 6 }); await page.mouse.up({ button: "right" }); // pan
  expect(await canvasFingerprint(page)).not.toBe(orbited);
  const panned = await canvasFingerprint(page);
  await page.mouse.wheel(0, -400); await page.waitForTimeout(400); // zoom
  expect(await canvasFingerprint(page)).not.toBe(panned);
  expect(await viewportTransform(page)).toBe(outer); // the React Flow canvas never moved

  const seen = new Set<string>();
  for (const preset of ["Top", "Front", "Right", "Axonometric", "Perspective"]) {
    await page.locator(`[aria-label="Camera preset"] button:has-text("${preset}")`).click();
    await expect(page.locator(`[aria-label="Camera preset"] button:has-text("${preset}")`)).toHaveAttribute("aria-pressed", "true");
    await page.waitForTimeout(350);
    seen.add(await canvasFingerprint(page));
  }
  expect(seen.size).toBe(5); // five genuinely different views
  await page.locator('.preview-panel__actions button:has-text("Reset view")').click();
  await page.waitForTimeout(500);
  await expect(page.locator('[aria-label="Camera preset"] button[aria-pressed="true"]')).toHaveText("Perspective");
  expect(await canvasFingerprint(page)).toBe(baseline); // reset restores the framed default view

  await canvas.focus(); for (let i = 0; i < 5; i++) await page.keyboard.press("ArrowRight"); await page.keyboard.press("+");
  expect(await canvasFingerprint(page)).not.toBe(baseline); // keyboard orbit/zoom
  await page.keyboard.press("f"); await page.waitForTimeout(400);

  await page.locator("button.viewer-focus").click();
  await expect(page.locator(".preview-panel--focus")).toBeVisible();
  await page.mouse.move(600, 500); await page.mouse.down(); await page.mouse.move(700, 540, { steps: 5 }); await page.mouse.up(); await page.mouse.wheel(0, 200);
  expect(await viewportTransform(page)).toBe(outer); // still isolated in focus mode
  await page.screenshot({ path: `${EVIDENCE}/02-focus-viewer.png` });
  await page.keyboard.press("Escape");
  await expect(page.locator(".preview-panel--focus")).toHaveCount(0);
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
test("3. floor and dimension changes visibly alter geometry and keep the previous version", async ({ page }) => {
  const errors = watchErrors(page);
  await openStudio(page);
  await selectNode(page, "Generation");
  const before = await caption(page);
  const beforePixels = await canvasFingerprint(page);
  await pickVolume(page, "tower");
  await setNumber(volumeField(page, "Floors"), "6");
  await expect.poll(async () => levelsOf(await caption(page))).toBeLessThan(levelsOf(before));
  expect(await canvasFingerprint(page)).not.toBe(beforePixels);

  await setNumber(page.locator('.inspector label:has(span:text-is("Width (m)")) input'), "90");
  await expect.poll(async () => caption(page)).toContain("90 ×");
  await expect(page.locator(".versions li")).toHaveCount(3); // original + two edits, nothing overwritten
  await expect(page.locator(".versions li").first()).toContainText("Original generation");

  await page.locator(".versions li").first().locator('button:has-text("Use this version")').click();
  await expect.poll(() => caption(page)).toBe(before); // the original is intact and restorable
  await expect(page.locator(".versions li")).toHaveCount(3); // later versions are still listed, just not current

  await setNumber(volumeField(page, "Floors"), "500"); // invalid values are rejected with a reason, not applied
  await expect(page.locator(".inspector__error")).toContainText(/floorCount/);
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
test("4. two child revisions stay visible with lineage after reload", async ({ page }) => {
  const errors = watchErrors(page);
  await openStudio(page);
  await selectNode(page, "Generation");
  await page.locator(".node-branch").click();
  await fitView(page);
  // Branch B (selected after branching): parameter edit.
  await pickVolume(page, "tower");
  await setNumber(volumeField(page, "Floors"), "9");
  // Branch A: follow-up prompt.
  await selectNode(page, "Variation", 0);
  await page.locator('article[aria-label="Variation node"] >> nth=0 >> textarea').fill("glass facade, crown roof");
  await page.locator(".save-state").click();
  await savedBadge(page);

  const snapshot = async () => {
    await selectNode(page, "Model", 0); const a = await caption(page);
    await selectNode(page, "Model", 1); const b = await caption(page);
    return { a, b };
  };
  const before = await snapshot();
  expect(before.a).not.toBe(before.b);
  await page.screenshot({ path: `${EVIDENCE}/04-two-branches.png` });

  await reopenFirstProject(page);
  await expect(page.locator(".node-label")).toHaveText([" · Branch A", " · Branch B"]);
  await expect(page.locator(".react-flow__edge")).toHaveCount(5);
  expect(await snapshot()).toEqual(before);
  await selectNode(page, "Variation", 0);
  await expect(page.locator(".versions li").last()).toContainText("glass facade, crown roof");
  await selectNode(page, "Variation", 1);
  await expect(page.locator(".versions li").last()).toContainText("floorCount → 9");

  const [project] = await storedProjects(page);
  const revisions = Object.values<any>(project.revisions);
  const generationArtifact = project.graph.nodes.find((n: any) => n.type === "generation").artifactId;
  const branchRevisions = revisions.filter((r) => ["prompt", "parameters"].includes(r.change) && r.parentArtifactId === generationArtifact);
  expect(branchRevisions.length).toBeGreaterThanOrEqual(2); // both branches point back at the shared parent
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
test("5. the selected camera and render mode produce a correct downloadable PNG", async ({ page }) => {
  const errors = watchErrors(page);
  await openStudio(page);
  await selectNode(page, "Variation");
  await page.locator('.node-next button:has-text("Render")').click();
  await fitView(page);
  await selectNode(page, "Render");
  const setting = (label: string) => page.locator(`.inspector label.field:has(span:text-is("${label}")) select`);
  const resolutions = await setting("Resolution").locator("option").allInnerTexts();
  const gpuAllows1080 = await page.evaluate(() => { const gl = document.createElement("canvas").getContext("webgl2")!; return gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) >= 1920 && gl.getParameter(gl.MAX_VIEWPORT_DIMS)[0] >= 1920; });
  expect(resolutions.some((r) => r.startsWith("1920"))).toBe(gpuAllows1080); // 1080p is offered exactly when the device can render it
  expect(resolutions.some((r) => r.startsWith("1024")) && resolutions.some((r) => r.startsWith("1600"))).toBe(true);

  async function renderAndDownload(settings: Record<string, string>) {
    for (const [label, value] of Object.entries(settings)) await setting(label).selectOption(value);
    await page.locator('button:has-text("Render PNG")').click();
    await expect(page.locator(".render-preview")).toBeVisible();
    await expect(page.locator('button:has-text("Render PNG")')).toBeEnabled();
    const [download] = await Promise.all([page.waitForEvent("download"), page.locator("a:has-text('Download PNG')").click()]);
    const path = await download.path();
    return { download, png: readFileSync(path!) };
  }
  const front = await renderAndDownload({ Camera: "front", "Material view": "clay", Resolution: "1024x1024" });
  const frontInfo = await inspectPng(page, front.png);
  expect(front.download.suggestedFilename()).toBe("sift-render-1024x1024.png");
  expect(frontInfo).toMatchObject({ width: 1024, height: 1024 });
  expect(frontInfo.different).toBeGreaterThan(2000); // the model is really in the image
  await front.download.saveAs(`${EVIDENCE}/05-render-front-clay-1024x1024.png`);

  const axo = await renderAndDownload({ Camera: "axonometric", "Material view": "shaded", Background: "transparent", Resolution: "1600x900" });
  const axoInfo = await inspectPng(page, axo.png);
  expect(axoInfo).toMatchObject({ width: 1600, height: 900 });
  expect(axoInfo.transparent).toBeGreaterThan(10_000); // transparent background really has alpha
  expect(axoInfo.bbox).not.toEqual(frontInfo.bbox); // different camera → different picture
  await axo.download.saveAs(`${EVIDENCE}/05-render-axonometric-1600x900.png`);

  await expect(page.locator('article[aria-label="Render node"] .status')).toHaveText("Ready");
  await setting("Camera").selectOption("top"); // settings change after rendering → the node says it is out of date
  await expect(page.locator('article[aria-label="Render node"] .status')).toHaveText("Needs render");
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
test("6. the GLB contains real, matching geometry and reopens successfully", async ({ page }) => {
  const errors = watchErrors(page);
  await openStudio(page, "Twin towers");
  const [download] = await Promise.all([page.waitForEvent("download"), page.locator('.preview-panel__actions button:text-is("GLB")').click()]);
  const bytes = readFileSync((await download.path())!);
  expect(bytes.subarray(0, 4).toString()).toBe("glTF");
  expect(bytes.readUInt32LE(8)).toBe(bytes.length); // header length matches the file
  const report = await validateBytes(new Uint8Array(bytes)); // Khronos glTF-Validator: an independent check of the file format
  expect(report.issues.messages.filter((m) => m.severity === 0).map((m) => m.code)).toEqual([]);
  expect(report.issues.numErrors).toBe(0);

  const gltf: { scene: Object3D } = await new Promise((resolve, reject) => new GLTFLoader().parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "", resolve as never, reject));
  let meshes = 0, triangles = 0;
  gltf.scene.traverse((child) => { if (child instanceof Mesh) { meshes += 1; triangles += (child.geometry.index?.count ?? child.geometry.attributes.position.count) / 3; } });

  const spec = deriveBuildingSpec(await page.locator('article[aria-label="Prompt node"] textarea').inputValue());
  const layout = computeLayout(spec);
  const expected = layoutComplexity(layout);
  expect(meshes).toBe(expected.meshes + 1); // every slab/glazing band, plus the ground plate
  expect(triangles).toBe(expected.triangles + 12);
  const box = new Box3().setFromObject(gltf.scene); const size = box.getSize(new Vector3());
  const metrics = sceneMetrics(layout);
  expect(size.y).toBeGreaterThan(metrics.size[1]); // ground plate sits below the building
  expect(size.y).toBeLessThan(metrics.size[1] + 1);
  expect(Math.max(metrics.size[0], metrics.size[2])).toBeLessThanOrEqual(46); // building footprint fits on the ground plate
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
test("7. reload restores the complete board, settings, assets and jobs", async ({ page }) => {
  const errors = watchErrors(page);
  await openStudio(page);
  await selectNode(page, "Generation");
  await page.locator(".node-branch").click();
  await fitView(page);
  await selectNode(page, "Variation", 1);
  await page.locator('.node-next button:has-text("Render")').click();
  await fitView(page);
  await selectNode(page, "Render");
  await page.locator('.inspector label.field:has(span:text-is("Camera")) select').selectOption("front");
  await page.locator('button:has-text("Render PNG")').click();
  await expect(page.locator(".render-preview")).toBeVisible();
  await page.locator('[aria-label="Display mode"] button:has-text("Clay")').click();
  await page.locator('[aria-label="Camera preset"] button:has-text("Top")').click();
  await page.locator('button:has-text("Axes")').click();
  await selectNode(page, "Prompt");
  await page.mouse.move(0, 0);
  const node = page.locator('article[aria-label="Prompt node"]');
  const nb = (await node.boundingBox())!;
  await page.mouse.move(nb.x + 60, nb.y + 12); await page.mouse.down(); await page.mouse.move(nb.x + 60, nb.y + 90, { steps: 6 }); await page.mouse.up();
  await page.mouse.move(900, 880); await page.mouse.down(); await page.mouse.move(800, 860, { steps: 6 }); await page.mouse.up();
  const outer = await viewportTransform(page);
  const nodeBefore = (await node.boundingBox())!;
  await savedBadge(page);
  await page.waitForTimeout(1200);
  await savedBadge(page);

  const stored = (await storedProjects(page))[0];
  await page.reload(); // no explicit Save click anywhere in this test
  await openSavedProject(page);
  await expect(page.locator(".react-flow__node")).toHaveCount(stored.graph.nodes.length);
  await expect(page.locator(".react-flow__edge")).toHaveCount(stored.graph.edges.length);
  expect(await viewportTransform(page)).toBe(outer);
  const nodeAfter = (await page.locator('article[aria-label="Prompt node"]').boundingBox())!;
  expect(Math.abs(nodeAfter.x - nodeBefore.x)).toBeLessThan(2); expect(Math.abs(nodeAfter.y - nodeBefore.y)).toBeLessThan(2);
  await expect(page.locator('[aria-label="Display mode"] button[aria-pressed="true"]')).toHaveText("Clay");
  await expect(page.locator('[aria-label="Camera preset"] button[aria-pressed="true"]')).toHaveText("Top");
  await expect(page.locator('[aria-label="Scene helpers"] button[aria-pressed="true"]')).toContainText(["Grid", "Axes", "Shadows"]);
  await selectNode(page, "Render");
  await expect(page.locator(".render-preview")).toBeVisible(); // render PNG comes back from the asset store
  const naturalWidth = await page.locator(".render-preview").evaluate(async (img: HTMLImageElement) => { await img.decode(); return img.naturalWidth; });
  expect(naturalWidth).toBe(1600);
  expect(Object.values<any>(stored.jobs).every((j) => j.status === "completed")).toBe(true);
  expect(Object.values<any>(stored.artifacts).some((a) => a.kind === "render-png")).toBe(true);
  await page.screenshot({ path: `${EVIDENCE}/07-restored-board.png` });
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
test("8. missing credentials leave procedural mode useful and never fake hosted success", async ({ page }) => {
  const errors = watchErrors(page);
  const providers = await (await page.request.get("/api/providers")).json();
  expect(providers.meshy).toMatchObject({ configured: false, verified: false });
  expect(providers.procedural).toMatchObject({ configured: true });

  const attempt = await page.request.post("/api/generate", { headers: { "x-sift-access-code": "anything" }, data: { prompt: "A tower", refinement: "", provider: "meshy", confirmSpend: true } });
  expect(attempt.status()).toBe(503);
  const body = await attempt.json();
  expect(body).toMatchObject({ code: "not-configured" });
  expect(JSON.stringify(body)).not.toMatch(/taskId|succe|completed/i);
  expect((await page.request.get("/api/generate/task-abc-123", { headers: { "x-sift-access-code": "x" } })).status()).toBe(503);

  await openStudio(page);
  await selectNode(page, "Generation");
  const meshy = page.locator('label.radio:has-text("Meshy")');
  await expect(meshy).toContainText("unavailable");
  await expect(meshy.locator("input")).toBeDisabled();
  await expect(page.locator('label.radio:has-text("Local procedural") input')).toBeChecked();
  await page.locator(".node-run").click(); // procedural generation still works
  await expect(page.locator(".preview-panel__caption")).toBeVisible();
  await savedBadge(page);
  const [project] = await storedProjects(page);
  expect(Object.values<any>(project.jobs).every((j) => j.provider === "procedural")).toBe(true);
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
for (const providerId of Object.keys(FAKE_PROVIDERS) as FakeProviderId[]) test(`9. hosted jobs for ${providerId} (mocked contract) handle ids, transitions, errors, assets and reload; live status stays unverified`, async ({ page, context }) => {
  const { label, supportsCancel, costLabel } = FAKE_PROVIDERS[providerId];
  const errors = watchErrors(page, [/status of (402|429)/]); // the mocked API deliberately returns 402 and 429
  const fake = await fakeHostedApi(context, providerId);
  await openStudio(page);
  await selectNode(page, "Generation");
  const option = page.locator(`label.radio[data-provider="${providerId}"]`);
  await expect(option).toContainText("unverified");
  await expect(option).toContainText(`${costLabel} (estimate, not a quote)`);
  await option.click();
  await page.locator(".node-run").click();
  const dialog = page.locator('[role="alertdialog"]');
  await expect(dialog).toContainText(`Spend ${label} credits?`);
  await expect(dialog).toContainText(`${costLabel} (an estimate`);
  await expect(dialog).toContainText(supportsCancel ? "only be cancelled while it is still queued" : "cannot be cancelled once started");
  expect(fake.posts).toHaveLength(0); // nothing sent before explicit confirmation
  await expect(page.locator("button.modal__go")).toBeDisabled();
  await page.screenshot({ path: `${EVIDENCE}/09-paid-confirmation-${providerId}.png` });

  fake.mode = "no-credits";
  await dialog.locator('input[type="password"]').fill("letmein");
  await page.locator("button.modal__go").click();
  await expect(dialog.locator('[role="alert"]')).toContainText("no credits");
  expect(fake.posts).toHaveLength(1);
  expect(fake.posts[0]).toMatchObject({ provider: providerId });
  await expect(dialog.locator('[role="alert"]')).toContainText(label);

  fake.mode = "hold";
  await page.locator("button.modal__go").click();
  await expect(dialog).toHaveCount(0);
  expect(fake.posts[1]).toMatchObject({ provider: providerId, confirmSpend: true });
  expect(fake.codes[1]).toBe("letmein");
  const hostedHint = page.locator(".inspector fieldset:has(legend:has-text('Hosted job')) .inspector__hint").first();
  await expect(hostedHint).toHaveText(`Queued at ${label}`);
  if (supportsCancel) {
    await page.locator('button:has-text("Cancel task")').click();
    await expect(hostedHint).toHaveText("Cancelled");
    expect(fake.deletes).toBe(1);
    expect(fake.deleteProviders).toEqual([providerId]);
  } else {
    await expect(page.locator('button:has-text("Cancel task")')).toHaveCount(0); // no cancel API for this provider
    await expect(page.locator(".inspector")).toContainText(`${label} tasks cannot be cancelled`);
    await page.locator('button:has-text("Stop waiting")').click();
    await expect(hostedHint).toHaveText("Cancelled");
    await expect(page.locator(".inspector")).toContainText(`${label} cannot cancel tasks, so this app only stopped waiting`);
    expect(fake.deletes).toBe(0); // the cancel endpoint is never called
  }

  fake.mode = "ok"; fake.polls = 0;
  await page.locator(".node-run").click();
  await dialog.locator('input[type="password"]').fill("letmein");
  await page.locator("button.modal__go").click();
  await expect(hostedHint).toContainText(/Generating/, { timeout: 20_000 });
  const stored = (await storedProjects(page))[0];
  expect(Object.values<any>(stored.jobs).some((j) => j.provider === providerId && j.providerTaskId === "task-fake-0001" && j.status !== "completed")).toBe(true); // task id persisted before completion

  await page.reload();
  await openSavedProject(page);
  await fitView(page); await selectNode(page, "Generation");
  await expect(page.locator(".inspector")).toContainText("Enter the access code to resume");
  fake.polls = 3;
  await page.locator('.inspector fieldset:has(legend:has-text("Hosted job")) input[type="password"]').fill("letmein");
  await expect(hostedHint).toHaveText("Hosted model saved in this browser", { timeout: 40_000 });
  await expect(page.locator(".preview-panel__caption")).toContainText(`${label} GLB`);
  await expect(page.locator(".preview-panel__caption")).toContainText("unverified");
  await expect(page.locator('[aria-label="Display mode"] button:disabled')).toHaveCount(3);
  await page.locator(".preview-panel__canvas").screenshot({ path: `${EVIDENCE}/09-hosted-model-${providerId}.png` });

  expect(fake.pollProviders.length).toBeGreaterThan(0);
  expect(new Set(fake.pollProviders)).toEqual(new Set([providerId])); // polling always uses the job's own provider
  expect(new Set(fake.modelProviders)).toEqual(new Set([providerId]));
  expect((Object.values<any>((await storedProjects(page))[0].artifacts).find((a) => a.kind === "model-glb")).metadata.origin).toBe(providerId);
  const [download] = await Promise.all([page.waitForEvent("download"), page.locator('button:has-text("Download GLB")').click()]);
  expect(readFileSync((await download.path())!).subarray(0, 4).toString()).toBe("glTF");

  await savedBadge(page); // autosave must settle before a reload, exactly as a user would see it
  await page.reload(); // the model now comes from IndexedDB alone
  await openSavedProject(page);
  await fitView(page); await selectNode(page, "Generation");
  await expect(page.locator(".preview-panel__caption")).toContainText(`${label} GLB`);
  await expect(hostedHint).toHaveText("Hosted model saved in this browser");
  expect((await (await page.request.get("/api/providers")).json())[providerId].verified).toBe(false); // the real endpoint never claims verification
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
test("10. production build is accessible, error-free, and screenshots are retained", async ({ page }) => {
  const errors = watchErrors(page);
  const serious = async (label: string) => {
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    const bad = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(bad.map((v) => `${label}: ${v.id} (${v.nodes.length}) ${v.help}`), label).toEqual([]);
  };
  await page.goto("/");
  await expect(page.locator("nextjs-portal")).toHaveCount(0); // production build: no dev overlay
  await serious("dashboard (first run)");
  await page.locator('.chip-row button:has-text("Terraced tower")').click();
  await page.locator(".node-run").click();
  await expect(page.locator(".preview-panel__caption")).toBeVisible();
  await fitView(page);
  await savedBadge(page);
  await selectNode(page, "Generation");
  await serious("studio with inspector");
  await page.screenshot({ path: `${EVIDENCE}/10-studio.png` });
  await selectNode(page, "Variation");
  await page.locator('.node-next button:has-text("Render")').click();
  await fitView(page);
  await selectNode(page, "Render");
  await expect(page.locator(".inspector")).toContainText("Camera");
  await serious("render inspector");
  await page.locator("button.viewer-focus").click();
  await serious("focus viewer");
  await page.keyboard.press("Escape");
  await page.locator('.brand').click();
  await expect(page.locator(".project-grid").first()).toBeVisible();
  await serious("dashboard (with projects)");
  await page.screenshot({ path: `${EVIDENCE}/10-dashboard.png` });
  expect(errors).toEqual([]);
});

// ---------------------------------------------------------------------------------------------------------------------
test("5b. a render looks like the viewer it was made from (same tone mapping, lighting, and framing)", async ({ page }) => {
  await openStudio(page);
  await page.locator('[aria-label="Camera preset"] button:has-text("Axonometric")').click();
  await page.waitForTimeout(600);
  const viewer = await viewerCentreColour(page);
  await selectNode(page, "Variation");
  await page.locator('.node-next button:has-text("Render")').click();
  await fitView(page);
  await selectNode(page, "Render");
  const setting = (label: string) => page.locator(`.inspector label.field:has(span:text-is("${label}")) select`);
  await setting("Camera").selectOption("axonometric");
  await setting("Resolution").selectOption("1024x1024");
  await page.locator('button:has-text("Render PNG")').click();
  await expect(page.locator(".render-preview")).toBeVisible();
  const [download] = await Promise.all([page.waitForEvent("download"), page.locator("a:has-text('Download PNG')").click()]);
  const rendered = await centreColour(page, readFileSync((await download.path())!));
  rendered.forEach((channel, i) => expect(Math.abs(channel - viewer[i]), `channel ${i}: render ${channel.toFixed(0)} vs viewer ${viewer[i].toFixed(0)}`).toBeLessThan(18));
});
