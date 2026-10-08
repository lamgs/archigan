import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { EVIDENCE, caption, fitView, openSavedProject, savedBadge, selectNode, watchErrors } from "./helpers";

const card = (page: Page, name: string) => page.locator(".project-card", { hasText: name }).locator(".project-card__open");

test("the featured Terraced Tower Study opens as a complete board and renders itself", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/");
  await page.locator(".dashboard__featured").click();
  await expect(page.locator(".react-flow__node")).toHaveCount(7); // prompt, generation, two variations, two models, render
  await expect(page.locator(".react-flow__edge")).toHaveCount(6);
  await expect(page.locator(".node-label")).toHaveText([" · Branch A", " · Branch B"]);
  await fitView(page);

  // The render node renders itself on open from the current geometry code.
  await expect(page.locator('article[aria-label="Render node"] .status')).toHaveText("Ready", { timeout: 30_000 });
  await selectNode(page, "Render");
  await expect(page.locator(".render-preview")).toBeVisible();
  expect(await page.locator(".render-preview").evaluate(async (img: HTMLImageElement) => { await img.decode(); return [img.naturalWidth, img.naturalHeight]; })).toEqual([1600, 900]);

  // Two branches, genuinely different models, with lineage.
  await selectNode(page, "Model", 0); const a = await caption(page);
  await selectNode(page, "Model", 1); const b = await caption(page);
  expect(a).not.toBe(b);
  await selectNode(page, "Variation", 0);
  await expect(page.locator(".versions li").last()).toContainText("glass facade");
  await selectNode(page, "Variation", 1);
  await expect(page.locator(".versions li").last()).toContainText("setback");
  await selectNode(page, "Generation");
  await page.screenshot({ path: `${EVIDENCE}/21-terraced-tower-study.png` });

  await savedBadge(page); // opening + rendering autosaved the copy
  await page.reload();
  await openSavedProject(page);
  await fitView(page);
  await selectNode(page, "Render");
  await expect(page.locator(".render-preview")).toBeVisible(); // the PNG came back from the asset store
  await expect(page.locator('article[aria-label="Render node"] .status')).toHaveText("Ready");

  // The bundled sample is never modified; opening it again makes another copy.
  await page.locator(".brand").click();
  await card(page, "Terraced Tower Study").last().click();
  await expect(page.locator('input[aria-label="Project name"]')).toHaveValue("Terraced Tower Study 2");
  expect(errors).toEqual([]);
});

test("the sample set covers three structurally distinct typologies", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/");
  await page.locator(".dashboard__featured").click();
  await expect(page.locator(".preview-panel__caption")).toBeVisible();
  await expect(page.locator('article[aria-label="Render node"] .status')).toHaveText("Ready", { timeout: 40_000 }); // let the sample finish rendering before navigating
  const terraced = await caption(page);
  expect(terraced).toMatch(/2 volumes/i);

  await page.locator(".brand").click();
  await card(page, "Twin Towers on a Shared Podium").click();
  await expect(page.locator(".preview-panel__caption")).toContainText("3 volumes"); // podium + two towers
  const twin = await caption(page);

  await page.locator(".brand").click();
  await card(page, "Cylindrical Residence").click();
  await expect(page.locator(".preview-panel__caption")).toContainText("⌀"); // circular footprint
  const cylinder = await caption(page);

  expect(new Set([terraced, twin, cylinder]).size).toBe(3);
  await page.screenshot({ path: `${EVIDENCE}/22-cylindrical-residence.png` });
  expect(errors).toEqual([]);
});

test("first-run guidance explains the three steps and the interpreter's vocabulary", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".steps li")).toHaveCount(3);
  await page.getByRole("button", { name: /New project/ }).click(); // a blank board
  await expect(page.locator(".canvas-hint")).toContainText("Run");
  await expect(page.locator('.chip-row--canvas button:has-text("Cylindrical tower")')).toBeVisible();
  await selectNode(page, "Prompt");
  await expect(page.locator(".inspector")).toContainText("The local engine reads these keywords");
  await expect(page.locator(".inspector")).toContainText("Other words are ignored");
  const axe = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(axe.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => v.id)).toEqual([]);
});

test("switching projects while a render is in flight never leaks into the other project", async ({ page }) => {
  const errors = watchErrors(page);
  // Hold PNG encoding for 3 s so the render is guaranteed to still be in flight when the user switches projects.
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.toBlob;
    HTMLCanvasElement.prototype.toBlob = function (callback, type, quality) { setTimeout(() => original.call(this, callback, type, quality), 3000); };
  });
  await page.goto("/");
  await page.locator(".dashboard__featured").click();
  await expect(page.locator(".react-flow__node")).toHaveCount(7);
  await page.locator(".brand").click(); // leave immediately, while the sample's auto-render is still running
  await card(page, "Twin Towers on a Shared Podium").click();
  await expect(page.locator('input[aria-label="Project name"]')).toHaveValue("Twin Towers on a Shared Podium");
  await page.waitForTimeout(7000); // long enough for the held render to finish in the background
  await expect(page.locator('input[aria-label="Project name"]')).toHaveValue("Twin Towers on a Shared Podium");
  await expect(page.locator(".react-flow__node")).toHaveCount(4); // still the plain twin-tower board: no render node, no foreign artifacts
  await expect(page.locator(".preview-panel__caption")).toContainText("3 volumes");

  // The first project received its own render, wherever the user was at the time.
  await expect.poll(async () => page.evaluate(async () => {
    const db: IDBDatabase = await new Promise((resolve) => { const request = indexedDB.open("sift-projects"); request.onsuccess = () => resolve(request.result); });
    const projects: any[] = await new Promise((resolve) => { const request = db.transaction("projects").objectStore("projects").get("projects-v2"); request.onsuccess = () => resolve(request.result ?? []); });
    return projects.filter((p) => Object.values<any>(p.artifacts).some((a) => a.kind === "render-png")).map((p) => p.name);
  }), { timeout: 40_000 }).toEqual(["Terraced Tower Study"]);
  expect(errors).toEqual([]);
});
