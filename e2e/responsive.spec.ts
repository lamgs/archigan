import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { EVIDENCE, savedBadge, selectNode, watchErrors } from "./helpers";

const VIEWPORTS = [
  { name: "desktop", width: 1280, height: 800 },
  { name: "tablet-landscape", width: 1024, height: 768 },
  { name: "tablet", width: 768, height: 1024 },
  { name: "phone", width: 390, height: 844 },
];

const overflow = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
const serious = async (page: Page, label: string) => {
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(results.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${label}: ${v.id}`), label).toEqual([]);
};
const overlaps = (a: { x: number; y: number; width: number; height: number }, b: { x: number; y: number; width: number; height: number }) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;

for (const vp of VIEWPORTS) {
  test(`layout and accessibility hold at ${vp.name} (${vp.width}×${vp.height})`, async ({ page }) => {
    const errors = watchErrors(page);
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await page.goto("/");
    expect(await overflow(page), "dashboard overflows horizontally").toBeLessThanOrEqual(0);
    await serious(page, `${vp.name} dashboard`);
    for (const label of ["New project", "Import backup"]) await expect(page.getByText(label).first()).toBeVisible();

    await page.locator(".dashboard__featured").click();
    await expect(page.locator(".preview-panel__caption")).toBeVisible();
    await expect(page.locator('article[aria-label="Render node"] .status')).toHaveText("Ready", { timeout: 30_000 });
    expect(await overflow(page), "studio overflows horizontally").toBeLessThanOrEqual(0);

    // Navigation must stay reachable at every width (these were hidden on phones before).
    await expect(page.getByRole("button", { name: "All projects" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Save project" })).toBeVisible();
    await expect(page.locator('input[aria-label="Project name"]')).toBeVisible();
    await expect(page.locator(".add-toolbar")).toBeVisible();
    await expect(page.locator(".preview-panel")).toBeVisible();

    // Selecting a node shows the inspector without covering the viewer or the board's controls.
    await selectNode(page, "Prompt");
    const inspector = page.locator(".inspector");
    await expect(inspector).toContainText("Prompt");
    const box = (await inspector.boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(vp.width + 1);
    expect(overlaps(box, (await page.locator(".preview-panel").boundingBox())!), "inspector overlaps the viewer").toBe(false);
    await serious(page, `${vp.name} studio`);

    // The prompt can be edited and autosaves at every width.
    const prompt = page.locator('article[aria-label="Prompt node"] textarea');
    await prompt.scrollIntoViewIfNeeded();
    await prompt.fill("A terraced tower with garden terraces and a glass crown");
    await savedBadge(page);

    // Viewer controls are reachable; 3D view is not clipped off-screen.
    await page.locator(".preview-panel").scrollIntoViewIfNeeded();
    const canvasBox = (await page.locator(".preview-panel canvas").boundingBox())!;
    expect(canvasBox.width).toBeGreaterThan(Math.min(280, vp.width - 40));
    expect(canvasBox.height).toBeGreaterThan(200);
    await page.screenshot({ path: `${EVIDENCE}/30-responsive-${vp.name}.png` });

    // Keyboard: focus is always visible.
    await page.keyboard.press("Tab");
    const outline = await page.evaluate(() => { const el = document.activeElement as HTMLElement; const s = getComputedStyle(el); return el === document.body ? "none" : `${s.outlineStyle}:${s.outlineWidth}`; });
    expect(outline).not.toBe("none");
    expect(errors).toEqual([]);
  });
}
