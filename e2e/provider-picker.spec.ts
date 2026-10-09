import { expect, test } from "@playwright/test";
import { EVIDENCE, fakeHostedApi, openStudio, selectNode } from "./helpers";

for (const [name, size] of [["desktop", { width: 1280, height: 900 }], ["mobile", { width: 390, height: 844 }]] as const) {
  test(`the provider picker shows exactly Local and Tripo as equal cards (${name})`, async ({ page, context }) => {
    await page.setViewportSize(size);
    await fakeHostedApi(context); // Tripo configured: both cards in their normal state
    await openStudio(page);
    await selectNode(page, "Generation");
    const picker = page.locator("fieldset.provider-picker");
    await picker.scrollIntoViewIfNeeded();
    await expect(picker.getByRole("radio")).toHaveCount(2);
    await expect(picker.locator("label.provider-option")).toHaveCount(2);
    await expect(picker.locator("label.provider-option")).toHaveText([/^Local procedural\s*Free$/, /^Tripo\s*Unverified\s*≈ \$0\.30 per model \(estimate\)$/]);
    const [a, b] = await Promise.all([0, 1].map((i) => picker.locator("label.provider-option").nth(i).boundingBox()));
    expect(Math.abs(a!.height - b!.height)).toBeLessThanOrEqual(1);
    expect(Math.abs(a!.width - b!.width)).toBeLessThanOrEqual(1);
    expect(Math.abs(b!.y - (a!.y + a!.height) - 8)).toBeLessThanOrEqual(1); // uniform gap
    await expect(picker).not.toContainText(/Meshy|Hunyuan|HY 3D|\(paid\)|not a quote/);
    await picker.screenshot({ path: `${EVIDENCE}/12-provider-picker-${name}.png` });
  });
}

test("an unconfigured Tripo keeps the same card height and a one-line hint", async ({ page }) => {
  await openStudio(page); // real /api/providers: nothing configured in the test environment
  await selectNode(page, "Generation");
  const picker = page.locator("fieldset.provider-picker");
  const tripo = picker.locator('label[data-provider="tripo"]');
  await expect(tripo).toContainText("Not configured on this server");
  await expect(tripo.locator("input")).toBeDisabled();
  await expect(tripo).not.toContainText(/TRIPO_|SIFT_ACCESS_CODE|Set /);
  const [a, b] = await Promise.all([picker.locator('label[data-provider="procedural"]'), tripo].map((l) => l.boundingBox()));
  expect(Math.abs(a!.height - b!.height)).toBeLessThanOrEqual(1);
  await picker.screenshot({ path: `${EVIDENCE}/12-provider-picker-unconfigured.png` });
});
