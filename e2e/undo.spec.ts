import { expect, test } from "@playwright/test";
import { caption, fitView, openStudio, selectNode, watchErrors } from "./helpers";

const undoButton = (page: import("@playwright/test").Page) => page.getByRole("button", { name: "Undo" });
const redoButton = (page: import("@playwright/test").Page) => page.getByRole("button", { name: "Redo" });
const nodes = (page: import("@playwright/test").Page) => page.locator(".react-flow__node");

test("deleting a node can be undone and redone, with its connections", async ({ page }) => {
  const errors = watchErrors(page);
  await openStudio(page);
  const before = { nodes: await nodes(page).count(), edges: await page.locator(".react-flow__edge").count() };
  await selectNode(page, "Model");
  await page.keyboard.press("Backspace");
  await expect(nodes(page)).toHaveCount(before.nodes - 1);
  await expect(page.locator(".react-flow__edge")).toHaveCount(before.edges - 1);
  await page.keyboard.press("Control+z");
  await expect(nodes(page)).toHaveCount(before.nodes);
  await expect(page.locator(".react-flow__edge")).toHaveCount(before.edges);
  await expect(redoButton(page)).toBeEnabled();
  await page.keyboard.press("Control+Shift+z");
  await expect(nodes(page)).toHaveCount(before.nodes - 1);
  await undoButton(page).click(); // toolbar works too
  await expect(nodes(page)).toHaveCount(before.nodes);
  expect(errors).toEqual([]);
});

test("branching, moving a node, and geometry edits are each one undoable step", async ({ page }) => {
  await openStudio(page);
  await selectNode(page, "Generation");
  await page.locator(".node-branch").click();
  await fitView(page);
  await expect(nodes(page)).toHaveCount(6);
  await undoButton(page).click();
  await expect(nodes(page)).toHaveCount(4);
  await expect(page.locator(".node-label")).toHaveCount(0);
  await redoButton(page).click();
  await expect(nodes(page)).toHaveCount(6);
  await expect(page.locator(".node-label")).toHaveText([" · Branch A", " · Branch B"]);

  // Moving a node.
  const prompt = page.locator('article[aria-label="Prompt node"]');
  const start = (await prompt.boundingBox())!;
  await page.mouse.move(start.x + 70, start.y + 12); await page.mouse.down(); await page.mouse.move(start.x + 70, start.y + 120, { steps: 6 }); await page.mouse.up();
  const moved = (await prompt.boundingBox())!;
  expect(Math.abs(moved.y - start.y)).toBeGreaterThan(60);
  await undoButton(page).click();
  await expect.poll(async () => Math.abs((await prompt.boundingBox())!.y - start.y)).toBeLessThan(3);

  // A geometry edit on the Generation node creates a revision; undo moves the pointer back and keeps the later version.
  await selectNode(page, "Generation");
  const original = await caption(page);
  await page.locator('.inspector fieldset:has(legend:text("Volume")) select').selectOption("tower");
  const floors = page.locator('.inspector fieldset:has(legend:text("Volume")) label:has(span:text-is("Floors")) input');
  await floors.fill("5"); await floors.press("Enter");
  await expect.poll(() => caption(page)).not.toBe(original);
  await expect(page.locator(".versions li")).toHaveCount(2);
  await undoButton(page).click();
  await expect.poll(() => caption(page)).toBe(original);
  await expect(page.locator(".versions li")).toHaveCount(2); // immutable: the edited version still exists
});

test("typing is undone per burst, and Ctrl+Z inside a text field keeps its native meaning", async ({ page }) => {
  await openStudio(page);
  await selectNode(page, "Prompt");
  const field = page.locator('article[aria-label="Prompt node"] textarea');
  const original = await field.inputValue();
  await field.fill("A twin tower scheme with glass");
  await page.waitForTimeout(1700); // a pause ends the burst
  await field.fill("A twin tower scheme with glass and brick");
  await field.press("Control+z"); // inside the field: must NOT trigger the board undo
  await expect(undoButton(page)).toBeEnabled();
  expect((await field.inputValue()).length).toBeGreaterThan(0);
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } }); // leave the field
  await undoButton(page).click();
  await expect(field).toHaveValue("A twin tower scheme with glass");
  await undoButton(page).click();
  await expect(field).toHaveValue(original);
});

test("history starts fresh for each project", async ({ page }) => {
  await openStudio(page);
  await selectNode(page, "Generation");
  await page.locator(".node-branch").click();
  await expect(undoButton(page)).toBeEnabled();
  await page.locator(".brand").click();
  await page.locator(".project-grid").first().locator(".project-card__open").first().click();
  await expect(nodes(page).first()).toBeVisible();
  await expect(undoButton(page)).toBeDisabled(); // not possible to undo into a different project's state
});
