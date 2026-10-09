import { expect, test, type Page } from "@playwright/test";
import { EVIDENCE, openStudio, savedBadge, selectNode, storedProjects, watchErrors } from "./helpers";

const nodes = (page: Page) => page.locator(".react-flow__node");
const edges = (page: Page) => page.locator(".react-flow__edge");
const undo = (page: Page) => page.getByRole("button", { name: "Undo" });
const redo = (page: Page) => page.getByRole("button", { name: "Redo" });
const counts = async (page: Page) => ({ nodes: await nodes(page).count(), edges: await edges(page).count() });

/** Selects an edge the way a keyboard user does: focus it and press Enter. */
async function selectEdge(page: Page, index = 0) {
  const edge = edges(page).nth(index);
  await edge.focus();
  await page.keyboard.press("Enter");
  await expect(edge).toHaveClass(/selected/);
  return edge;
}

test("Delete node button removes the node and its connections as one undoable step", async ({ page }) => {
  const errors = watchErrors(page);
  await openStudio(page);
  const before = await counts(page);
  await selectNode(page, "Model");
  await page.getByRole("button", { name: "Delete node" }).click();
  await expect(nodes(page)).toHaveCount(before.nodes - 1);
  await expect(edges(page)).toHaveCount(before.edges - 1);
  await expect(page.locator('article[aria-label="Model node"]')).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Delete node" })).toHaveCount(0); // nothing selected any more
  await expect(page.locator(".flow-wrap")).toBeFocused(); // focus is not lost to <body>
  await undo(page).click(); // one step restores the node and its edge
  expect(await counts(page)).toEqual(before);
  await expect(redo(page)).toBeEnabled();
  await redo(page).click();
  await expect(nodes(page)).toHaveCount(before.nodes - 1);
  await undo(page).click();
  expect(await counts(page)).toEqual(before);
  expect(errors).toEqual([]);
});

test("deleting a Generation node keeps its artifacts and revisions as history", async ({ page }) => {
  await openStudio(page);
  const before = await counts(page);
  await savedBadge(page);
  const [saved] = await storedProjects(page);
  await selectNode(page, "Generation");
  await page.getByRole("button", { name: "Delete node" }).click();
  await expect(page.locator('article[aria-label="Generation node"]')).toHaveCount(0);
  await expect(edges(page)).toHaveCount(before.edges - 2); // its prompt and downstream connections went with it
  await expect.poll(async () => (await storedProjects(page))[0].graph.nodes.some((n: any) => n.type === "generation")).toBe(false);
  const [after] = await storedProjects(page);
  expect(Object.keys(after.artifacts).sort()).toEqual(Object.keys(saved.artifacts).sort());
  expect(Object.keys(after.jobs)).toEqual([]); // execution state of a deleted node is not stored (the contract forbids orphan jobs); Undo restores it in memory
  expect(Object.keys(after.revisions).sort()).toEqual(Object.keys(saved.revisions).sort());
  await undo(page).click();
  await expect(page.locator('article[aria-label="Generation node"]')).toHaveCount(1);
  await expect(page.locator(".preview-panel__caption")).toBeVisible();
});

test("Delete and Backspace remove the selected node, but never while typing in a field", async ({ page }) => {
  await openStudio(page);
  const before = await counts(page);
  const field = page.locator('article[aria-label="Prompt node"] textarea');
  await selectNode(page, "Prompt");
  await field.click();
  await page.keyboard.press("Control+a");
  await page.keyboard.press("Backspace");
  await page.keyboard.press("Delete");
  await expect(field).toHaveValue("");
  expect(await counts(page)).toEqual(before); // the keys edited text, not the board
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await selectNode(page, "Model");
  await page.keyboard.press("Delete");
  await expect(nodes(page)).toHaveCount(before.nodes - 1);
  await undo(page).click();
  await expect(nodes(page)).toHaveCount(before.nodes);
  await selectNode(page, "Model");
  await page.keyboard.press("Backspace");
  await expect(nodes(page)).toHaveCount(before.nodes - 1);
});

test("a selected connector shows a named remove control; removing it is undoable and leaves a valid graph", async ({ page }) => {
  const errors = watchErrors(page);
  await openStudio(page);
  const before = await counts(page);
  await expect(page.getByRole("button", { name: "Remove connection" })).toHaveCount(0); // only shown for the selected edge
  await selectEdge(page);
  const remove = page.getByRole("button", { name: "Remove connection" });
  await expect(remove).toBeVisible();
  await remove.click();
  await expect(edges(page)).toHaveCount(before.edges - 1);
  await expect(nodes(page)).toHaveCount(before.nodes); // nodes (and their artifacts) are untouched
  await expect(page.locator(".status--blocked").first()).toBeVisible(); // the downstream node reports it is waiting for an input
  await page.screenshot({ path: `${EVIDENCE}/11-connection-removed.png` });
  await undo(page).click();
  expect(await counts(page)).toEqual(before);
  await expect(page.locator(".status--blocked")).toHaveCount(0);
  await redo(page).click();
  await expect(edges(page)).toHaveCount(before.edges - 1);
  await undo(page).click();
  expect(errors).toEqual([]);
});

test("Delete and Backspace remove a selected connector, one undo step each", async ({ page }) => {
  await openStudio(page);
  const before = await counts(page);
  for (const key of ["Delete", "Backspace"]) {
    await selectEdge(page);
    await page.keyboard.press(key);
    await expect(edges(page)).toHaveCount(before.edges - 1);
    expect((await counts(page)).nodes).toBe(before.nodes);
    await undo(page).click();
    expect(await counts(page)).toEqual(before);
  }
  // The connection can be re-made afterwards: removing it never leaves a half-connected port behind.
  await selectEdge(page);
  await page.keyboard.press("Delete");
  await expect(edges(page)).toHaveCount(before.edges - 1);
  await page.keyboard.press("Control+z");
  expect(await counts(page)).toEqual(before);
});
