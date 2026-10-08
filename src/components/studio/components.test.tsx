// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createWorkflowProject } from "@/lib/projects";
import { DEFAULT_RENDER_SETTINGS } from "@/lib/render-settings";
import { deriveBuildingSpec } from "@/lib/typologies";
import { Dashboard } from "./dashboard";
import { Inspector, type HostedPanelState, type RenderPanelState } from "./inspector";
import { PaidConfirm } from "./paid-confirm";
import { ViewerBoundary, ViewerFallback } from "./viewer-fallback";

afterEach(cleanup);

const NOW = "2026-10-09T10:00:00.000Z";
const project = (id: string, name: string) => ({ ...createWorkflowProject({ id, name, now: NOW, prompt: "A terraced tower", refinement: "" }) });

describe("Dashboard", () => {
  const base = { samples: [project("s1", "Sample One")], loading: false, notice: "", onNew: vi.fn(), onOpen: vi.fn(), onOpenSample: vi.fn(), onImport: vi.fn(), onRename: vi.fn(async () => null), onDelete: vi.fn(async () => {}) };

  it("shows a first-run welcome with example briefs that start a project", () => {
    const onNew = vi.fn();
    render(<Dashboard {...base} projects={[]} onNew={onNew} />);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Start your first study");
    fireEvent.click(screen.getByRole("button", { name: "Twin towers" }));
    expect(onNew).toHaveBeenCalledWith(expect.objectContaining({ label: "Twin towers" }));
  });

  it("lists saved projects and opens samples as copies", () => {
    const onOpen = vi.fn(); const onOpenSample = vi.fn();
    const saved = project("p1", "My Tower");
    render(<Dashboard {...base} projects={[saved]} onOpen={onOpen} onOpenSample={onOpenSample} />);
    fireEvent.click(screen.getByRole("button", { name: /My Tower/ }));
    expect(onOpen).toHaveBeenCalledWith(saved);
    fireEvent.click(screen.getByRole("button", { name: /Sample One/ }));
    expect(onOpenSample).toHaveBeenCalled();
    expect(screen.getByText("Opens as an editable copy")).toBeTruthy();
  });

  it("renames inline and surfaces validation errors from the handler", async () => {
    const onRename = vi.fn(async (_p: unknown, name: string) => (name.trim() ? null : "Give the project a name."));
    render(<Dashboard {...base} projects={[project("p1", "My Tower")]} onRename={onRename} />);
    fireEvent.click(screen.getByRole("button", { name: "Rename" }));
    const input = screen.getByLabelText("Rename My Tower");
    fireEvent.change(input, { target: { value: "  " } });
    fireEvent.click(screen.getByRole("button", { name: "Save name" }));
    expect((await screen.findByRole("alert")).textContent).toBe("Give the project a name.");
    fireEvent.change(input, { target: { value: "Better name" } });
    fireEvent.click(screen.getByRole("button", { name: "Save name" }));
    await vi.waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(onRename).toHaveBeenLastCalledWith(expect.anything(), "Better name");
  });

  it("requires explicit confirmation before deleting, and can be cancelled", () => {
    const onDelete = vi.fn(async () => {});
    render(<Dashboard {...base} projects={[project("p1", "My Tower")]} onDelete={onDelete} />);
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(onDelete).not.toHaveBeenCalled();
    const group = screen.getByRole("group", { name: /Confirm deleting My Tower/ });
    fireEvent.click(within(group).getByRole("button", { name: "Keep" }));
    expect(onDelete).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    fireEvent.click(within(screen.getByRole("group", { name: /Confirm deleting/ })).getByRole("button", { name: "Delete" }));
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it("imports a chosen backup file", () => {
    const onImport = vi.fn();
    const { container } = render(<Dashboard {...base} projects={[]} onImport={onImport} />);
    const file = new File(["{}"], "x.sift.json", { type: "application/json" });
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [file] } });
    expect(onImport).toHaveBeenCalledWith(file);
  });
});

describe("PaidConfirm", () => {
  const props = { prompt: "A tower by the sea", accessCode: "", onAccessCode: vi.fn(), busy: false, error: null, onConfirm: vi.fn(), onCancel: vi.fn() };

  it("names the provider, shows the brief, and blocks confirmation without an access code", () => {
    render(<PaidConfirm {...props} />);
    const dialog = screen.getByRole("alertdialog");
    expect(dialog.textContent).toMatch(/Meshy/);
    expect(dialog.textContent).toMatch(/credits/);
    expect(dialog.textContent).toMatch(/A tower by the sea/);
    expect((screen.getByRole("button", { name: /Spend credits/ }) as HTMLButtonElement).disabled).toBe(true);
  });
  it("confirms only with a code, cancels with the button or Escape, and reports errors", () => {
    const onConfirm = vi.fn(); const onCancel = vi.fn();
    const { rerender } = render(<PaidConfirm {...props} accessCode="letmein" onConfirm={onConfirm} onCancel={onCancel} error="No credits left." />);
    expect(screen.getByRole("alert").textContent).toBe("No credits left.");
    fireEvent.click(screen.getByRole("button", { name: /Spend credits/ }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onCancel).toHaveBeenCalledTimes(1);
    rerender(<PaidConfirm {...props} accessCode="letmein" busy onConfirm={onConfirm} onCancel={onCancel} />);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onCancel).toHaveBeenCalledTimes(1); // Escape is ignored while a request is in flight
  });
  it("keeps the access code hidden", () => {
    render(<PaidConfirm {...props} accessCode="secret" />);
    expect((screen.getByLabelText("Access code") as HTMLInputElement).type).toBe("password");
  });
});

describe("Inspector", () => {
  const spec = deriveBuildingSpec("A terraced stepped tower with a podium");
  const hosted: HostedPanelState = { configured: false, accessCode: "", error: null, modelInfo: null, cannotCancelNotice: false };
  const renderState: RenderPanelState = { settings: DEFAULT_RENDER_SETTINGS, resolutions: ["1024x1024", "1600x900"], gpuKnown: true, busy: false, error: null, imageUrl: null, imageInfo: null, fresh: false };
  const base = { spec, provider: "procedural" as const, meshyConfigured: false, revisionCount: 0, versions: [], onRestore: vi.fn(), collapsed: false, error: null, onToggle: vi.fn(), onProvider: vi.fn(), onEdit: vi.fn(), onClearEdits: vi.fn(), render: renderState, canRender: true, onRenderSetting: vi.fn(), onRender: vi.fn(), hosted, onAccessCode: vi.fn(), onCancelJob: vi.fn(), onDownloadHosted: vi.fn() };
  const node = (type: "prompt" | "generation" | "variation" | "model" | "render") => ({ id: "n", type, position: { x: 0, y: 0 }, params: {} as Record<string, unknown> });

  it("shows only the controls relevant to the selected node", () => {
    const { rerender } = render(<Inspector {...base} node={node("prompt")} spec={undefined} />);
    expect(screen.queryByText("Footprint")).toBeNull();
    expect(screen.queryByText("Camera & look")).toBeNull();
    rerender(<Inspector {...base} node={node("generation")} />);
    expect(screen.getByText("Footprint")).toBeTruthy();
    expect(screen.getByText("Provider")).toBeTruthy();
    expect(screen.queryByText("Camera & look")).toBeNull();
    rerender(<Inspector {...base} node={node("model")} />);
    expect(screen.queryByText("Footprint")).toBeNull();
    expect(screen.getByText(/displays the upstream model/)).toBeTruthy();
    rerender(<Inspector {...base} node={node("render")} />);
    expect(screen.getByText("Camera & look")).toBeTruthy();
    expect(screen.queryByText("Footprint")).toBeNull();
  });

  it("commits numeric edits on blur/Enter only, and ignores empty or unchanged input", () => {
    const onEdit = vi.fn();
    render(<Inspector {...base} node={node("generation")} onEdit={onEdit} />);
    const width = screen.getByLabelText("Width (m)") as HTMLInputElement;
    fireEvent.change(width, { target: { value: "70" } });
    expect(onEdit).not.toHaveBeenCalled(); // typing alone never creates a revision
    fireEvent.blur(width);
    expect(onEdit).toHaveBeenCalledWith({ op: "footprint", width: 70 });
    onEdit.mockClear();
    fireEvent.change(width, { target: { value: "" } });
    fireEvent.blur(width);
    expect(onEdit).not.toHaveBeenCalled();
  });

  it("disables the Meshy option with an explanation when it is not configured", () => {
    render(<Inspector {...base} node={node("generation")} />);
    const meshy = screen.getByLabelText(/Meshy/) as HTMLInputElement;
    expect(meshy.disabled).toBe(true);
    expect(screen.getByText(/not configured on this server/)).toBeTruthy();
    expect((screen.getByLabelText(/Local procedural/) as HTMLInputElement).checked).toBe(true);
  });

  it("explains missing WebGL in the render panel and disables rendering", () => {
    render(<Inspector {...base} node={node("render")} render={{ ...renderState, resolutions: [] }} />);
    expect(screen.getByRole("alert").textContent).toMatch(/cannot create WebGL images/);
    expect((screen.getByRole("button", { name: "Render PNG" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("only offers the supported resolutions and notes when 1080p is missing", () => {
    render(<Inspector {...base} node={node("render")} />);
    const options = within(screen.getByLabelText("Resolution")).getAllByRole("option").map((o) => o.textContent);
    expect(options).toEqual(["1024 × 1024 (square)", "1600 × 900 (wide)"]);
    expect(screen.getByText(/1920 × 1080 is not offered/)).toBeTruthy();
  });

  it("can be collapsed", () => {
    const onToggle = vi.fn();
    render(<Inspector {...base} node={node("generation")} onToggle={onToggle} />);
    fireEvent.click(screen.getByRole("button", { name: "Collapse inspector" }));
    expect(onToggle).toHaveBeenCalled();
    cleanup();
    render(<Inspector {...base} node={node("generation")} collapsed />);
    expect(screen.queryByText("Footprint")).toBeNull();
  });

  it("shows hosted job progress, access-code prompts, and failure messages", () => {
    const job = { id: "j", nodeId: "n", provider: "meshy" as const, providerTaskId: "task-1", status: "running" as const, progress: 40 };
    render(<Inspector {...base} node={node("generation")} provider="meshy" meshyConfigured hosted={{ ...hosted, configured: true, job, error: "Meshy rejected the key." }} />);
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("40");
    expect(screen.getByText(/Enter the access code to resume/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Stop waiting" })).toBeTruthy();
    expect(screen.getByText("Meshy rejected the key.")).toBeTruthy();
  });
});

describe("ViewerFallback / ViewerBoundary", () => {
  it("renders an accessible message with recovery actions", () => {
    const onClick = vi.fn();
    render(<ViewerFallback title="3D preview unavailable" message="No WebGL." actions={[{ label: "Download GLB", onClick }]} />);
    expect(screen.getByRole("alert").textContent).toMatch(/3D preview unavailable/);
    fireEvent.click(screen.getByRole("button", { name: "Download GLB" }));
    expect(onClick).toHaveBeenCalled();
  });

  it("catches render errors, offers a restart, and recovers when reset", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    let shouldThrow = true;
    const Bomb = () => { if (shouldThrow) throw new Error("boom"); return <p>scene ok</p>; };
    const { rerender } = render(<ViewerBoundary resetKey={0} onReset={() => {}}><Bomb /></ViewerBoundary>);
    expect(screen.getByText("The 3D view stopped working")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Restart 3D view" })).toBeTruthy();
    shouldThrow = false;
    rerender(<ViewerBoundary resetKey={1} onReset={() => {}}><Bomb /></ViewerBoundary>);
    expect(screen.getByText("scene ok")).toBeTruthy();
    spy.mockRestore();
  });
});
