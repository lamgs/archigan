"use client";

import { Background, BackgroundVariant, Controls, ReactFlow, ReactFlowProvider, useEdgesState, useNodesState, useReactFlow, useStore, type Connection, type Edge, type Viewport } from "@xyflow/react";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { DesignEdge, DesignNode, DesignNodeType, GenerationJob as GenerationJobT, Provider, SiftProjectV2, ViewerSettings } from "@/lib/contracts";
import { DEFAULT_VIEWER_SETTINGS } from "@/lib/viewer";
import { NODE_PORTS, validateConnection } from "@/lib/graph";
import { copyFromSample, createBlankProject, EXAMPLE_PROMPTS, isBlankProject, NODE_ORDER, projectSignature, renameProject, validateProjectName } from "@/lib/projects";
import { sampleProjects } from "@/lib/samples";
import { applyTaskUpdate, buildHostedArtifact, completeJob, failIfRetiredProvider, failJob, isActiveJob, isUnknownProviderError, isHostedJob, markRateLimited, newHostedJob, nextPollDelayMs, timeoutIfStale, userCancel } from "@/lib/hosted";
import { isHostedProvider, isSupportedProvider, providerCost, providerLabel, providerSupportsCancel, type ProviderCatalog } from "@/lib/provider-meta";
import { cancelHostedTask, createHostedTask, downloadHostedModel, fetchHostedTask } from "@/lib/hosted-client";
import { probeGpu, renderPng } from "@/lib/render-image";
import { describeRender, parseRenderSettings, supportedResolutions, type GpuLimits, type RenderSettings } from "@/lib/render-settings";
import { backupFilename, exportProjectJson, parseProjectJson } from "@/lib/backup";
import { deleteProject, listProjects, loadAsset, probeStorage, saveAsset, saveProject } from "@/lib/storage";
import type { SpecEdit } from "@/lib/spec-edit";
import { addConnectedNode, addNode, branchFrom, commitVariations, connectNodes, editNodeGeometry, evaluateGraph, nextNodeTypes, NODE_LABELS, previewSpec, recordRender, restoreVersion, runGeneration, versionsOf, type FlowGraph } from "@/lib/workflow";
import { Dashboard } from "./dashboard";
import { Inspector } from "./inspector";
import { PaidConfirm } from "./paid-confirm";
import { removeEdge, removeNode } from "./graph-edit";
import { RemovableEdge } from "./removable-edge";
import { NodeGlyph, StudioNode, type StudioFlowNode } from "./studio-node";

const nodeTypes = { studio: StudioNode };
const edgeTypes = { removable: RemovableEdge };
const ModelPreview = dynamic(() => import("./model-preview").then((module) => module.ModelPreview), {
  ssr: false,
  loading: () => <aside className="preview-panel preview-panel--loading">Preparing 3D study…</aside>,
});

/** Fit options that keep nodes clear of the floating chrome (top pill, left toolbar, bottom pills). */
const FIT_DESKTOP = { padding: { top: "88px", right: "32px", bottom: "72px", left: "84px" }, maxZoom: 1 } as const;
const FIT_NARROW = { padding: { top: "132px", right: "16px", bottom: "72px", left: "16px" }, maxZoom: 1 } as const;
const NARROW_QUERY = "(max-width: 900px)";
const subscribeNarrow = (notify: () => void) => { const query = window.matchMedia(NARROW_QUERY); query.addEventListener("change", notify); return () => query.removeEventListener("change", notify); };
/** True on tablet/phone widths, where the inspector docks under the board instead of beside it. */
const useNarrow = () => useSyncExternalStore(subscribeNarrow, () => window.matchMedia(NARROW_QUERY).matches, () => false);

function ZoomReadout() {
  const zoom = useStore((state) => state.transform[2]);
  return <span className="zoom-readout" aria-hidden="true">{Math.round(zoom * 100)}%</span>;
}

const PORT_COLORS: Record<string, string> = { prompt: "#8f2f24", "building-spec": "#2d2c29", "model-glb": "#4f7a5a", "render-png": "#3d6a8c" };

function toFlowNodes(graph: FlowGraph): StudioFlowNode[] {
  return graph.nodes.map((node) => ({
    id: node.id, type: "studio", position: node.position,
    data: { type: node.type, params: node.params, artifactId: node.artifactId, status: "blocked", message: "", summary: "", nextTypes: [], busy: false, onText: () => {}, onAdd: () => {}, onRun: () => {}, onBranch: () => {}, onCommit: () => {} },
  }));
}
function toFlowEdges(graph: FlowGraph): Edge[] {
  return graph.edges.map((edge) => ({ id: edge.id, source: edge.source, sourceHandle: edge.sourcePort, target: edge.target, targetHandle: edge.targetPort }));
}

type Meta = Pick<SiftProjectV2, "id" | "name" | "createdAt" | "updatedAt" | "artifacts" | "jobs" | "revisions" | "settings" | "viewport">;
const metaOf = (project: SiftProjectV2): Meta => ({ id: project.id, name: project.name, createdAt: project.createdAt, updatedAt: project.updatedAt, artifacts: project.artifacts, jobs: project.jobs, revisions: project.revisions, settings: project.settings, viewport: project.viewport });

function Studio() {
  const flow = useReactFlow();
  const narrow = useNarrow();
  const fitOptions = narrow ? FIT_NARROW : FIT_DESKTOP;
  const wrap = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<"dashboard" | "studio">("dashboard");
  const [saved, setSaved] = useState<SiftProjectV2[]>([]);
  const [loadingProjects, setLoadingProjects] = useState(true);
  const [meta, setMeta] = useState<Meta>(() => metaOf(createBlankProject("pending", new Date(0).toISOString(), [])));
  const [nodes, setNodes, onNodesChange] = useNodesState<StudioFlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [notice, setNotice] = useState("Ready to shape a study.");
  const [catalog, setCatalog] = useState<ProviderCatalog | null>(null);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const [gpu, setGpu] = useState<GpuLimits | null | undefined>(undefined);
  const [rendering, setRendering] = useState(false);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const counter = useRef(0);
  // Async work (saves, renders) must not apply its results to a different project than the one it started on.
  const openProjectId = useRef("pending");
  const [storageOk, setStorageOk] = useState<boolean | null>(null);
  const [accessCode, setAccessCode] = useState("");
  const [confirm, setConfirm] = useState<{ nodeId: string; prompt: string; provider: Provider } | null>(null);
  const [confirmBusy, setConfirmBusy] = useState(false);
  const [confirmError, setConfirmError] = useState<string | null>(null);
  const [hostedError, setHostedError] = useState<string | null>(null);
  const [cancelNotice, setCancelNotice] = useState(false);
  const [hostedBlob, setHostedBlob] = useState<Blob | null>(null);
  const [savedSig, setSavedSig] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveFailed, setSaveFailed] = useState(false);

  useEffect(() => {
    void listProjects().then(setSaved).catch(() => setNotice("Local storage is unavailable; this session still works.")).finally(() => setLoadingProjects(false));
    void probeStorage().then(setStorageOk);
    queueMicrotask(() => setGpu(probeGpu())); // browser-only capability probe
    void fetch("/api/providers", { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error(String(response.status)))))
      .then((data: ProviderCatalog) => setCatalog(data))
      .catch(() => { setCatalog(null); setNotice("Could not read provider status from the server, so hosted providers appear unavailable. Reload to retry."); });
  }, []);

  const graph = useMemo<FlowGraph>(() => ({
    nodes: nodes.map((node): DesignNode => ({ id: node.id, type: node.data.type, position: node.position, params: node.data.params, ...(node.data.artifactId ? { artifactId: node.data.artifactId } : {}) })),
    edges: edges.map((edge): DesignEdge => ({ id: edge.id, source: edge.source, sourcePort: edge.sourceHandle ?? "", target: edge.target, targetPort: edge.targetHandle ?? "" })),
  }), [nodes, edges]);
  const results = useMemo(() => evaluateGraph(graph, meta.artifacts), [graph, meta.artifacts]);
  const signature = useMemo(() => projectSignature({ ...meta, graph }), [meta, graph]);
  const selectedId = nodes.find((node) => node.selected)?.id;
  const preview = useMemo(() => previewSpec(results, graph, selectedId), [results, graph, selectedId]);
  const blank = graph.nodes.filter((node) => node.type === "prompt").every((node) => !String(node.params.text ?? "").trim());

  const uid = useCallback((prefix: string) => `${prefix}-${crypto.randomUUID().slice(0, 8)}-${(counter.current += 1)}`, []);
  // ---- Undo / redo of board edits (artifacts are append-only, so history only moves graph state and pointers) ----
  const graphRef = useRef(graph);
  useEffect(() => { graphRef.current = graph; });
  const history = useRef<{ past: FlowGraph[]; future: FlowGraph[] }>({ past: [], future: [] });
  const [historyState, setHistoryState] = useState({ canUndo: false, canRedo: false });
  const syncHistory = useCallback(() => setHistoryState({ canUndo: history.current.past.length > 0, canRedo: history.current.future.length > 0 }), []);
  const lastTextEdit = useRef(0);
  /** Call BEFORE a user edit changes the board; it remembers what to go back to. */
  const record = useCallback(() => {
    const h = history.current;
    h.past.push(structuredClone(graphRef.current));
    if (h.past.length > 60) h.past.shift();
    h.future = [];
    syncHistory();
  }, [syncHistory]);
  const resetHistory = () => { history.current = { past: [], future: [] }; syncHistory(); };

  const setParam = useCallback((id: string, value: string) => {
    const now = Date.now();
    if (now - lastTextEdit.current > 1500) record(); // one history step per typing burst, not per keystroke
    lastTextEdit.current = now;
    setNodes((current) => current.map((node) => (node.id === id ? { ...node, data: { ...node.data, params: { ...node.data.params, text: value } } } : node)));
  }, [setNodes, record]);

  // The stored project may not contain jobs whose node was deleted (contracts: "Job references a missing node"). They stay in memory so Undo restores the node with its jobs; artifacts and revisions are always kept.
  const buildProject = (base: Meta, g: FlowGraph, viewport: Viewport = base.viewport): SiftProjectV2 => {
    const present = new Set(g.nodes.map((node) => node.id));
    const jobs = Object.fromEntries(Object.entries(base.jobs).filter(([, job]) => present.has(job.nodeId)));
    return { schemaVersion: 2, ...base, jobs, viewport, updatedAt: new Date().toISOString(), graph: g };
  };

  const persist = async (base: Meta = meta, g: FlowGraph = graph) => {
    const named = validateProjectName(base.name);
    if (!named.ok) return setNotice(named.error);
    if (g.nodes.filter((node) => node.type === "prompt").every((node) => !String(node.params.text ?? "").trim())) return setNotice("Add an architectural brief before saving.");
    const viewport = flow.getViewport();
    // Record lineage: snapshot any variation whose follow-up text or parameter edits changed since its last snapshot.
    const committed = commitVariations({ ...g, artifacts: base.artifacts, jobs: base.jobs, revisions: base.revisions }, uid, new Date().toISOString());
    const committedMeta = { ...base, name: named.name, artifacts: committed.artifacts, revisions: committed.revisions };
    const project = buildProject(committedMeta, { nodes: committed.nodes, edges: committed.edges }, viewport);
    setSaving(true);
    try {
      setSaved(await saveProject(project));
      if (openProjectId.current !== project.id) return; // the user opened another project while this write was in flight
      // Merge only what the save produced. Replacing the whole state would roll back edits made while the write was in flight.
      setMeta((current) => ({ ...current, name: project.name, updatedAt: project.updatedAt, artifacts: { ...current.artifacts, ...project.artifacts }, revisions: { ...current.revisions, ...project.revisions }, viewport: flow.getViewport() })); // programmatic fit-view emits no move event, so re-read the live viewport
      setSavedSig(projectSignature(project));
      setSaveFailed(false);
      setNodes((current) => current.map((node) => {
        const updated = committed.nodes.find((item) => item.id === node.id);
        return updated && updated.artifactId !== node.data.artifactId ? { ...node, data: { ...node.data, artifactId: updated.artifactId } } : node;
      }));
      setNotice("Project saved in this browser.");
    } catch {
      setSaveFailed(true);
      setNotice("Could not write to IndexedDB; your open session is unchanged.");
    } finally {
      setSaving(false);
    }
  };

  // Autosave: persist shortly after any change so a refresh never loses work. The ref keeps the timer on the latest closure.
  const persistRef = useRef(persist);
  useEffect(() => { persistRef.current = persist; openProjectId.current = meta.id; });
  const dirty = savedSig !== null && signature !== savedSig;
  useEffect(() => {
    if (view !== "studio" || blank || !dirty) return;
    const timer = window.setTimeout(() => void persistRef.current(), 900);
    return () => window.clearTimeout(timer);
  }, [view, blank, dirty, signature]);
  useEffect(() => {
    const flush = () => { if (document.visibilityState === "hidden" && dirty && !blank && view === "studio") void persistRef.current(); };
    document.addEventListener("visibilitychange", flush);
    window.addEventListener("pagehide", flush);
    return () => { document.removeEventListener("visibilitychange", flush); window.removeEventListener("pagehide", flush); };
  }, [dirty, blank, view]);
  const flushIfDirty = async () => { if (view === "studio" && dirty && !blank) await persistRef.current(); };

  const renderNode = async (id: string) => {
    const node = graph.nodes.find((item) => item.id === id);
    const result = results[id];
    if (!node || !result || result.status === "blocked" || result.output.kind !== "spec") return setNotice("Connect a generated model to this Render node first.");
    const settings = parseRenderSettings(node.params);
    if (!supportedResolutions(gpu ?? null).includes(settings.resolution)) { setNotice("This device cannot render the image (WebGL or the chosen size is unavailable)."); return setRenderError("That resolution is not supported on this device. Choose another size."); }
    setRendering(true);
    setRenderError(null);
    try {
      const image = await renderPng(result.output.spec, settings);
      const artifactId = uid("render");
      await saveAsset(`asset:${artifactId}`, image.blob);
      const recorded = recordRender({ ...graph, artifacts: meta.artifacts, jobs: meta.jobs, revisions: meta.revisions }, id, { artifactId, width: image.width, height: image.height, bytes: image.blob.size }, new Date().toISOString());
      if (!recorded.ok) return setRenderError(recorded.message);
      if (openProjectId.current !== meta.id) {
        // The user moved to another project while rendering: store the result on the project it belongs to, and leave the open one untouched.
        const { autoRender: _auto, ...rest } = recorded.state.nodes.find((node) => node.id === id)?.params ?? {};
        void _auto;
        const finished = recorded.state.nodes.map((node) => (node.id === id ? { ...node, params: rest } : node));
        await saveProject(buildProject({ ...meta, artifacts: recorded.state.artifacts }, { nodes: finished, edges: recorded.state.edges }, meta.viewport));
        return;
      }
      const next = { ...meta, artifacts: recorded.state.artifacts };
      setMeta(next);
      record();
      setNodes((current) => current.map((item) => {
        if (item.id !== id) return item;
        const { autoRender: _auto, ...params } = item.data.params;
        void _auto;
        return { ...item, data: { ...item.data, params, artifactId } };
      }));
      setNotice(`Rendered ${describeRender(settings)}.`);
      void persist(next, { nodes: recorded.state.nodes, edges: recorded.state.edges });
    } catch (error) {
      setRenderError(error instanceof Error ? error.message : "The render failed.");
    } finally {
      setRendering(false);
    }
  };

  // Samples ship Render nodes flagged `autoRender`; they render once, on open, from the current geometry code.
  const renderNodeRef = useRef(renderNode);
  useEffect(() => { renderNodeRef.current = renderNode; });
  const autoRendered = useRef(new Set<string>());
  const pendingAutoRender = view === "studio" && gpu !== undefined ? graph.nodes.find((node) => node.type === "render" && node.params.autoRender === true && !node.artifactId && results[node.id] && results[node.id].status !== "blocked") : undefined;
  useEffect(() => {
    if (!pendingAutoRender) return;
    const key = `${meta.id}:${pendingAutoRender.id}`;
    if (autoRendered.current.has(key)) return;
    autoRendered.current.add(key);
    void renderNodeRef.current(pendingAutoRender.id);
  }, [pendingAutoRender, meta.id]);

  const setRenderSetting = (key: keyof RenderSettings, value: string) => {
    if (!selectedId) return;
    setRenderError(null);
    record();
    setNodes((current) => current.map((node) => (node.id === selectedId ? { ...node, data: { ...node.data, params: { ...node.data.params, [key]: value } } } : node)));
  };

  const downloadBackup = () => {
    const project = buildProject(meta, graph, flow.getViewport());
    const blob = new Blob([exportProjectJson(project)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    Object.assign(document.createElement("a"), { href: url, download: backupFilename(project) }).click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Backup downloaded. Render images and hosted models are not included in backups.");
  };

  const importBackup = async (file: File) => {
    const parsed = parseProjectJson(await file.text(), () => crypto.randomUUID(), saved.map((item) => item.id));
    if (!parsed.ok) return setNotice(parsed.error);
    const project = { ...parsed.project, name: validateProjectName(parsed.project.name).ok ? parsed.project.name : "Imported project" };
    if (storageOk === false) { loadNow(project); return setNotice("Imported for this session only — browser storage is unavailable. " + parsed.warnings.join(" ")); }
    try {
      setSaved(await saveProject(project));
      loadNow(project);
      setNotice(["Project imported.", ...parsed.warnings].join(" "));
    } catch {
      setNotice("Could not save the imported project to IndexedDB.");
    }
  };

  // ---- Hosted generation (any provider): explicit confirmation → create → poll → ingest GLB → persist -------------------------
  const patchJob = useCallback((jobId: string, change: (job: GenerationJobT) => GenerationJobT) => setMeta((current) => (current.jobs[jobId] ? { ...current, jobs: { ...current.jobs, [jobId]: change(current.jobs[jobId]) } } : current)), []);

  const startHosted = (nodeId: string, hostedProvider: Provider) => {
    const source = graph.edges.find((edge) => edge.target === nodeId);
    const prompt = String(graph.nodes.find((node) => node.id === source?.source)?.params.text ?? "").trim();
    if (!prompt) return setNotice("Connect a Prompt node with a brief before generating.");
    if (!catalog?.[hostedProvider]?.configured) return setNotice(`${providerLabel(hostedProvider, catalog ?? undefined)} is not configured on this deployment. Use the Local provider.`);
    setConfirmError(null);
    setConfirm({ nodeId, prompt, provider: hostedProvider });
  };

  const confirmSpend = async () => {
    if (!confirm) return;
    setConfirmBusy(true);
    setConfirmError(null);
    const created = await createHostedTask({ prompt: confirm.prompt, refinement: "", code: accessCode.trim(), provider: confirm.provider });
    setConfirmBusy(false);
    if (!created.ok) return setConfirmError(created.error.message);
    const job = newHostedJob({ id: uid("job"), nodeId: confirm.nodeId, taskId: created.value.taskId, provider: confirm.provider as Exclude<Provider, "procedural">, now: new Date().toISOString() });
    const next = { ...meta, jobs: { ...meta.jobs, [job.id]: job } };
    setMeta(next);
    setConfirm(null);
    setHostedError(null);
    setCancelNotice(false);
    setNotice(`${providerLabel(confirm.provider, catalog ?? undefined)} task started. You can keep working; progress is shown in the inspector.`);
    void persist(next); // record the task id immediately so a reload cannot lose a paid task
  };

  const cancelJob = async () => {
    const job = latestJob;
    if (!job) return;
    const now = new Date().toISOString();
    if (job.status === "queued" && job.providerTaskId && accessCode && providerSupportsCancel(job.provider, catalog ?? undefined)) {
      const result = await cancelHostedTask(job.provider, job.providerTaskId, accessCode.trim());
      if (!result.ok && result.error.code !== "running") return setHostedError(result.error.message);
      setCancelNotice(!result.ok);
    } else setCancelNotice(true);
    patchJob(job.id, (current) => userCancel(current, now));
  };

  const downloadHosted = () => {
    if (!hostedBlob) return;
    const url = URL.createObjectURL(hostedBlob);
    const anchor = Object.assign(document.createElement("a"), { href: url, download: "sift-hosted-model.glb" });
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  // A task left over from a provider the product no longer offers cannot be polled: show it as failed instead of waiting forever.
  const unsupportedActiveKey = Object.values(meta.jobs).filter((job) => isHostedJob(job) && isActiveJob(job) && !isSupportedProvider(job.provider)).map((job) => job.id).join(",");
  useEffect(() => {
    if (!unsupportedActiveKey) return;
    const now = new Date().toISOString();
    unsupportedActiveKey.split(",").forEach((id) => patchJob(id, (job) => failIfRetiredProvider(job, now)));
  }, [unsupportedActiveKey, patchJob]);

  const jobsRef = useRef(meta.jobs);
  useEffect(() => { jobsRef.current = meta.jobs; });
  const activeJobKey = view === "studio" ? Object.values(meta.jobs).filter((job) => isHostedJob(job) && isActiveJob(job) && isSupportedProvider(job.provider)).map((job) => job.id).join(",") : "";
  useEffect(() => {
    const code = accessCode.trim();
    if (!activeJobKey || !code) return;
    let cancelled = false;
    const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));
    const loop = async (jobId: string) => {
      let attempt = 0;
      let ingestTries = 0;
      while (!cancelled) {
        const job = jobsRef.current[jobId];
        if (!job || !isActiveJob(job) || !job.providerTaskId) return;
        const stale = timeoutIfStale(job, Date.now());
        if (stale !== job) return patchJob(jobId, () => stale);
        const status = await fetchHostedTask(job.provider, job.providerTaskId, code);
        if (cancelled) return;
        const now = new Date().toISOString();
        let delay = nextPollDelayMs(attempt++);
        if (!status.ok) {
          const { error } = status;
          if (error.code === "rate-limited") { patchJob(jobId, (current) => markRateLimited(current, now)); delay = nextPollDelayMs(attempt, error.retryAfterSeconds ?? 10); }
          else if (isUnknownProviderError(error)) return patchJob(jobId, (current) => failIfRetiredProvider(current, now));
          else if (!error.retryable) return patchJob(jobId, (current) => failJob(current, { code: error.code, message: error.message, retryable: false }, now));
        } else if (status.value.status === "completed") {
          const model = await downloadHostedModel(job.provider, job.providerTaskId, code);
          if (cancelled) return;
          if (model.ok) {
            const artifact = buildHostedArtifact({ artifactId: uid("hosted"), job, bytes: model.value.size, now });
            await saveAsset(artifact.storageKey, model.value);
            if (cancelled || !isActiveJob(jobsRef.current[jobId] ?? { status: "cancelled" })) return;
            setMeta((current) => ({ ...current, artifacts: { ...current.artifacts, [artifact.id]: artifact }, jobs: { ...current.jobs, [jobId]: completeJob(current.jobs[jobId], artifact.id, now) } }));
            setNodes((current) => current.map((node) => (node.id === job.nodeId ? { ...node, data: { ...node.data, params: { ...node.data.params, hostedArtifactId: artifact.id } } } : node)));
            setNotice("Hosted model downloaded and saved in this browser.");
            return;
          }
          if (isUnknownProviderError(model.error)) return patchJob(jobId, (current) => failIfRetiredProvider(current, now));
          if (model.error.retryable && (ingestTries += 1) < 4) delay = 4000 * ingestTries;
          else return patchJob(jobId, (current) => failJob(current, { code: model.error.code, message: model.error.message, retryable: false }, now));
        } else patchJob(jobId, (current) => applyTaskUpdate(current, status.value, now));
        await sleep(delay);
      }
    };
    activeJobKey.split(",").forEach((id) => void loop(id));
    return () => { cancelled = true; };
  }, [activeJobKey, accessCode, patchJob, uid, setNodes]);

  const run = (id: string) => {
    if (graph.nodes.find((node) => node.id === id)?.type === "render") return void renderNode(id);
    if (isHostedProvider(meta.settings.provider) && graph.nodes.find((node) => node.id === id)?.type === "generation") return startHosted(id, meta.settings.provider);
    const result = runGeneration({ ...graph, artifacts: meta.artifacts, jobs: meta.jobs, revisions: meta.revisions }, id, meta.settings.provider, { artifact: uid("artifact"), job: uid("job"), revision: uid("rev") }, new Date().toISOString());
    if (!result.ok) return setNotice(result.message);
    record();
    const next = { ...meta, artifacts: result.state.artifacts, jobs: result.state.jobs, revisions: result.state.revisions };
    const nextGraph = { nodes: result.state.nodes, edges: result.state.edges };
    setMeta(next);
    setNodes((current) => current.map((node) => (node.id === id ? { ...node, data: { ...node.data, artifactId: result.state.nodes.find((item) => item.id === id)?.artifactId } } : node)));
    setNotice("Generated a new building artifact.");
    void persist(next, nextGraph);
  };

  const selectedNode = graph.nodes.find((node) => node.id === selectedId);
  const selectedResult = selectedId ? results[selectedId] : undefined;
  const selectedSpec = selectedResult && selectedResult.status !== "blocked" && selectedResult.output.kind === "spec" ? selectedResult.output.spec : undefined;

  const selectedRenderArtifact = selectedNode?.type === "render" && selectedNode.artifactId ? meta.artifacts[selectedNode.artifactId] : undefined;
  const selectedAssetKey = selectedRenderArtifact?.storageKey;
  useEffect(() => {
    let url: string | null = null;
    let cancelled = false;
    if (selectedAssetKey) {
      void loadAsset(selectedAssetKey).then((blob) => {
        if (cancelled) return;
        if (blob) { url = URL.createObjectURL(blob); setImageUrl(url); } else setImageUrl(null);
      }).catch(() => !cancelled && setImageUrl(null));
    } else queueMicrotask(() => !cancelled && setImageUrl(null));
    return () => { cancelled = true; if (url) URL.revokeObjectURL(url); };
  }, [selectedAssetKey]);

  const renderSettings = parseRenderSettings(selectedNode?.params ?? {});
  const renderState = {
    settings: renderSettings,
    resolutions: supportedResolutions(gpu ?? null),
    gpuKnown: gpu !== undefined,
    busy: rendering,
    error: renderError,
    imageUrl,
    imageInfo: selectedRenderArtifact ? `${selectedRenderArtifact.metadata.width}×${selectedRenderArtifact.metadata.height} PNG · ${Math.round(Number(selectedRenderArtifact.metadata.bytes ?? 0) / 1024)} KB` : null,
    fresh: selectedResult?.status === "ready",
  };

  const latestJob = selectedNode?.type === "generation"
    ? Object.values(meta.jobs).filter((job) => job.nodeId === selectedNode.id && isHostedJob(job)).sort((a, b) => (b.createdAt ?? "").localeCompare(a.createdAt ?? ""))[0]
    : undefined;
  const hostedArtifact = selectedNode?.type === "generation" && typeof selectedNode.params.hostedArtifactId === "string" ? meta.artifacts[selectedNode.params.hostedArtifactId] : undefined;
  const hostedKey = isHostedProvider(meta.settings.provider) ? hostedArtifact?.storageKey : undefined;
  useEffect(() => {
    let cancelled = false;
    if (hostedKey) void loadAsset(hostedKey).then((blob) => !cancelled && setHostedBlob(blob)).catch(() => !cancelled && setHostedBlob(null));
    else queueMicrotask(() => !cancelled && setHostedBlob(null));
    return () => { cancelled = true; };
  }, [hostedKey]);
  const hostedState = {
    configured: Boolean(catalog?.[meta.settings.provider]?.configured),
    accessCode,
    job: latestJob,
    error: hostedError,
    modelInfo: hostedArtifact ? `${Math.round(Number(hostedArtifact.metadata.bytes ?? 0) / 1024)} KB` : null,
    cannotCancelNotice: cancelNotice && latestJob?.status === "cancelled",
  };

  const editGeometry = (edit: SpecEdit) => {
    if (!selectedId) return;
    const result = editNodeGeometry({ ...graph, artifacts: meta.artifacts, jobs: meta.jobs, revisions: meta.revisions }, selectedId, edit, { artifact: uid("artifact"), revision: uid("rev") }, new Date().toISOString());
    if (!result.ok) return setEditError(result.message);
    setEditError(null);
    record();
    const next = { ...meta, artifacts: result.state.artifacts, revisions: result.state.revisions };
    const nextGraph = { nodes: result.state.nodes, edges: result.state.edges };
    setMeta(next);
    setNodes((current) => current.map((node) => {
      const updated = result.state.nodes.find((item) => item.id === node.id);
      return updated && node.id === selectedId ? { ...node, data: { ...node.data, params: updated.params, artifactId: updated.artifactId } } : node;
    }));
    void persist(next, nextGraph);
  };

  const clearEdits = () => {
    if (!selectedId) return;
    record();
    setNodes((current) => current.map((node) => (node.id === selectedId ? { ...node, data: { ...node.data, params: { ...node.data.params, edits: [] } } } : node)));
  };

  const applyGraph = (next: FlowGraph) => {
    setNodes((current) => {
      const known = new Map(current.map((node) => [node.id, node]));
      return toFlowNodes(next).map((node) => {
        const existing = known.get(node.id);
        return existing ? { ...existing, position: node.position, data: { ...existing.data, params: node.data.params, artifactId: node.data.artifactId } } : node;
      });
    });
    setEdges(toFlowEdges(next));
  };

  const undo = () => {
    const h = history.current;
    const previous = h.past.pop();
    if (!previous) return;
    h.future.push(structuredClone(graphRef.current));
    applyGraph(previous);
    syncHistory();
    setNotice("Undone.");
  };
  const redo = () => {
    const h = history.current;
    const next = h.future.pop();
    if (!next) return;
    h.past.push(structuredClone(graphRef.current));
    applyGraph(next);
    syncHistory();
    setNotice("Redone.");
  };
  const undoRef = useRef(undo);
  const redoRef = useRef(redo);
  useEffect(() => { undoRef.current = undo; redoRef.current = redo; });
  useEffect(() => {
    if (view !== "studio") return;
    const onKey = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable)) return; // keep native field undo
      const key = event.key.toLowerCase();
      if (key === "z" && !event.shiftKey) { event.preventDefault(); undoRef.current(); }
      else if ((key === "z" && event.shiftKey) || key === "y") { event.preventDefault(); redoRef.current(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view]);

  // Escape collapses the inspector the same way clicking empty canvas does: by clearing the selection.
  useEffect(() => {
    if (view !== "studio") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      const target = event.target as HTMLElement | null;
      if (target?.tagName === "SELECT" || document.querySelector(".modal-backdrop, .preview-panel--focus")) return;
      setNodes((current) => (current.some((node) => node.selected) ? current.map((node) => (node.selected ? { ...node, selected: false } : node)) : current));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view, setNodes]);

  // On narrow screens the inspector docks under the board and shrinks it, so bring the selected node back into view.
  useEffect(() => {
    if (view !== "studio" || !selectedId || !narrow) return;
    const timer = window.setTimeout(() => void flow.fitView({ nodes: [{ id: selectedId }], ...FIT_NARROW, maxZoom: 0.8, duration: 250 }), 300);
    return () => window.clearTimeout(timer);
  }, [view, selectedId, narrow, flow]);

  const branch = (sourceId: string) => {
    const result = branchFrom(graph, sourceId, { variation: uid("variation"), model: uid("model"), edgeA: uid("edge"), edgeB: uid("edge") });
    if (!result.ok) return setNotice(result.message);
    record();
    applyGraph(result.graph);
    setNodes((current) => current.map((node) => ({ ...node, selected: node.id === result.variationId })));
    setNotice("Branched: a new variation lane starts from the same source. Edit it in the inspector or add a follow-up prompt.");
  };

  const restore = (artifactId: string) => {
    if (!selectedId) return;
    const result = restoreVersion({ ...graph, artifacts: meta.artifacts, jobs: meta.jobs, revisions: meta.revisions }, selectedId, artifactId);
    if (!result.ok) return setNotice(result.message);
    record();
    setNodes((current) => current.map((node) => (node.id === selectedId ? { ...node, data: { ...node.data, artifactId } } : node)));
    setNotice("Switched to the earlier version; later versions are kept.");
    void persist(meta, { nodes: result.state.nodes, edges: result.state.edges });
  };

  const add = (sourceId: string, type: DesignNodeType) => {
    const result = addConnectedNode(graph, sourceId, type, { node: uid(type), edge: uid("edge") });
    if (!result.ok) return setNotice(result.message);
    record();
    applyGraph(result.graph);
    setNotice(`${NODE_LABELS[type]} node added and connected.`);
  };

  const addFree = (type: DesignNodeType) => {
    const box = wrap.current?.getBoundingClientRect();
    const position = box ? flow.screenToFlowPosition({ x: box.left + box.width / 2 + (counter.current % 4) * 24, y: box.top + box.height / 2 + (counter.current % 4) * 24 }) : { x: 100, y: 100 };
    counter.current += 1;
    record();
    applyGraph(addNode(graph, type, uid(type), position));
    setNotice(`${NODE_LABELS[type]} node added. Drag from its ports to connect it.`);
  };

  const onConnect = (connection: Connection) => {
    const result = connectNodes(graph, connection, uid("edge"));
    if (!result.ok) return setNotice(result.message);
    record();
    setEdges(toFlowEdges(result.graph));
  };
  /** Focus returns to the board after a removal, so keyboard users are not left on a button that no longer exists. */
  const focusBoard = () => window.setTimeout(() => wrap.current?.focus(), 0);
  const deleteNode = (id: string) => {
    const result = removeNode(graph, id);
    if (!result.ok) return setNotice(result.message);
    record();
    applyGraph(result.graph);
    setNotice(`${NODE_LABELS[graph.nodes.find((node) => node.id === id)?.type ?? "prompt"]} node deleted with its connections. Undo brings it back.`);
    focusBoard();
  };
  const deleteEdge = (id: string) => {
    const result = removeEdge(graph, id);
    if (!result.ok) return setNotice(result.message);
    record();
    applyGraph(result.graph);
    setNotice("Connection removed. Undo brings it back.");
    focusBoard();
  };
  const isValid = (connection: Connection | Edge) => validateConnection(graph.nodes, graph.edges, { source: connection.source, sourcePort: connection.sourceHandle ?? "", target: connection.target, targetPort: connection.targetHandle ?? "" }).ok;

  const displayNodes = useMemo(() => nodes.map((node): StudioFlowNode => {
    const result = results[node.id];
    const status = !result ? "blocked" : result.status;
    const summary = result && result.status !== "blocked" && result.output.kind === "spec" ? `${Math.max(...result.output.spec.volumes.map((v) => v.startFloor + v.floorCount))} levels · ${result.output.spec.volumes.length} ${result.output.spec.volumes.length === 1 ? "volume" : "volumes"}` : "";
    const message = result && "message" in result && result.message ? result.message : "";
    const renderSummary = node.data.type === "render" && result?.status === "ready" ? `Rendered ${describeRender(parseRenderSettings(node.data.params))}` : summary;
    const busy = (node.data.type === "generation" && Object.values(meta.jobs).some((job) => job.nodeId === node.id && isHostedJob(job) && isActiveJob(job))) || (node.data.type === "render" && rendering && node.id === selectedId);
    return { ...node, data: { ...node.data, status, message, busy, summary: renderSummary, nextTypes: nextNodeTypes(node.data.type), onText: setParam, onAdd: add, onRun: run, onBranch: branch, onCommit: () => { if (!blank) void persist(); } } };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [nodes, results, graph, meta]);

  const flowEdges = useMemo(() => edges.map((edge) => {
    const type = graph.nodes.find((node) => node.id === edge.source)?.type;
    const kind = type ? NODE_PORTS[type].outputs.find((port) => port.id === edge.sourceHandle)?.kind : undefined;
    return { ...edge, type: "removable", data: { onRemove: deleteEdge }, style: { stroke: (kind && PORT_COLORS[kind]) || "#8f2f24", strokeWidth: 1.5 } };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [edges, graph, results]);

  const loadNow = (project: SiftProjectV2) => {
    const copy = structuredClone(project);
    setMeta(metaOf(copy));
    setNodes(toFlowNodes(copy.graph));
    setEdges(toFlowEdges(copy.graph));
    resetHistory();
    setNotice(isBlankProject(copy) ? "Describe a building to begin." : `${copy.name} loaded.`);
    setSavedSig(projectSignature({ ...copy, graph: copy.graph }));
    setView("studio");
  };

  const load = async (project: SiftProjectV2) => {
    await flushIfDirty();
    loadNow(project);
  };

  const goToDashboard = async () => {
    await flushIfDirty();
    setView("dashboard");
  };

  const names = () => saved.map((item) => item.name);
  const newId = () => crypto.randomUUID();
  const startNew = async (example?: (typeof EXAMPLE_PROMPTS)[number]) => {
    await load(createBlankProject(newId(), new Date().toISOString(), names()));
    if (example) applyExample(example);
  };
  const applyExample = (example: (typeof EXAMPLE_PROMPTS)[number]) => {
    setNodes((current) => current.map((node) => (node.data.type === "prompt" ? { ...node, data: { ...node.data, params: { text: example.prompt } } } : node.data.type === "variation" ? { ...node, data: { ...node.data, params: { text: example.refinement } } } : node)));
    setNotice("Example brief loaded — press Run on the Generation node.");
  };
  const openSample = (sample: SiftProjectV2) => void load(copyFromSample(sample, newId(), new Date().toISOString(), names()));

  const renameSaved = async (target: SiftProjectV2, name: string) => {
    const result = renameProject(target, name, new Date().toISOString());
    if (!result.ok) return result.error;
    if (saved.some((item) => item.id !== target.id && item.name.toLowerCase() === result.project.name.toLowerCase())) return "Another project already uses that name.";
    try {
      setSaved(await saveProject(result.project));
      if (target.id === meta.id) setMeta((current) => ({ ...current, name: result.project.name }));
      setNotice(`Renamed to “${result.project.name}”.`);
      return null;
    } catch {
      return "Could not write to IndexedDB; the name was not changed.";
    }
  };

  const deleteSaved = async (target: SiftProjectV2) => {
    try {
      setSaved(await deleteProject(target.id));
      setNotice(`Deleted “${target.name}”.`);
      if (target.id === meta.id) {
        const blankProject = createBlankProject(newId(), new Date().toISOString(), saved.map((item) => item.name));
        setMeta(metaOf(blankProject)); setNodes(toFlowNodes(blankProject.graph)); setEdges(toFlowEdges(blankProject.graph));
      }
    } catch {
      setNotice("Could not delete from IndexedDB; the project is unchanged.");
    }
  };

  const setProvider = (provider: Provider) => {
    setMeta((current) => ({ ...current, settings: { ...current.settings, provider } }));
    if (isHostedProvider(provider)) setNotice(catalog?.[provider]?.configured ? `${providerLabel(provider, catalog)} is a paid provider: you will be asked to confirm before any credits are spent.` : "Hosted generation is not configured on this deployment; the Local provider keeps working.");
  };
  const provider = meta.settings.provider;
  const saveLabel = view !== "studio" ? { state: "idle", text: "" }
    : saving ? { state: "saving", text: "Saving…" }
    : saveFailed ? { state: "error", text: "Not saved — storage error" }
    : blank ? { state: "idle", text: "Not saved yet" }
    : dirty || savedSig === null ? { state: "dirty", text: "Unsaved changes" }
    : { state: "saved", text: `Saved ${new Date(meta.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` };
  const viewerSettings: ViewerSettings = meta.settings.viewer ?? DEFAULT_VIEWER_SETTINGS;

  const studio = view === "studio";
  const sideCollapsed = inspectorCollapsed || !selectedNode;
  return (
    <main className={`studio-shell${studio ? " studio-shell--studio" : ""}`}>
      {!studio && (
        <header className="topbar">
          <a className="brand" href="#workspace" aria-label="Sift home — all projects" onClick={(event) => { event.preventDefault(); void goToDashboard(); }}><span>S</span><strong>Sift</strong><small>AI Architectural Form Studio</small></a>
          <div className="topbar__actions"><span className="save-badge" data-state={saveLabel.state} role="status" aria-live="polite">{saveLabel.text}</span><span className="save-state" role="status">{notice}</span></div>
        </header>
      )}

      {(storageOk === false || saveFailed) && (
        <div className="banner" role="alert">
          <span>{storageOk === false ? "Browser storage is unavailable (private browsing or blocked), so nothing can be saved. Changes will be lost when you close this tab." : "The last save failed (storage may be full or blocked). Your open project is unchanged, but is not saved."}</span>
          {studio && <button type="button" onClick={downloadBackup}>Download backup</button>}
          {studio && storageOk !== false && <button type="button" onClick={() => void persist()}>Retry save</button>}
        </div>
      )}
      {!studio ? (
        <Dashboard projects={saved} samples={sampleProjects} loading={loadingProjects} notice={notice} onNew={startNew} onOpen={load} onOpenSample={openSample} onImport={(file) => void importBackup(file)} onRename={renameSaved} onDelete={deleteSaved} />
      ) : (
        <section className="workspace" id="workspace">
          <section className="canvas-panel" aria-label="Generation workflow">
            <div className="project-pill">
              <a className="brand brand-mark" href="#workspace" aria-label="Sift home — all projects" title="Sift — all projects" onClick={(event) => { event.preventDefault(); void goToDashboard(); }}>S</a>
              <input className="project-pill__name" aria-label="Project name" value={meta.name} onChange={(event) => setMeta((current) => ({ ...current, name: event.target.value }))} />
              <span className="save-badge" data-state={saveLabel.state} role="status" aria-live="polite">{saveLabel.text}</span>
              <span className="project-pill__actions">
                <button type="button" className="pill-button" onClick={() => void persist()}>Save project</button>
                <button type="button" className="pill-button" onClick={() => void goToDashboard()}>All projects</button>
              </span>
            </div>
            <div className="add-toolbar" role="toolbar" aria-label="Add node" aria-orientation="vertical">
              {NODE_ORDER.map((type) => (
                <button type="button" className="tool" key={type} aria-label={`Add ${NODE_LABELS[type]} node`} onClick={() => addFree(type)}>
                  <NodeGlyph type={type} /><span className="tool__tip" aria-hidden="true">{NODE_LABELS[type]}</span>
                </button>
              ))}
              <span className="add-toolbar__rule" aria-hidden="true" />
              <button type="button" className="tool" aria-label="Undo" disabled={!historyState.canUndo} onClick={undo}>
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false"><path d="M6 3L3 6l3 3M3 6h6a4 4 0 0 1 0 8H7" /></svg>
                <span className="tool__tip" aria-hidden="true">Undo (Ctrl/⌘+Z)</span>
              </button>
              <button type="button" className="tool" aria-label="Redo" disabled={!historyState.canRedo} onClick={redo}>
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false"><path d="M10 3l3 3-3 3M13 6H7a4 4 0 0 0 0 8h2" /></svg>
                <span className="tool__tip" aria-hidden="true">Redo (Ctrl/⌘+Shift+Z)</span>
              </button>
            </div>
            <div className="flow-wrap" ref={wrap} tabIndex={-1}>
              <ReactFlow key={meta.id} nodes={displayNodes} edges={flowEdges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeClick={() => setInspectorCollapsed(false)} deleteKeyCode={["Backspace", "Delete"]} onBeforeDelete={async (deletion) => { record(); return deletion; }} onNodeDragStart={() => record()} isValidConnection={isValid} onConnectEnd={(_, state) => { if (state.toNode && !state.isValid) setNotice("Those ports are not compatible, or the connection would create a cycle."); }} nodeTypes={nodeTypes} edgeTypes={edgeTypes} defaultViewport={meta.viewport} onMoveEnd={(_, viewport) => setMeta((current) => ({ ...current, viewport }))} minZoom={0.3} maxZoom={1.5} fitViewOptions={fitOptions} attributionPosition="bottom-left">
                <Background variant={BackgroundVariant.Dots} gap={20} size={1.4} color="#aea797" />
                <Controls showInteractive={false} orientation="horizontal" position="bottom-right" fitViewOptions={fitOptions}><ZoomReadout /></Controls>
              </ReactFlow>
            </div>
            {blank && (
              <div className="canvas-start">
                <p className="canvas-hint">Start here: write a brief in the <strong>Prompt</strong> node (or pick an example below), then press <strong>Run</strong> on the <strong>Generation</strong> node.</p>
                <div className="chip-row chip-row--canvas" aria-label="Example briefs">{EXAMPLE_PROMPTS.map((example) => <button type="button" key={example.label} onClick={() => applyExample(example)}>{example.label}</button>)}</div>
              </div>
            )}
            <p className="save-state" role="status" title={notice}>{notice}</p>
          </section>

          <div className="side">
            {preview || hostedBlob ? (
              <ModelPreview spec={hostedBlob ? undefined : preview?.spec} hosted={hostedBlob ? { blob: hostedBlob, label: `${providerLabel(hostedArtifact?.metadata.origin as string ?? provider, catalog ?? undefined)} GLB` } : undefined} provider={provider} stale={!hostedBlob && Boolean(preview?.stale)} settings={viewerSettings} onSettings={(viewer) => setMeta((current) => ({ ...current, settings: { ...current.settings, viewer } }))} />
            ) : (
              <aside className="preview-panel preview-panel--loading" aria-label="3D study preview"><p>No model yet. Write a brief, connect it to a Generation node, and press Run.</p></aside>
            )}
            <Inspector
              node={selectedNode}
              spec={selectedSpec}
              blockedMessage={selectedResult?.status === "blocked" ? selectedResult.message : undefined}
              provider={provider}
              catalog={catalog}
              revisionCount={Object.keys(meta.revisions).length}
              versions={selectedId ? versionsOf(meta, selectedId, selectedNode?.artifactId) : []}
              onRestore={restore}
              collapsed={sideCollapsed}
              error={editError}
              onToggle={() => setInspectorCollapsed((value) => !value)}
              onProvider={setProvider}
              onEdit={editGeometry}
              onClearEdits={clearEdits}
              render={renderState}
              canRender={Boolean(selectedSpec)}
              onRenderSetting={setRenderSetting}
              onRender={() => void renderNode(selectedId ?? "")}
              hosted={hostedState}
              onAccessCode={setAccessCode}
              onCancelJob={() => void cancelJob()}
              onDownloadHosted={downloadHosted}
              onDeleteNode={() => selectedId && deleteNode(selectedId)}
            />
          </div>
        </section>
      )}
      {confirm && <PaidConfirm providerLabel={providerLabel(confirm.provider, catalog ?? undefined)} costLabel={providerCost(confirm.provider, catalog ?? undefined)} supportsCancel={providerSupportsCancel(confirm.provider, catalog ?? undefined)} prompt={confirm.prompt} accessCode={accessCode} onAccessCode={setAccessCode} busy={confirmBusy} error={confirmError} onConfirm={() => void confirmSpend()} onCancel={() => setConfirm(null)} />}
    </main>
  );
}

export function StudioShell() {
  return <ReactFlowProvider><Studio /></ReactFlowProvider>;
}
