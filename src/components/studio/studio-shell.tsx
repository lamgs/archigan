"use client";

import { Background, BackgroundVariant, Controls, MiniMap, ReactFlow, ReactFlowProvider, addEdge, type Connection, type Edge, useEdgesState, useNodesState } from "@xyflow/react";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { Provider, SiftProject } from "@/lib/contracts";
import { deriveMassing } from "@/lib/massing";
import { deriveBuildingSpec, describeSpec, detectTypology } from "@/lib/typologies";
import { defaultGraph, sampleProjects } from "@/lib/samples";
import { listProjects, saveProject } from "@/lib/storage";
import { StudioNode, type StudioFlowNode, type StudioNodeData } from "./studio-node";

const nodeTypes = { studio: StudioNode };
const ModelPreview = dynamic(() => import("./model-preview").then((module) => module.ModelPreview), {
  ssr: false,
  loading: () => <aside className="preview-panel preview-panel--loading">Preparing 3D study…</aside>,
});

function truncate(value: string, length = 88) {
  return value.length > length ? `${value.slice(0, length).trim()}…` : value;
}

function nodeData(type: SiftProject["graph"]["nodes"][number]["type"], project: Pick<SiftProject, "prompt" | "refinement" | "massing" | "provider">): StudioNodeData {
  if (type === "brief") return { kind: type, eyebrow: "01 / Intent", title: "Design brief", body: truncate(project.prompt), meta: `${project.prompt.length} characters` };
  if (type === "massing") return { kind: type, eyebrow: "02 / Generate", title: "Massing study", body: (() => { const d = describeSpec(deriveBuildingSpec(project.prompt, project.refinement)); return `${d.levels} levels · ${d.volumes} ${d.volumes === 1 ? "volume" : "volumes"} · ${d.footprint}`; })(), meta: project.provider === "procedural" ? "Deterministic local model" : "Hosted preview" };
  if (type === "refine") return { kind: type, eyebrow: "03 / Direct", title: "Refine form", body: project.refinement || "Add a material, void, terrace, or proportion change.", meta: project.refinement ? "Applied to current study" : "Optional" };
  return { kind: type, eyebrow: "04 / Deliver", title: "Export study", body: "Capture the active view or download editable geometry.", meta: "PNG · GLB" };
}

function makeFlowNodes(project: Pick<SiftProject, "prompt" | "refinement" | "massing" | "provider" | "graph">): StudioFlowNode[] {
  return project.graph.nodes.map((node) => ({ id: node.id, type: "studio", position: node.position, data: nodeData(node.type, project) }));
}

function makeProject(source: SiftProject, nodes: StudioFlowNode[], edges: Edge[], name = source.name): SiftProject {
  const now = new Date().toISOString();
  return {
    ...source,
    name,
    updatedAt: now,
    graph: {
      nodes: nodes.map((node) => ({ id: node.id, type: node.data.kind, position: node.position })),
      edges: edges.map((edge) => ({ id: edge.id, source: edge.source, target: edge.target })),
    },
  };
}

function Studio() {
  const initial = sampleProjects[0];
  const [project, setProject] = useState<SiftProject>(initial);
  const [prompt, setPrompt] = useState(initial.prompt);
  const [refinement, setRefinement] = useState(initial.refinement);
  const [provider, setProvider] = useState<Provider>("procedural");
  const [nodes, setNodes, onNodesChange] = useNodesState<StudioFlowNode>(makeFlowNodes(initial));
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(defaultGraph.edges);
  const [saved, setSaved] = useState<SiftProject[]>([]);
  const [notice, setNotice] = useState("Ready to shape a study.");
  const [meshyConfigured, setMeshyConfigured] = useState(false);

  useEffect(() => {
    void listProjects().then(setSaved).catch(() => setNotice("Local storage is unavailable; this session still works."));
    void fetch("/api/providers").then((response) => response.json()).then((data: { meshy?: { configured?: boolean } }) => setMeshyConfigured(Boolean(data.meshy?.configured))).catch(() => setMeshyConfigured(false));
  }, []);

  useEffect(() => {
    setNodes((current) => current.map((node) => ({ ...node, data: nodeData(node.data.kind, { ...project, prompt, refinement, provider }) })));
  }, [project, prompt, refinement, provider, setNodes]);

  const onConnect = useCallback((connection: Connection) => setEdges((current) => addEdge(connection, current)), [setEdges]);

  const generate = () => {
    if (prompt.trim().length < 3) {
      setNotice("Add a more specific architectural brief first.");
      return;
    }
    if (provider === "meshy") {
      setNotice(meshyConfigured ? "Meshy is configured but live credit-spending calls remain disabled in this slice." : "Add the server-only Meshy key to enable hosted generation later.");
      return;
    }
    const next = { ...project, prompt: prompt.trim(), refinement: refinement.trim(), provider, massing: deriveMassing(prompt, refinement), updatedAt: new Date().toISOString() };
    setProject(next);
    setNotice("Local massing regenerated from the full brief.");
  };

  const load = (next: SiftProject) => {
    const copy = structuredClone(next);
    setProject(copy);
    setPrompt(copy.prompt);
    setRefinement(copy.refinement);
    setProvider(copy.provider);
    setNodes(makeFlowNodes(copy));
    setEdges(copy.graph.edges);
    setNotice(`${copy.name} loaded.`);
  };

  const persist = async () => {
    const current = makeProject({ ...project, prompt, refinement, provider }, nodes, edges);
    try {
      const next = await saveProject(current);
      setProject(current);
      setSaved(next);
      setNotice("Project saved in this browser.");
    } catch {
      setNotice("Could not write to IndexedDB; your open session is unchanged.");
    }
  };

  const buildingSpec = useMemo(() => deriveBuildingSpec(project.prompt, project.refinement), [project.prompt, project.refinement]);

  const flowEdges = useMemo(() => edges.map((edge) => ({ ...edge, animated: edge.target === "massing", style: { stroke: "#8f2f24", strokeWidth: 1.8 } })), [edges]);

  return (
    <main className="studio-shell">
      <header className="topbar">
        <a className="brand" href="#workspace" aria-label="Sift home"><span>S</span><strong>Sift</strong><small>Architectural intelligence</small></a>
        <div className="project-title"><span>Project /</span><input aria-label="Project name" value={project.name} onChange={(event) => setProject((current) => ({ ...current, name: event.target.value }))} /></div>
        <div className="topbar__actions"><span className="save-state" role="status">{notice}</span><button type="button" className="ghost-button" onClick={() => void persist()}>Save project</button><button type="button" className="avatar" aria-label="Local profile">G</button></div>
      </header>

      <section className="workspace" id="workspace">
        <nav className="project-rail" aria-label="Projects">
          <div><span className="section-kicker">Starting points</span><h2>Studies</h2></div>
          <div className="sample-list">
            {sampleProjects.map((sample, index) => <button type="button" key={sample.id} onClick={() => load(sample)}><span>0{index + 1}</span><strong>{sample.name}</strong><small>{describeSpec(deriveBuildingSpec(sample.prompt, sample.refinement)).levels} levels · {detectTypology(`${sample.prompt} ${sample.refinement}`.toLowerCase())}</small></button>)}
          </div>
          <div className="saved-list"><span className="section-kicker">Saved here</span>{saved.length === 0 ? <p>No local projects yet.</p> : saved.slice(0, 4).map((item) => <button type="button" key={item.id} onClick={() => load(item)}>{item.name}</button>)}</div>
          <footer><span>Local-first</span><p>Your projects stay in this browser.</p></footer>
        </nav>

        <section className="canvas-panel" aria-label="Generation workflow">
          <header className="canvas-panel__header"><div><span className="section-kicker">Workflow 01</span><h1>Shape the idea</h1></div><div className="provider-switch" aria-label="Generation provider"><button type="button" className={provider === "procedural" ? "is-active" : ""} onClick={() => setProvider("procedural")}>Local</button><button type="button" className={provider === "meshy" ? "is-active" : ""} onClick={() => setProvider("meshy")}>Meshy <i className={meshyConfigured ? "is-configured" : ""} /></button></div></header>
          <div className="flow-wrap">
            <ReactFlow nodes={nodes} edges={flowEdges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.18 }} minZoom={0.45} maxZoom={1.5} attributionPosition="bottom-left">
              <Background variant={BackgroundVariant.Dots} gap={22} size={1.1} color="#b9b2a5" />
              <Controls showInteractive={false} />
              <MiniMap pannable zoomable nodeColor="#8f2f24" maskColor="rgba(236,232,223,.72)" />
            </ReactFlow>
          </div>
          <div className="prompt-dock">
            <label><span>Architectural brief</span><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} maxLength={800} rows={2} placeholder="Describe the building, atmosphere, material, and organization…" /></label>
            <label><span>Refinement <small>optional</small></span><input value={refinement} onChange={(event) => setRefinement(event.target.value)} maxLength={400} placeholder="e.g. Step back upper levels into planted terraces" /></label>
            <button type="button" className="generate-button" onClick={generate}><span>Generate study</span><i aria-hidden="true">↗</i></button>
          </div>
        </section>

        <ModelPreview spec={buildingSpec} provider={provider} />
      </section>
    </main>
  );
}

export function StudioShell() {
  return <ReactFlowProvider><Studio /></ReactFlowProvider>;
}

