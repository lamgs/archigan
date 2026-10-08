"use client";

import { useState } from "react";
import type { SiftProjectV2 as SiftProject } from "@/lib/contracts";
import { EXAMPLE_PROMPTS, projectBrief } from "@/lib/projects";
import { deriveBuildingSpec, describeSpec, detectTypology } from "@/lib/typologies";

type Props = {
  projects: SiftProject[];
  samples: SiftProject[];
  loading: boolean;
  notice: string;
  onNew: (example?: (typeof EXAMPLE_PROMPTS)[number]) => void;
  onOpen: (project: SiftProject) => void;
  onOpenSample: (sample: SiftProject) => void;
  onImport: (file: File) => void;
  onRename: (project: SiftProject, name: string) => Promise<string | null>;
  onDelete: (project: SiftProject) => Promise<void>;
};

function summary(project: SiftProject) {
  const { prompt, refinement } = projectBrief(project);
  if (!prompt) return "Empty — add a brief";
  return `${describeSpec(deriveBuildingSpec(prompt, refinement)).levels} levels · ${detectTypology(`${prompt} ${refinement}`.toLowerCase())}`;
}

function ProjectCard({ project, onOpen, onRename, onDelete }: { project: SiftProject } & Pick<Props, "onOpen" | "onRename" | "onDelete">) {
  const [mode, setMode] = useState<"view" | "rename" | "confirm-delete">("view");
  const [name, setName] = useState(project.name);
  const [error, setError] = useState<string | null>(null);

  const commit = async () => {
    if (name.trim() === project.name) return setMode("view");
    const problem = await onRename(project, name);
    if (problem) return setError(problem);
    setError(null);
    setMode("view");
  };

  return (
    <li className="project-card">
      {mode === "rename" ? (
        <form onSubmit={(event) => { event.preventDefault(); void commit(); }}>
          <input aria-label={`Rename ${project.name}`} value={name} maxLength={80} autoFocus onChange={(event) => setName(event.target.value)} onKeyDown={(event) => event.key === "Escape" && (setName(project.name), setError(null), setMode("view"))} />
          {error && <p role="alert" className="project-card__error">{error}</p>}
          <div className="project-card__actions"><button type="submit">Save name</button><button type="button" onClick={() => { setName(project.name); setError(null); setMode("view"); }}>Cancel</button></div>
        </form>
      ) : (
        <>
          <button type="button" className="project-card__open" onClick={() => onOpen(project)}>
            <strong>{project.name}</strong>
            <small>{summary(project)}</small>
            <small>Edited {new Date(project.updatedAt).toLocaleDateString()}</small>
          </button>
          {mode === "confirm-delete" ? (
            <div className="project-card__actions" role="group" aria-label={`Confirm deleting ${project.name}`}>
              <span>Delete this project from this browser?</span>
              <button type="button" className="is-danger" onClick={() => void onDelete(project)}>Delete</button>
              <button type="button" onClick={() => setMode("view")}>Keep</button>
            </div>
          ) : (
            <div className="project-card__actions"><button type="button" onClick={() => setMode("rename")}>Rename</button><button type="button" onClick={() => setMode("confirm-delete")}>Delete</button></div>
          )}
        </>
      )}
    </li>
  );
}

export function Dashboard({ projects, samples, loading, notice, onNew, onOpen, onOpenSample, onImport, onRename, onDelete }: Props) {
  const first = !loading && projects.length === 0;
  return (
    <section className="dashboard" aria-label="Projects">
      <header className="dashboard__header">
        <div><span className="section-kicker">{first ? "Welcome" : "Workspace"}</span><h1>{first ? "Start your first study" : "Your studies"}</h1></div>
        <div className="dashboard__header-actions">
          <label className="ghost-button dashboard__import">Import backup<input type="file" accept=".json,application/json" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) onImport(file); event.target.value = ""; }} /></label>
          <button type="button" className="generate-button dashboard__new" onClick={() => onNew()}><span>New project</span><i aria-hidden="true">+</i></button>
        </div>
      </header>
      <p className="dashboard__notice" role="status">{notice}</p>

      {first && (
        <div className="dashboard__first-run">
          <p>Describe a building and Sift shapes it into editable architectural massing. Start from an example brief:</p>
          <div className="chip-row">{EXAMPLE_PROMPTS.map((example) => <button type="button" key={example.label} onClick={() => onNew(example)}>{example.label}</button>)}</div>
        </div>
      )}

      {!first && (
        <>
          <h2 className="dashboard__section">Saved in this browser</h2>
          {loading ? <p>Loading projects…</p> : <ul className="project-grid">{projects.map((project) => <ProjectCard key={project.id} project={project} onOpen={onOpen} onRename={onRename} onDelete={onDelete} />)}</ul>}
        </>
      )}

      <h2 className="dashboard__section">Sample studies</h2>
      <ul className="project-grid">
        {samples.map((sample) => (
          <li key={sample.id} className="project-card">
            <button type="button" className="project-card__open" onClick={() => onOpenSample(sample)}>
              <strong>{sample.name}</strong><small>{summary(sample)}</small><small>Opens as an editable copy</small>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
