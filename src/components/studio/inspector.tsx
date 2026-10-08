"use client";

import { useState } from "react";
import type { BuildingSpec, DesignNode, Provider } from "@/lib/contracts";
import { LIMITS, type SpecEdit } from "@/lib/spec-edit";
import { NODE_LABELS } from "@/lib/workflow";

type Limit = { min: number; max: number; step: number };

/** Commits on blur/Enter so a generation-node edit creates one revision, not one per keystroke. */
function NumberField({ label, value, limit, onCommit }: { label: string; value: number; limit: Limit; onCommit: (value: number) => void }) {
  const [edit, setEdit] = useState<{ base: number; text: string }>({ base: value, text: String(value) });
  // Re-sync the draft when the committed value changes from outside (derived state, no effect needed).
  const draft = edit.base === value ? edit.text : String(value);
  const setDraft = (text: string) => setEdit({ base: value, text });
  const commit = () => {
    const parsed = Number(draft);
    if (draft.trim() === "" || !Number.isFinite(parsed)) return setDraft(String(value));
    if (parsed !== value) onCommit(parsed);
  };
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" inputMode="decimal" value={draft} min={limit.min} max={limit.max} step={limit.step} onChange={(event) => setDraft(event.target.value)} onBlur={commit} onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()} />
    </label>
  );
}

function SelectField<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: readonly T[]; onChange: (value: T) => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value as T)}>{options.map((option) => <option key={option} value={option}>{option}</option>)}</select>
    </label>
  );
}

function Geometry({ spec, onEdit }: { spec: BuildingSpec; onEdit: (edit: SpecEdit) => void }) {
  const [volumeId, setVolumeId] = useState(spec.volumes[0]?.id ?? "");
  const volume = spec.volumes.find((item) => item.id === volumeId) ?? spec.volumes[0];
  const fp = spec.footprint;
  return (
    <div className="inspector__group">
      <fieldset>
        <legend>Footprint</legend>
        <SelectField label="Shape" value={fp.type} options={["rectangle", "circle"] as const} onChange={(type) => onEdit({ op: "footprintType", type })} />
        {fp.type === "rectangle" ? (
          <>
            <NumberField label="Width (m)" value={fp.width} limit={LIMITS.width} onCommit={(width) => onEdit({ op: "footprint", width })} />
            <NumberField label="Depth (m)" value={fp.depth} limit={LIMITS.depth} onCommit={(depth) => onEdit({ op: "footprint", depth })} />
          </>
        ) : (
          <NumberField label="Radius (m)" value={fp.radius} limit={LIMITS.radius} onCommit={(radius) => onEdit({ op: "footprint", radius })} />
        )}
        <NumberField label="Floor height (m)" value={spec.floorHeight} limit={LIMITS.floorHeight} onCommit={(value) => onEdit({ op: "floorHeight", value })} />
      </fieldset>

      {volume && (
        <fieldset>
          <legend>Volume</legend>
          <SelectField label="Edit" value={volume.id} options={spec.volumes.map((item) => item.id)} onChange={setVolumeId} />
          <NumberField label="Floors" value={volume.floorCount} limit={LIMITS.floorCount} onCommit={(value) => onEdit({ op: "volume", id: volume.id, field: "floorCount", value })} />
          <NumberField label="Start floor" value={volume.startFloor} limit={LIMITS.startFloor} onCommit={(value) => onEdit({ op: "volume", id: volume.id, field: "startFloor", value })} />
          <NumberField label="Scale" value={volume.footprintScale} limit={LIMITS.footprintScale} onCommit={(value) => onEdit({ op: "volume", id: volume.id, field: "footprintScale", value })} />
          <NumberField label="Offset X (m)" value={volume.offsetX} limit={LIMITS.offsetX} onCommit={(value) => onEdit({ op: "volume", id: volume.id, field: "offsetX", value })} />
          <NumberField label="Offset Z (m)" value={volume.offsetZ} limit={LIMITS.offsetZ} onCommit={(value) => onEdit({ op: "volume", id: volume.id, field: "offsetZ", value })} />
          <NumberField label="Twist (°)" value={volume.rotationDegrees} limit={LIMITS.rotationDegrees} onCommit={(value) => onEdit({ op: "volume", id: volume.id, field: "rotationDegrees", value })} />
          <NumberField label="Taper" value={volume.taper} limit={LIMITS.taper} onCommit={(value) => onEdit({ op: "volume", id: volume.id, field: "taper", value })} />
          <NumberField label="Setback every (floors, 0 = none)" value={volume.setbackEvery ?? 0} limit={{ ...LIMITS.setbackEvery, min: 0 }} onCommit={(value) => onEdit({ op: "setback", id: volume.id, every: value === 0 ? null : value })} />
          {volume.setbackEvery !== undefined && <NumberField label="Setback amount (m)" value={volume.setbackAmount ?? 0} limit={LIMITS.setbackAmount} onCommit={(amount) => onEdit({ op: "setback", id: volume.id, every: volume.setbackEvery ?? 1, amount })} />}
        </fieldset>
      )}

      <fieldset>
        <legend>Facade &amp; roof</legend>
        <SelectField label="Facade" value={spec.facade.style} options={["solid", "horizontal", "vertical", "grid"] as const} onChange={(style) => onEdit({ op: "facade", style })} />
        <NumberField label="Glazing (0–1)" value={spec.facade.glazingRatio} limit={LIMITS.glazingRatio} onCommit={(glazingRatio) => onEdit({ op: "facade", glazingRatio })} />
        <SelectField label="Roof" value={spec.roof.style} options={["flat", "terrace", "crown"] as const} onChange={(style) => onEdit({ op: "roof", style })} />
      </fieldset>

      <fieldset>
        <legend>Materials</legend>
        {Object.entries(spec.materials).map(([id, material]) => (
          <div className="field field--material" key={id}>
            <span>{id}</span>
            <input type="color" aria-label={`${id} colour`} value={material.color} onChange={(event) => onEdit({ op: "material", id, color: event.target.value })} />
            <select aria-label={`${id} kind`} value={material.kind} onChange={(event) => onEdit({ op: "material", id, kind: event.target.value as typeof material.kind })}>{["clay", "concrete", "glass", "metal"].map((kind) => <option key={kind}>{kind}</option>)}</select>
          </div>
        ))}
      </fieldset>
    </div>
  );
}

type Props = {
  node?: DesignNode;
  spec?: BuildingSpec;
  blockedMessage?: string;
  provider: Provider;
  meshyConfigured: boolean;
  revisionCount: number;
  collapsed: boolean;
  error: string | null;
  onToggle: () => void;
  onProvider: (provider: Provider) => void;
  onEdit: (edit: SpecEdit) => void;
  onClearEdits: () => void;
};

export function Inspector({ node, spec, blockedMessage, provider, meshyConfigured, revisionCount, collapsed, error, onToggle, onProvider, onEdit, onClearEdits }: Props) {
  const editable = node && (node.type === "generation" || node.type === "variation") && spec;
  return (
    <aside className={`inspector ${collapsed ? "is-collapsed" : ""}`} aria-label="Node inspector">
      <header>
        <div><span className="section-kicker">Inspector</span><h2>{node ? NODE_LABELS[node.type] : "Nothing selected"}</h2></div>
        <button type="button" onClick={onToggle} aria-expanded={!collapsed} aria-label={collapsed ? "Expand inspector" : "Collapse inspector"}>{collapsed ? "‹" : "›"}</button>
      </header>
      {!collapsed && (
        <div className="inspector__body">
          {!node && <p className="inspector__hint">Select a node on the canvas to see its settings.</p>}
          {node?.type === "prompt" && <p className="inspector__hint">Write the brief in the node. {String(node.params.text ?? "").length}/800 characters. Connect it to a Generation node, then press Run.</p>}
          {node?.type === "generation" && (
            <fieldset>
              <legend>Provider</legend>
              <label className="radio"><input type="radio" name="provider" checked={provider === "procedural"} onChange={() => onProvider("procedural")} /> Local procedural <small>no account needed</small></label>
              <label className="radio"><input type="radio" name="provider" checked={provider === "meshy"} onChange={() => onProvider("meshy")} /> Meshy <small>{meshyConfigured ? "configured · unverified" : "not configured"}</small></label>
            </fieldset>
          )}
          {editable && spec && (
            <>
              <p className="inspector__hint">{node.type === "generation" ? `Each change creates a new revision; earlier versions are kept (${revisionCount} so far).` : "Changes are stored on this variation; the source model stays untouched."}</p>
              {error && <p role="alert" className="inspector__error">{error}</p>}
              <Geometry key={node.id + spec.volumes.map((v) => v.id).join()} spec={spec} onEdit={onEdit} />
              {node.type === "variation" && Array.isArray(node.params.edits) && node.params.edits.length > 0 && <button type="button" className="ghost-button" onClick={onClearEdits}>Clear parameter edits</button>}
            </>
          )}
          {node && (node.type === "generation" || node.type === "variation") && !spec && <p className="inspector__hint">{blockedMessage || "Run the generation to unlock geometry controls."}</p>}
          {node?.type === "model" && <p className="inspector__hint">This node displays the upstream model. Edit geometry on its Generation or Variation node. Camera presets and view modes arrive with the expanded viewer.</p>}
          {node?.type === "render" && <p className="inspector__hint">Render settings (camera, lighting, resolution) arrive with the render pipeline. For now this node only shows its upstream model.</p>}
        </div>
      )}
    </aside>
  );
}
