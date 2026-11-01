import type { ReactNode } from "react";
import { Panel } from "./Panel";

export type Call = { label: string; value: string; sub?: ReactNode; mark?: ReactNode };

/** "The model's calls": winner / spread / total (or projected score) as three tiles. */
export function CallTiles({ calls, children }: { calls: Call[]; children?: ReactNode }) {
  return (
    <Panel title="The model's calls">
      <dl className="grid gap-5 sm:grid-cols-3">
        {calls.map(c => (
          <div key={c.label} className="min-w-0">
            <dt className="ql-data-label">{c.label}</dt>
            <dd className="ql-metric mt-1 flex items-center gap-2">{c.value}{c.mark}</dd>
            {c.sub && <dd className="mt-1 text-xs text-text-sec">{c.sub}</dd>}
          </div>
        ))}
      </dl>
      {children}
    </Panel>
  );
}
