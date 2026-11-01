import type { ReactNode } from "react";

/** The page's section panel: condensed uppercase heading, body, optional footnote. */
export function Panel({ title, children, note, aside }: { title: string; children: ReactNode; note?: string; aside?: ReactNode }) {
  return (
    <section aria-label={title} className="ql-panel min-w-0 p-4 sm:p-5">
      <div className="mb-3 flex items-baseline justify-between gap-3"><h2 className="ql-section-title">{title}</h2>{aside}</div>
      {children}
      {note && <p className="mt-3 text-xs leading-snug text-text-sec">{note}</p>}
    </section>
  );
}
