import { Construction } from "lucide-react";

/**
 * Honest placeholder for a route that exists in the navigation but is not built
 * yet. Better than a 404: the user stays oriented inside the app and is told
 * plainly what is and is not there.
 */
export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="panel grid min-h-[420px] place-items-center p-8 text-center">
      <div className="max-w-md">
        <div className="mx-auto grid place-items-center w-11 h-11 rounded-full bg-elevated ring-1 ring-stroke">
          <Construction className="w-5 h-5 text-ink-muted" />
        </div>
        <h2 className="mt-4 font-display text-[16px] font-semibold">{title}</h2>
        <p className="mt-2 text-[12.5px] leading-relaxed text-ink-secondary">{detail}</p>
      </div>
    </div>
  );
}
