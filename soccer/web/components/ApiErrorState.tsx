import { AlertTriangle } from "lucide-react";

/**
 * Shown when the requested data is not in the snapshot.
 */
export function ApiErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="grid min-h-screen place-items-center p-6">
      <div className="panel max-w-lg p-7 text-center">
        <div className="mx-auto grid place-items-center w-11 h-11 rounded-full bg-live/15 ring-1 ring-live/30">
          <AlertTriangle className="w-5 h-5 text-live" />
        </div>
        <h1 className="mt-4 font-display text-[17px] font-semibold">
          Cannot load football data
        </h1>
        <p className="mt-2 text-[12.5px] leading-relaxed text-ink-secondary">{message}</p>
        <p className="mt-4 text-[11.5px] leading-relaxed text-ink-muted">
          This site serves a saved snapshot of the football data. The page you asked for
          was not included in the snapshot; try another competition or season.
        </p>
      </div>
    </div>
  );
}
