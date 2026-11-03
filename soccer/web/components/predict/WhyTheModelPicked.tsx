import type { Prediction } from "@/lib/types";

/**
 * The reasoning behind each pick, and what it rests on.
 *
 * Every line is generated from the same feature values the model consumed, so
 * this is an account of the forecast rather than a story told over the top of
 * it. The limitations are in the same box on purpose: a reader who is being
 * given a confident number should meet the caveat in the same glance.
 */
export function WhyTheModelPicked({ prediction }: { prediction: Prediction }) {
  const reasons = prediction.reasons ?? [];
  const limits = prediction.limitations ?? [];
  if (!reasons.length && !limits.length) return null;

  return (
    <section className="rounded-lg border border-stroke bg-elevated p-3">
      <h4 className="text-[11px] uppercase tracking-wider text-ink-muted">
        Why the model picked this
      </h4>

      {reasons.length > 0 && (
        <dl className="mt-2 space-y-2.5">
          {reasons.map((r) => (
            <div key={r.market}>
              <dt className="text-[11px] font-semibold text-ink-secondary">{r.label}</dt>
              <dd className="mt-0.5 text-[12px] leading-relaxed text-ink">{r.text}</dd>
            </div>
          ))}
        </dl>
      )}

      {limits.length > 0 && (
        <div className="mt-3 border-t border-stroke pt-2">
          <p className="text-[10px] uppercase tracking-wider text-ink-muted">Read with care</p>
          <ul className="mt-1 space-y-1">
            {limits.map((line) => (
              <li key={line} className="text-[11px] leading-relaxed text-ink-muted">{line}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
