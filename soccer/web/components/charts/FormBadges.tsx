/**
 * W/D/L form badges. The letter is always rendered alongside the colour, so
 * form is never encoded by colour alone.
 */
export function FormBadges({ form }: { form: ("W" | "D" | "L")[] }) {
  const bg = { W: "var(--color-form-w)", D: "var(--color-form-d)", L: "var(--color-form-l)" };
  const label = { W: "Win", D: "Draw", L: "Loss" };
  return (
    <div className="flex gap-1" role="list" aria-label="Recent form, most recent last">
      {form.map((f, i) => (
        <span
          key={i}
          role="listitem"
          title={label[f]}
          className="grid place-items-center w-[17px] h-[17px] rounded text-[9px] font-bold text-white"
          style={{ background: bg[f] }}
        >
          {f}
        </span>
      ))}
    </div>
  );
}
