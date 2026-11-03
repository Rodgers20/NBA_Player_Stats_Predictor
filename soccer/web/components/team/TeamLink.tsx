import Link from "next/link";
import { cn } from "@/lib/format";

/**
 * Makes a club open its page from anywhere — tables, fixture rows, cards.
 *
 * Unlike {@link PlayerLink} this needs no lookup. The club page resolves any
 * spelling it is handed (`teampage.lake_name_for`), so the name a surface
 * already has is a good enough address, and the link can render on the server.
 *
 * `competition` only picks which league the page opens on; leave it off where
 * the surface spans several and the page will choose one the club plays in.
 */
export function teamHref(name: string, competition?: string | null): string {
  const qs = new URLSearchParams({ team: name });
  if (competition) qs.set("competition", competition);
  return `/team?${qs.toString()}`;
}

export function TeamLink({
  name, competition, className, children,
}: {
  name: string | null | undefined;
  competition?: string | null;
  className?: string;
  children?: React.ReactNode;
}) {
  if (!name) return <span className={className}>{children ?? "—"}</span>;
  return (
    <Link
      href={teamHref(name, competition)}
      className={cn("transition-colors hover:text-pitch", className)}
    >
      {children ?? name}
    </Link>
  );
}
