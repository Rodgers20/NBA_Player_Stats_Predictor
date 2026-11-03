import { snapshotGet } from "@/lib/snapshot";

export const runtime = "nodejs";

export async function GET(request: Request, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const query = Object.fromEntries(new URL(request.url).searchParams);
  const r = await snapshotGet(path.join("/"), query);
  return new Response(r.text, {
    status: r.status,
    headers: {
      "Content-Type": r.contentType,
      "Cache-Control": r.status === 200
        ? "public, s-maxage=3600, stale-while-revalidate=86400"
        : "public, s-maxage=60",
    },
  });
}
