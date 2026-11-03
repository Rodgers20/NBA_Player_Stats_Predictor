import { snapshotImage } from "@/lib/snapshot";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const name = new URL(request.url).searchParams.get("name") ?? "";
  const img = await snapshotImage("crest", name);
  if (!img) {
    return Response.json({ success: false, data: null, error: `no crest found for ${name}`, meta: {} }, { status: 404 });
  }
  return new Response(new Uint8Array(img.buf), {
    headers: { "Content-Type": img.type, "Cache-Control": "public, max-age=2592000, immutable" },
  });
}
