import { redirect } from "next/navigation";

export const dynamic = "force-dynamic";

/** Predictions moved onto the Games page; keep old links working. */
export default async function PredictionsRedirect({ searchParams }: { searchParams: Promise<{ competition?: string }> }) {
  const { competition } = await searchParams;
  redirect(competition ? `/games?competition=${encodeURIComponent(competition)}` : "/games");
}
