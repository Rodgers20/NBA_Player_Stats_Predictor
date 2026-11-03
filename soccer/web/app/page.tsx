import { redirect } from "next/navigation";

export const dynamic = "force-dynamic";

/** The slate is the home page: games and their forecasts. */
export default async function Home({ searchParams }: { searchParams: Promise<{ competition?: string }> }) {
  const { competition } = await searchParams;
  redirect(competition ? `/games?competition=${encodeURIComponent(competition)}` : "/games");
}
