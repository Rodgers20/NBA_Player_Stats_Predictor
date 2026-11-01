import { Suspense } from "react";
import { CbbGamePage } from "@/components/cbb/CbbGamePage";

export const metadata = { title: "Game | College Hoops" };

export default function CbbGameRoute() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading game…</p>}><CbbGamePage /></Suspense>;
}
