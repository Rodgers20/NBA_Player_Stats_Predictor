import { Suspense } from "react";
import { CbbSlate } from "@/components/cbb/CbbSlate";

export const metadata = { title: "College Hoops | Court Vision" };

export default function CbbPage() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading games…</p>}><CbbSlate /></Suspense>;
}
