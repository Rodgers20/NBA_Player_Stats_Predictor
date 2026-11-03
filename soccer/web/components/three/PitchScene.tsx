"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Suspense, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { Pitch } from "./Pitch";
import { cn } from "@/lib/format";

export type CameraPreset = "broadcast" | "tactical" | "goalline";

// Framing was tuned against the rendered output, not guessed: the pitch is
// 120x80, so a 42-degree camera needs roughly 120 units of standoff to hold the
// whole surface without clipping the touchlines.
const PRESETS: Record<CameraPreset, { position: [number, number, number]; label: string }> = {
  broadcast: { position: [0, 66, 116], label: "Broadcast" },
  tactical: { position: [0, 132, 0.01], label: "Tactical" },
  goalline: { position: [-92, 20, 0.01], label: "Goal line" },
};

/** True when the browser can actually give us a WebGL context. */
function useWebGLSupport() {
  const [supported, setSupported] = useState<boolean | null>(null);
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
    try {
      const canvas = document.createElement("canvas");
      const gl =
        canvas.getContext("webgl2") ?? canvas.getContext("webgl");
      setSupported(Boolean(gl));
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
    } catch {
      setSupported(false);
    }
    });
    return () => cancelAnimationFrame(frame);
  }, []);
  return supported;
}

function subscribeMotion(callback: () => void) {
  const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
  mq.addEventListener("change", callback);
  return () => mq.removeEventListener("change", callback);
}
function usePrefersReducedMotion() {
  return useSyncExternalStore(subscribeMotion, () => window.matchMedia("(prefers-reduced-motion: reduce)").matches, () => false);
}

export function PitchScene({
  children, height = 380, preset = "broadcast", fallback, className,
}: {
  children?: React.ReactNode;
  height?: number;
  preset?: CameraPreset;
  fallback?: React.ReactNode;
  className?: string;
}) {
  const webgl = useWebGLSupport();
  const reduced = usePrefersReducedMotion();
  const [active, setActive] = useState<CameraPreset>(preset);
  const cam = useMemo(() => PRESETS[active], [active]);

  if (webgl !== true) {
    return (
      <div
        className={cn("grid place-items-center rounded-xl border border-stroke bg-elevated/40", className)}
        style={{ height }}
      >
        {fallback ?? (
          <p className="px-6 text-center text-[12px] text-ink-muted">
            3D view unavailable — WebGL is disabled in this browser.
            The same data is shown in the 2D panels.
          </p>
        )}
      </div>
    );
  }

  return (
    <div className={cn("relative rounded-xl overflow-hidden border border-stroke", className)} style={{ height }}>
      <div className="absolute z-10 top-2 left-2 flex gap-1">
        {(Object.keys(PRESETS) as CameraPreset[]).map((k) => (
          <button
            key={k}
            onClick={() => setActive(k)}
            className={cn(
              "rounded-md px-2 py-1 text-[10px] font-medium transition-colors backdrop-blur",
              active === k
                ? "bg-pitch/20 text-pitch ring-1 ring-pitch/40"
                : "bg-void/60 text-ink-secondary hover:text-ink",
            )}
          >
            {PRESETS[k].label}
          </button>
        ))}
      </div>

      <Canvas
        shadows
        dpr={[1, 2]}
        // Render on demand rather than every frame: an idle tactical view should
        // not burn GPU. OrbitControls invalidates as the user moves.
        frameloop="demand"
        gl={{ antialias: true, powerPreference: "high-performance" }}
      >
        <color attach="background" args={["#07090C"]} />
        <fog attach="fog" args={["#07090C", 120, 320]} />

        <PerspectiveCamera makeDefault position={cam.position} fov={42} near={0.5} far={800} />
        <OrbitControls
          makeDefault
          enablePan={false}
          enableDamping={!reduced}
          dampingFactor={0.06}
          minDistance={40}
          maxDistance={230}
          maxPolarAngle={Math.PI / 2.08}
          autoRotate={false}
          target={[0, 0, 0]}
        />

        {/* Floodlight rig: four corner keys plus soft ambient fill. */}
        <ambientLight intensity={0.55} />
        <hemisphereLight args={["#9DC9FF", "#0B1A12", 0.5]} />
        {([[-70, 60, -50], [70, 60, -50], [-70, 60, 50], [70, 60, 50]] as const).map(
          (p, i) => (
            <directionalLight
              key={i}
              position={p}
              intensity={0.55}
              castShadow={i === 0}
              shadow-mapSize={[1024, 1024]}
            />
          ),
        )}

        <Suspense fallback={null}>
          <Pitch />
          {children}
        </Suspense>
      </Canvas>
    </div>
  );
}
