import type { NextConfig } from "next";

// The deployed site lives under /soccer on the Court Vision domain. Set via the
// `build` script; plain `npm run dev` serves at the root of localhost.
// `npm run dev:base` mirrors the deployed prefix at localhost:3000/soccer.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || undefined;

const nextConfig: NextConfig = {
  basePath,
  // Support both documented loopback URLs when running next dev.
  allowedDevOrigins: ["127.0.0.1"],
  // The snapshot (JSON + images) is read with fs at request time, so ship it with every function.
  outputFileTracingIncludes: { "/**": ["./snapshot/**/*"] },
};
export default nextConfig;
