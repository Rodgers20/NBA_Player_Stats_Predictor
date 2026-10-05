const IS_EXPORT = process.env.NEXT_EXPORT === "1";
const BACKEND   = process.env.API_BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  env: { NEXT_PUBLIC_DATA_MODE: process.env.NEXT_PUBLIC_DATA_MODE || (IS_EXPORT ? "static" : "api") },
  ...(IS_EXPORT ? { output: "export" } : {}),

  ...(!IS_EXPORT ? {
    async rewrites() {
      return [{ source: "/api/:path*", destination: `${BACKEND}/api/:path*` }];
    },
  } : {}),
};

export default nextConfig;
