const IS_EXPORT = process.env.NEXT_EXPORT === "1";
const BACKEND   = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  ...(IS_EXPORT ? { output: "export" } : {}),

  async rewrites() {
    if (IS_EXPORT) return [];
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
    ];
  },
};

export default nextConfig;
