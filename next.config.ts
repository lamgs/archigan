import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  experimental: {
    // The managed test environment cannot capture output from Node child
    // processes, which the default CLI checker needs for `tsc --showConfig`.
    // TypeScript 6 still exposes the compiler API, so use Next's supported
    // in-process checker and keep production builds fully type-checked.
    useTypeScriptCli: false,
  },
};

export default nextConfig;

