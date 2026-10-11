import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// Dev server only. In production nginx does the same job (see nginx.conf.template): /api/* is
// forwarded to the backend with the X-API-Key header added server-side, so the key never ships
// in the browser bundle.
export default defineConfig(({ mode }) => {
  // Reads the repo-level .env (API_KEY is NOT VITE_-prefixed, so it is never exposed to client code).
  const env = loadEnv(mode, "..", "");
  const target = env.DEV_API_URL || `http://localhost:${env.API_HOST_PORT || "8000"}`;
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target,
          changeOrigin: true,
          rewrite: (path: string) => path.replace(/^\/api/, ""),
          headers: { "X-API-Key": env.API_KEY ?? "" },
        },
      },
    },
  };
});
