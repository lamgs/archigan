import { existsSync } from "node:fs";
import { defineConfig } from "@playwright/test";

// Browsers: use PLAYWRIGHT_CHROMIUM_EXECUTABLE if set, else the container's pre-installed Chromium, else Playwright's own.
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ?? (existsSync("/opt/pw-browsers/chromium") ? "/opt/pw-browsers/chromium" : undefined);
const PORT = Number(process.env.E2E_PORT ?? 3200);
// Smoke-test a deployed URL instead of a local server: E2E_BASE_URL=https://… (add VERCEL_AUTOMATION_BYPASS_SECRET for protected deployments).
const REMOTE = process.env.E2E_BASE_URL;

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: true,
  workers: Number(process.env.E2E_WORKERS ?? 2),
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: REMOTE ?? `http://localhost:${PORT}`,
    extraHTTPHeaders: process.env.VERCEL_AUTOMATION_BYPASS_SECRET ? { "x-vercel-protection-bypass": process.env.VERCEL_AUTOMATION_BYPASS_SECRET } : undefined,
    viewport: { width: 1600, height: 950 },
    acceptDownloads: true,
    trace: "off",
    launchOptions: { executablePath, args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"] },
  },
  // Locally, tests run against the PRODUCTION build (`npm run build` first; `npm run test:e2e` does both).
  webServer: REMOTE ? undefined : { command: `npm run start -- -p ${PORT}`, url: `http://localhost:${PORT}`, reuseExistingServer: true, timeout: 60_000 },
});
