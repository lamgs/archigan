"use client";

export default function Error({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
      <div style={{ maxWidth: 420, textAlign: "center" }}>
        <h1 style={{ fontFamily: "Georgia, serif", fontWeight: 400 }}>Something went wrong</h1>
        <p style={{ color: "#706c65", lineHeight: 1.6 }}>Sift hit an unexpected error. Projects are stored in this browser and were not deleted — reload to pick up where you left off.</p>
        <p style={{ display: "flex", gap: 10, justifyContent: "center" }}>
          <button type="button" onClick={reset}>Try again</button>
          <button type="button" onClick={() => window.location.reload()}>Reload</button>
        </p>
      </div>
    </main>
  );
}
