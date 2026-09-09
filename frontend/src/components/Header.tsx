"use client";

import { Activity, Database, ExternalLink, TriangleAlert } from "lucide-react";
import * as React from "react";
import { Badge } from "@/components/ui/primitives";
import { API_BASE, type HealthStatus } from "@/lib/utils";

/**
 * Header with a live readiness indicator.
 *
 * The backend genuinely runs in a degraded mode - it starts before the
 * recommender is trained and the UI falls back to evidence search - so showing
 * real component status beats a decorative "online" dot.
 */
export function Header() {
  const [health, setHealth] = React.useState<HealthStatus | null>(null);
  const [failed, setFailed] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
        if (!res.ok) throw new Error(String(res.status));
        const data = (await res.json()) as HealthStatus;
        if (!cancelled) {
          setHealth(data);
          setFailed(false);
        }
      } catch {
        if (!cancelled) setFailed(true);
      }
    };
    check();
    const id = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <header className="sticky top-0 z-20 border-b border-[var(--border)] bg-[var(--surface)]/85 backdrop-blur">
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center gap-3 px-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--accent)]">
            <Activity className="h-4 w-4 text-white" />
          </div>
          <div className="leading-tight">
            <p className="text-[15px] font-semibold tracking-tight text-[var(--text)]">
              Medicine Prescription Engine
            </p>
            <p className="hidden text-[11px] text-[var(--text-subtle)] sm:block">
              Evidence-grounded decision support
            </p>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          {failed ? (
            <Badge tone="danger">
              <TriangleAlert className="h-3 w-3" />
              API offline
            </Badge>
          ) : health ? (
            <>
              {typeof health.chunks === "number" && (
                <Badge tone="neutral" className="hidden sm:inline-flex">
                  <Database className="h-3 w-3" />
                  {health.chunks.toLocaleString()} chunks
                </Badge>
              )}
              <Badge tone={health.recommender_trained ? "success" : "warning"}>
                <span
                  className={
                    health.recommender_trained
                      ? "h-1.5 w-1.5 rounded-full bg-[var(--success)]"
                      : "h-1.5 w-1.5 rounded-full bg-[var(--warning)]"
                  }
                />
                {health.recommender_trained ? "Model ready" : "Model untrained"}
              </Badge>
            </>
          ) : (
            <Badge tone="neutral">Checking…</Badge>
          )}

          <a
            href="https://github.com/Hariprasath-5128/Medicine-Prescription-Engine"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Repository"
            className="flex h-8 w-8 items-center justify-center rounded-lg text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--text)]"
          >
            <ExternalLink className="h-4 w-4" />
          </a>
        </div>
      </div>
    </header>
  );
}
