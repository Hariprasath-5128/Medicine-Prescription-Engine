"use client";

import { Activity, Database, ExternalLink, FileText, MessagesSquare, Stethoscope, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import * as React from "react";
import { Badge } from "@/components/ui/primitives";
import { API_BASE, cn, type HealthStatus } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Chat", icon: MessagesSquare },
  { href: "/prescription", label: "Prescription", icon: Stethoscope },
  { href: "/evidence", label: "Evidence", icon: FileText },
] as const;

/**
 * Header with navigation and a live readiness indicator.
 *
 * The backend genuinely runs in a degraded mode - it starts before the
 * recommender is trained, and the evidence pages work without it - so showing
 * real component status beats a decorative "online" dot.
 */
export function Header() {
  const pathname = usePathname();
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
    <header className="sticky top-0 z-30 border-b border-[var(--border)] bg-[var(--surface)]/85 backdrop-blur print:hidden">
      <div className="mx-auto flex h-14 w-full max-w-7xl items-center gap-4 px-4">
        <Link href="/" className="flex shrink-0 items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--accent)]">
            <Activity className="h-4 w-4 text-white" />
          </div>
          <div className="hidden leading-tight sm:block">
            <p className="text-[15px] font-semibold tracking-tight text-[var(--text)]">
              Prescription Engine
            </p>
          </div>
        </Link>

        <nav className="flex items-center gap-1 rounded-lg bg-[var(--surface-muted)] p-1">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[13px] font-medium transition-colors",
                  active
                    ? "bg-[var(--surface)] text-[var(--text)] shadow-[var(--shadow-sm)]"
                    : "text-[var(--text-muted)] hover:text-[var(--text)]",
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          {failed ? (
            <Badge tone="danger">
              <TriangleAlert className="h-3 w-3" />
              API offline
            </Badge>
          ) : health ? (
            <>
              {typeof health.chunks === "number" && (
                <Badge tone="neutral" className="hidden md:inline-flex">
                  <Database className="h-3 w-3" />
                  {health.chunks.toLocaleString()} chunks
                </Badge>
              )}
              <Badge tone={health.recommender_trained ? "success" : "warning"}>
                <span
                  className={cn(
                    "h-1.5 w-1.5 rounded-full",
                    health.recommender_trained
                      ? "bg-[var(--success)]"
                      : "bg-[var(--warning)]",
                  )}
                />
                <span className="hidden sm:inline">
                  {health.recommender_trained ? "Model ready" : "Model untrained"}
                </span>
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
            className="hidden h-8 w-8 items-center justify-center rounded-lg text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--text)] sm:flex"
          >
            <ExternalLink className="h-4 w-4" />
          </a>
        </div>
      </div>
    </header>
  );
}
