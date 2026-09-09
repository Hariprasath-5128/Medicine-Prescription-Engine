"use client";

import {
  ExternalLink,
  Highlighter,
  Loader2,
  Quote,
  Search,
  SlidersHorizontal,
  TriangleAlert,
} from "lucide-react";
import * as React from "react";
import { Badge, Button, Card, Meter } from "@/components/ui/primitives";
import { HighlightLegend, HighlightedText, queryTerms } from "@/lib/highlight";
import { useSession } from "@/lib/store";
import { API_BASE, cn, type Evidence } from "@/lib/utils";

interface SearchResult {
  query: string;
  grounded: boolean;
  filtered_all: boolean;
  evidence: Evidence[];
}

export default function EvidencePage() {
  const session = useSession();
  const [query, setQuery] = React.useState("");
  const [topK, setTopK] = React.useState(6);
  const [maxDistance, setMaxDistance] = React.useState(0.55);
  const [highlight, setHighlight] = React.useState(true);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<SearchResult | null>(null);

  // Terms come from the query plus the last recommendation, so a chunk retrieved
  // for a prescription still lights up the condition and drug names.
  const terms = React.useMemo(() => {
    const rec = session.latest;
    return queryTerms(
      result?.query ?? query,
      rec?.condition ?? "",
      rec?.drugs.map((d) => d.name).join(" ") ?? "",
    );
  }, [result?.query, query, session.latest]);

  const search = React.useCallback(
    async (q: string) => {
      const text = q.trim();
      if (text.length < 3) return;
      setBusy(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/evidence`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: text, top_k: topK, max_distance: maxDistance }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.detail ?? `search failed (${res.status})`);
        setResult(json as SearchResult);
      } catch (e) {
        setError(e instanceof Error ? e.message : "could not reach the API");
      } finally {
        setBusy(false);
      }
    },
    [topK, maxDistance],
  );

  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-4 py-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-[var(--text)]">
          Clinical evidence & citations
        </h1>
        <p className="mt-1 text-sm text-[var(--text-muted)]">
          Search the indexed corpus directly. Every result shows the exact retrieved chunk with
          its provenance, and highlights dosage, risk and caution language.
        </p>
      </div>

      <Card className="p-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void search(query);
          }}
          className="flex gap-2"
        >
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-subtle)]" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. what causes keratoderma with woolly hair"
              className="h-10 w-full rounded-lg border border-[var(--border)] bg-[var(--surface)] pl-9 pr-3 text-[14px] text-[var(--text)] outline-none transition-colors placeholder:text-[var(--text-subtle)] focus:border-[var(--accent-border)]"
            />
          </div>
          <Button type="submit" disabled={busy || query.trim().length < 3}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            Search
          </Button>
        </form>

        {session.latest && (
          <button
            onClick={() => {
              const q = `${session.latest!.condition} treatment and management`;
              setQuery(q);
              void search(q);
            }}
            className="mt-2 inline-flex items-center gap-1.5 text-xs text-[var(--accent)] hover:underline"
          >
            <Quote className="h-3 w-3" />
            Search evidence for the last assessment: {session.latest.condition}
          </button>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-[var(--border)] pt-3">
          <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
            <SlidersHorizontal className="h-3.5 w-3.5" />
            <label htmlFor="topk">Results</label>
            <input
              id="topk"
              type="range"
              min={1}
              max={15}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-24 accent-[var(--accent)]"
            />
            <span className="w-5 tabular-nums">{topK}</span>
          </div>

          <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
            <label htmlFor="dist">Max distance</label>
            <input
              id="dist"
              type="range"
              min={10}
              max={100}
              value={maxDistance * 100}
              onChange={(e) => setMaxDistance(Number(e.target.value) / 100)}
              className="w-24 accent-[var(--accent)]"
            />
            <span className="w-8 tabular-nums">{maxDistance.toFixed(2)}</span>
            <span className="text-[var(--text-subtle)]">(lower is stricter)</span>
          </div>

          <button
            onClick={() => setHighlight((v) => !v)}
            aria-pressed={highlight}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition-colors",
              highlight
                ? "border-[var(--accent-border)] bg-[var(--accent-soft)] text-[var(--accent-hover)]"
                : "border-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--surface-muted)]",
            )}
          >
            <Highlighter className="h-3 w-3" />
            Highlighting {highlight ? "on" : "off"}
          </button>

          {highlight && <HighlightLegend />}
        </div>
      </Card>

      {error && (
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-soft)] px-3 py-2.5">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-[var(--danger)]" />
          <p className="text-[13px] text-[var(--text)]">{error}</p>
        </div>
      )}

      {result && (
        <section className="mt-6">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
              {result.evidence.length} result{result.evidence.length === 1 ? "" : "s"}
            </h2>
            {result.filtered_all && (
              <Badge tone="warning">
                <TriangleAlert className="h-3 w-3" />
                All neighbours exceeded the threshold
              </Badge>
            )}
          </div>

          {result.evidence.length === 0 ? (
            <Card className="p-6 text-center">
              <p className="text-sm text-[var(--text-muted)]">
                Nothing passed the relevance threshold. Loosen &ldquo;max distance&rdquo; to
                inspect the nearest neighbours anyway.
              </p>
            </Card>
          ) : (
            <ol className="space-y-3">
              {result.evidence.map((ev, i) => (
                <Card key={ev.chunk_id} className="overflow-hidden">
                  <div className="flex items-start gap-3 border-b border-[var(--border)] bg-[var(--surface-muted)] px-4 py-2.5">
                    <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-[var(--accent-soft)] text-[11px] font-semibold text-[var(--accent-hover)]">
                      {i + 1}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold capitalize text-[var(--text)]">
                        {ev.question_focus || ev.document_id}
                      </p>
                      {ev.section && (
                        <p className="truncate text-xs capitalize text-[var(--text-subtle)]">
                          {ev.section.toLowerCase()}
                        </p>
                      )}
                    </div>
                    <div className="hidden w-28 shrink-0 sm:block">
                      <Meter value={ev.relevance} />
                      <p className="mt-1 text-right text-[11px] tabular-nums text-[var(--text-subtle)]">
                        relevance {ev.relevance}
                      </p>
                    </div>
                  </div>

                  <div className="px-4 py-3">
                    <p className="text-[13px] leading-relaxed text-[var(--text-muted)]">
                      <HighlightedText text={ev.text} terms={terms} enabled={highlight} />
                    </p>

                    <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-[var(--border)] pt-2.5">
                      {ev.umls_semantic_group && (
                        <Badge tone="neutral">{ev.umls_semantic_group}</Badge>
                      )}
                      <span className="font-mono text-[11px] text-[var(--text-subtle)]">
                        {ev.chunk_id}
                      </span>
                      {ev.source_url && (
                        <a
                          href={ev.source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ml-auto inline-flex items-center gap-1 text-xs font-medium text-[var(--accent)] hover:underline"
                        >
                          View original source
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
            </ol>
          )}
        </section>
      )}

      {!result && !busy && (
        <Card className="mt-6 p-8 text-center">
          <Search className="mx-auto h-6 w-6 text-[var(--text-subtle)]" />
          <p className="mt-2 text-sm text-[var(--text-muted)]">
            Search the corpus to inspect the exact text behind a citation.
          </p>
        </Card>
      )}
    </div>
  );
}
