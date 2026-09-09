"use client";

import {
  AlertTriangle,
  ChevronDown,
  ExternalLink,
  FileText,
  Pill,
  ShieldCheck,
  ShieldAlert,
  Stethoscope,
} from "lucide-react";
import * as React from "react";
import { Badge, Card, Meter } from "@/components/ui/primitives";
import { cn, percent, type Evidence, type Recommendation } from "@/lib/utils";

/**
 * One citation, collapsed by default. Expanding reveals the exact retrieved
 * chunk plus its provenance - the point of the whole evidence layer is that a
 * reader can check the source rather than trust the model.
 */
function EvidenceItem({ evidence, index }: { evidence: Evidence; index: number }) {
  const [open, setOpen] = React.useState(index === 0);
  const title = evidence.question_focus || evidence.document_id || "source";

  return (
    <li className="overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-[var(--surface-muted)]"
      >
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-[var(--accent-soft)] text-[11px] font-semibold text-[var(--accent-hover)]">
          {index}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium capitalize text-[var(--text)]">
            {title}
          </span>
          {evidence.section && (
            <span className="block truncate text-xs capitalize text-[var(--text-subtle)]">
              {evidence.section.toLowerCase()}
            </span>
          )}
        </span>
        <span className="hidden shrink-0 items-center gap-2 sm:flex">
          <span className="w-16">
            <Meter value={evidence.relevance} />
          </span>
          <span className="w-10 text-right text-xs tabular-nums text-[var(--text-subtle)]">
            {evidence.relevance.toFixed(2)}
          </span>
        </span>
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-[var(--text-subtle)] transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {open && (
        <div className="border-t border-[var(--border)] px-3 py-3">
          <blockquote className="border-l-2 border-[var(--accent-border)] pl-3 text-[13px] leading-relaxed text-[var(--text-muted)]">
            {evidence.text}
          </blockquote>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {evidence.umls_semantic_group && (
              <Badge tone="neutral">{evidence.umls_semantic_group}</Badge>
            )}
            <span className="font-mono text-[11px] text-[var(--text-subtle)]">
              {evidence.chunk_id}
            </span>
            {evidence.source_url && (
              <a
                href={evidence.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="ml-auto inline-flex items-center gap-1 text-xs font-medium text-[var(--accent)] hover:text-[var(--accent-hover)] hover:underline"
              >
                View original source
                <ExternalLink className="h-3 w-3" />
              </a>
            )}
          </div>
        </div>
      )}
    </li>
  );
}

export function RecommendationCard({ data }: { data: Recommendation }) {
  return (
    <div className="animate-fade-up space-y-3">
      {data.red_flag && (
        <Card className="border-[var(--danger-border)] bg-[var(--danger-soft)] p-4">
          <div className="flex gap-3">
            <AlertTriangle className="h-5 w-5 shrink-0 text-[var(--danger)]" />
            <div>
              <p className="text-sm font-semibold text-[var(--danger)]">
                Urgent — seek immediate medical attention
              </p>
              <p className="mt-1 text-[13px] leading-relaxed text-[var(--text-muted)]">
                Your description mentions{" "}
                <strong className="font-semibold">{data.red_flag_terms.join(", ")}</strong>. These
                can indicate a medical emergency. Contact emergency services or attend an emergency
                department now.
              </p>
            </div>
          </div>
        </Card>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        {/* Assessment */}
        <Card className="p-4">
          <div className="flex items-center gap-2 text-[var(--text-muted)]">
            <Stethoscope className="h-4 w-4" />
            <h3 className="text-xs font-semibold uppercase tracking-wide">Assessment</h3>
          </div>
          <p className="mt-3 text-lg font-semibold leading-tight text-[var(--text)]">
            {data.condition}
          </p>
          <div className="mt-2 flex items-center gap-2">
            <Meter value={data.condition_confidence} />
            <span className="shrink-0 text-xs tabular-nums text-[var(--text-muted)]">
              {percent(data.condition_confidence)}
            </span>
          </div>

          {data.alternatives.length > 0 && (
            <div className="mt-4 border-t border-[var(--border)] pt-3">
              <p className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-subtle)]">
                Also considered
              </p>
              <ul className="mt-2 space-y-1.5">
                {data.alternatives.map((alt) => (
                  <li key={alt.name} className="flex items-center justify-between gap-3 text-[13px]">
                    <span className="truncate text-[var(--text-muted)]">{alt.name}</span>
                    <span className="shrink-0 tabular-nums text-[var(--text-subtle)]">
                      {percent(alt.score)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Card>

        {/* Medications */}
        <Card className="p-4">
          <div className="flex items-center gap-2 text-[var(--text-muted)]">
            <Pill className="h-4 w-4" />
            <h3 className="text-xs font-semibold uppercase tracking-wide">
              Associated medications
            </h3>
          </div>
          <p className="mt-1 text-[11px] leading-relaxed text-[var(--text-subtle)]">
            Ranked by association in patient-reported data. Not a dosage or a prescription.
          </p>
          <ol className="mt-3 space-y-2.5">
            {data.drugs.map((drug, i) => (
              <li key={drug.name}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="truncate text-sm font-medium text-[var(--text)]">
                    <span className="mr-1.5 text-[var(--text-subtle)]">{i + 1}.</span>
                    {drug.name}
                  </span>
                  <span className="shrink-0 text-xs tabular-nums text-[var(--text-subtle)]">
                    {percent(drug.score)}
                  </span>
                </div>
                <div className="mt-1">
                  <Meter value={drug.score} tone={i === 0 ? "accent" : "muted"} />
                </div>
              </li>
            ))}
          </ol>
        </Card>
      </div>

      {/* Evidence */}
      <Card className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-[var(--text-muted)]">
            <FileText className="h-4 w-4" />
            <h3 className="text-xs font-semibold uppercase tracking-wide">Supporting evidence</h3>
          </div>
          {data.grounded ? (
            <Badge tone="success">
              <ShieldCheck className="h-3 w-3" />
              {data.evidence.length} source{data.evidence.length === 1 ? "" : "s"}
            </Badge>
          ) : (
            <Badge tone="warning">
              <ShieldAlert className="h-3 w-3" />
              Unverified
            </Badge>
          )}
        </div>

        {data.grounded ? (
          <ul className="mt-3 space-y-2">
            {data.evidence.map((ev, i) => (
              <EvidenceItem key={ev.chunk_id} evidence={ev} index={i + 1} />
            ))}
          </ul>
        ) : (
          <p className="mt-3 rounded-lg border border-[var(--warning-border)] bg-[var(--warning-soft)] px-3 py-2.5 text-[13px] leading-relaxed text-[var(--text-muted)]">
            No chunk in the indexed corpus passed the relevance threshold, so this suggestion is
            unverified. Treat it as a hypothesis only.
          </p>
        )}
      </Card>

      <p className="px-1 text-[11px] leading-relaxed text-[var(--text-subtle)]">{data.disclaimer}</p>
    </div>
  );
}
