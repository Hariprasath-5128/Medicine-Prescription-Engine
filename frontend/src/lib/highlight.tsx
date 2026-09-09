"use client";

import * as React from "react";

/**
 * Highlighting for retrieved chunks.
 *
 * Two kinds of emphasis, visually distinct because they mean different things:
 *
 *   query   terms the user actually asked about, or the predicted condition
 *           and drugs - "why was this chunk retrieved?"
 *   salient clinically load-bearing phrases (dosage, contraindication, risk)
 *           - "what in here matters even if you did not ask for it?"
 *
 * Salient terms are matched from a fixed vocabulary rather than inferred. A
 * model-free rule is predictable, and a wrong highlight in clinical text is
 * worse than no highlight.
 */
const SALIENT_PATTERNS: { label: string; re: RegExp }[] = [
  // Anything dose-shaped: numbers with units, or explicit dosing language.
  { label: "dosage", re: /\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?|iu)\b(?:\s?(?:per|\/)\s?(?:day|kg|dose|hour))?/gi },
  { label: "frequency", re: /\b(?:once|twice|three times)?\s?(?:daily|per day|a day|every \d+ hours?|weekly|monthly|as needed)\b/gi },
  { label: "risk", re: /\b(?:contraindicat\w*|not recommended|should not (?:be )?(?:used|take\w*)|avoid|do not (?:use|take)|black box|boxed warning)\b/gi },
  { label: "adverse", re: /\b(?:side effects?|adverse (?:effects?|reactions?)|toxicity|overdose|allerg\w+|anaphylax\w+)\b/gi },
  { label: "caution", re: /\b(?:pregnan\w+|breastfeed\w+|children under|elderly|renal impairment|hepatic impairment|kidney disease|liver disease)\b/gi },
  { label: "urgent", re: /\b(?:emergency|immediately|seek (?:medical|immediate)\w*|life-threatening|fatal|death)\b/gi },
];

const STOP = new Set([
  "the", "and", "for", "with", "that", "this", "have", "has", "was", "are", "you",
  "your", "from", "not", "but", "can", "may", "will", "them", "they", "their",
  "what", "when", "which", "who", "how", "its", "it", "a", "an", "of", "in", "on",
  "to", "is", "be", "or", "as", "at", "by", "my", "me", "i", "do", "does", "did",
  "been", "were", "would", "could", "should", "about", "into", "over", "after",
  "treatment", "treatments", "patient", "patients", "symptoms", "condition",
]);

/** Words worth matching on: 4+ chars, not stopwords. */
export function queryTerms(...sources: string[]): string[] {
  const seen = new Set<string>();
  for (const source of sources) {
    for (const raw of String(source).toLowerCase().split(/[^a-z0-9]+/)) {
      if (raw.length >= 4 && !STOP.has(raw)) seen.add(raw);
    }
  }
  return [...seen];
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

interface Span {
  start: number;
  end: number;
  kind: "query" | "salient";
  label?: string;
}

/** Non-overlapping spans, salient winning ties (it carries more meaning). */
function findSpans(text: string, terms: string[]): Span[] {
  const spans: Span[] = [];

  for (const { label, re } of SALIENT_PATTERNS) {
    const rx = new RegExp(re.source, re.flags);
    let m: RegExpExecArray | null;
    while ((m = rx.exec(text)) !== null) {
      if (!m[0].trim()) {
        rx.lastIndex++;
        continue;
      }
      spans.push({ start: m.index, end: m.index + m[0].length, kind: "salient", label });
    }
  }

  if (terms.length) {
    // Word-boundary match so "pain" does not light up inside "painting".
    const rx = new RegExp(`\\b(${terms.map(escapeRe).join("|")})\\w{0,3}\\b`, "gi");
    let m: RegExpExecArray | null;
    while ((m = rx.exec(text)) !== null) {
      spans.push({ start: m.index, end: m.index + m[0].length, kind: "query" });
    }
  }

  spans.sort((a, b) => a.start - b.start || (a.kind === "salient" ? -1 : 1));

  const merged: Span[] = [];
  let cursor = -1;
  for (const span of spans) {
    if (span.start >= cursor) {
      merged.push(span);
      cursor = span.end;
    }
  }
  return merged;
}

export function HighlightedText({
  text,
  terms,
  enabled = true,
}: {
  text: string;
  terms: string[];
  enabled?: boolean;
}) {
  const parts = React.useMemo(() => {
    if (!enabled) return null;
    const spans = findSpans(text, terms);
    if (!spans.length) return null;

    const nodes: React.ReactNode[] = [];
    let last = 0;
    spans.forEach((span, i) => {
      if (span.start > last) nodes.push(text.slice(last, span.start));
      const body = text.slice(span.start, span.end);
      nodes.push(
        span.kind === "salient" ? (
          <mark
            key={i}
            title={span.label}
            className="rounded bg-[var(--warning-soft)] px-0.5 font-medium text-[var(--warning)] ring-1 ring-inset ring-[var(--warning-border)]"
          >
            {body}
          </mark>
        ) : (
          <mark
            key={i}
            className="rounded bg-[var(--accent-soft)] px-0.5 font-medium text-[var(--accent-hover)]"
          >
            {body}
          </mark>
        ),
      );
      last = span.end;
    });
    if (last < text.length) nodes.push(text.slice(last));
    return nodes;
  }, [text, terms, enabled]);

  return <>{parts ?? text}</>;
}

export function HighlightLegend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-[11px] text-[var(--text-subtle)]">
      <span className="inline-flex items-center gap-1.5">
        <span className="h-2.5 w-4 rounded-sm bg-[var(--accent-soft)] ring-1 ring-[var(--accent-border)]" />
        query match
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="h-2.5 w-4 rounded-sm bg-[var(--warning-soft)] ring-1 ring-[var(--warning-border)]" />
        dosage · risk · caution
      </span>
    </div>
  );
}
