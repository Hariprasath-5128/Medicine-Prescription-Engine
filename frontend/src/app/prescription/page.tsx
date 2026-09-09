"use client";

import {
  FileUp,
  FileWarning,
  Loader2,
  Paperclip,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import * as React from "react";
import { RecommendationCard } from "@/components/RecommendationCard";
import { ReportDocument } from "@/components/ReportDocument";
import { Button, Card } from "@/components/ui/primitives";
import { pushRecommendation, useSession } from "@/lib/store";
import { API_BASE, cn, type Recommendation } from "@/lib/utils";

interface UploadInfo {
  filename: string;
  words: number;
}

export default function PrescriptionPage() {
  const session = useSession();
  const [text, setText] = React.useState("");
  const [upload, setUpload] = React.useState<UploadInfo | null>(null);
  const [busy, setBusy] = React.useState<"upload" | "generate" | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [dragging, setDragging] = React.useState(false);
  const [showReport, setShowReport] = React.useState(false);
  const [result, setResult] = React.useState<Recommendation | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  // Show the newest result from any page, so a chat answer can be reported here.
  const active = result ?? session.latest ?? null;

  const handleFile = React.useCallback(async (file: File) => {
    setError(null);
    setBusy("upload");
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? `upload failed (${res.status})`);
      setText(json.text);
      setUpload({ filename: json.filename, words: json.words });
    } catch (e) {
      setError(e instanceof Error ? e.message : "upload failed");
    } finally {
      setBusy(null);
    }
  }, []);

  const generate = React.useCallback(async () => {
    const query = text.trim();
    if (query.length < 3) {
      setError("Enter or upload at least a few words of description.");
      return;
    }
    setError(null);
    setBusy("generate");
    try {
      const res = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: query, top_k_drugs: 5 }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? `request failed (${res.status})`);
      setResult(json as Recommendation);
      pushRecommendation(json as Recommendation);
    } catch (e) {
      setError(e instanceof Error ? e.message : "could not reach the API");
    } finally {
      setBusy(null);
    }
  }, [text]);

  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-4 py-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-[var(--text)]">
          Generate a prescription
        </h1>
        <p className="mt-1 text-sm text-[var(--text-muted)]">
          Upload clinical notes or type a description, then generate an evidence-grounded
          recommendation and export it as a report.
        </p>
      </div>

      <Card className="p-4">
        {/* Upload */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) void handleFile(file);
          }}
          className={cn(
            "flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors",
            dragging
              ? "border-[var(--accent)] bg-[var(--accent-soft)]"
              : "border-[var(--border-strong)] bg-[var(--surface-muted)]",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.txt,.md,.csv,.json"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleFile(file);
              e.target.value = ""; // allow re-selecting the same file
            }}
          />
          {busy === "upload" ? (
            <Loader2 className="h-6 w-6 animate-spin text-[var(--accent)]" />
          ) : (
            <FileUp className="h-6 w-6 text-[var(--text-subtle)]" />
          )}
          <p className="mt-2 text-sm font-medium text-[var(--text)]">
            Drop patient notes here
          </p>
          <p className="mt-0.5 text-xs text-[var(--text-subtle)]">
            PDF or text · up to 8 MB · text is extracted in memory, never stored
          </p>
          <Button
            variant="secondary"
            size="sm"
            className="mt-3"
            disabled={busy !== null}
            onClick={() => inputRef.current?.click()}
          >
            <Paperclip className="h-3.5 w-3.5" />
            Choose file
          </Button>
        </div>

        {upload && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-[var(--accent-border)] bg-[var(--accent-soft)] px-3 py-2">
            <Paperclip className="h-3.5 w-3.5 shrink-0 text-[var(--accent-hover)]" />
            <span className="min-w-0 flex-1 truncate text-[13px] text-[var(--text)]">
              {upload.filename}
            </span>
            <span className="shrink-0 text-xs text-[var(--text-muted)]">
              {upload.words.toLocaleString()} words
            </span>
            <button
              onClick={() => {
                setUpload(null);
                setText("");
              }}
              aria-label="Remove file"
              className="shrink-0 rounded p-1 text-[var(--text-muted)] hover:bg-white/60"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {/* Editable text: extraction can be imperfect, so review before running. */}
        <div className="mt-4">
          <div className="mb-1.5 flex items-center justify-between">
            <label
              htmlFor="notes"
              className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]"
            >
              Clinical description
            </label>
            {text && (
              <button
                onClick={() => {
                  setText("");
                  setUpload(null);
                }}
                className="inline-flex items-center gap-1 text-xs text-[var(--text-subtle)] hover:text-[var(--text)]"
              >
                <Trash2 className="h-3 w-3" />
                Clear
              </button>
            )}
          </div>
          <textarea
            id="notes"
            rows={6}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. 34-year-old presenting with a persistent productive cough for two weeks, mild fever, no chest pain…"
            className="w-full resize-y rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2.5 text-[14px] leading-relaxed text-[var(--text)] outline-none transition-colors placeholder:text-[var(--text-subtle)] focus:border-[var(--accent-border)]"
          />
          <p className="mt-1 text-right text-[11px] tabular-nums text-[var(--text-subtle)]">
            {text.trim() ? text.trim().split(/\s+/).length.toLocaleString() : 0} words
          </p>
        </div>

        {error && (
          <div className="mt-3 flex items-start gap-2 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-soft)] px-3 py-2.5">
            <FileWarning className="mt-0.5 h-4 w-4 shrink-0 text-[var(--danger)]" />
            <p className="text-[13px] leading-relaxed text-[var(--text)]">{error}</p>
          </div>
        )}

        <div className="mt-4 flex flex-wrap gap-2">
          <Button onClick={generate} disabled={busy !== null || text.trim().length < 3}>
            {busy === "generate" ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            Generate prescription
          </Button>
          <Button
            variant="secondary"
            disabled={!active}
            onClick={() => setShowReport(true)}
          >
            <FileUp className="h-4 w-4 rotate-180" />
            Generate report
          </Button>
        </div>
      </Card>

      {active && (
        <section className="mt-6">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            Recommendation
          </h2>
          <RecommendationCard data={active} />
        </section>
      )}

      {showReport && active && (
        <ReportDocument data={active} onClose={() => setShowReport(false)} />
      )}
    </div>
  );
}
