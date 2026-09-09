"use client";

import {
  ActionBarPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import { ArrowDown, Copy, FileDown, RefreshCw, Send, Square, Stethoscope } from "lucide-react";
import * as React from "react";
import { RecommendationCard } from "@/components/RecommendationCard";
import { ReportDocument } from "@/components/ReportDocument";
import { Button } from "@/components/ui/primitives";
import { useLatestRecommendation } from "@/lib/runtime";
import { cn } from "@/lib/utils";

const SUGGESTIONS = [
  "I have had a persistent cough with mucus for two weeks",
  "Constant anxiety and I cannot sleep at night",
  "Itchy red rash on my arms that will not go away",
  "Frequent heartburn and bloating after meals",
];

/**
 * Assistant turn. The backend streams prose *and* a structured payload; when
 * the payload has arrived we render the rich card instead of the raw text,
 * because the card exposes the citations the text can only describe.
 */
function AssistantMessage() {
  const [showReport, setShowReport] = React.useState(false);
  // The structured payload arrives on the same stream as the prose. Subscribing
  // to the store keeps this in sync without threading state through the runtime
  // or depending on version-specific message-state hooks.
  const data = useLatestRecommendation();

  return (
    <MessagePrimitive.Root className="group mb-6 flex gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--accent-soft)] ring-1 ring-[var(--accent-border)]">
        <Stethoscope className="h-4 w-4 text-[var(--accent-hover)]" />
      </div>

      <div className="min-w-0 flex-1">
        {data ? (
          <RecommendationCard data={data} />
        ) : (
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] px-4 py-3 shadow-[var(--shadow-sm)]">
            <div className="whitespace-pre-wrap text-[14px] leading-relaxed text-[var(--text)]">
              <MessagePrimitive.Parts />
            </div>
          </div>
        )}

        <ActionBarPrimitive.Root
          hideWhenRunning
          autohide="not-last"
          className="mt-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100"
        >
          <ActionBarPrimitive.Copy asChild>
            <Button variant="ghost" size="icon" aria-label="Copy reply">
              <Copy className="h-3.5 w-3.5" />
            </Button>
          </ActionBarPrimitive.Copy>
          <ActionBarPrimitive.Reload asChild>
            <Button variant="ghost" size="icon" aria-label="Regenerate">
              <RefreshCw className="h-3.5 w-3.5" />
            </Button>
          </ActionBarPrimitive.Reload>
          {data && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowReport(true)}
              aria-label="Generate report document"
            >
              <FileDown className="h-3.5 w-3.5" />
              Report
            </Button>
          )}
        </ActionBarPrimitive.Root>

        {showReport && data && (
          <ReportDocument data={data} onClose={() => setShowReport(false)} />
        )}
      </div>
    </MessagePrimitive.Root>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="mb-6 flex justify-end">
      <div className="max-w-[80%] rounded-xl rounded-br-sm bg-[var(--accent)] px-4 py-2.5 text-[14px] leading-relaxed text-white shadow-[var(--shadow-sm)]">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}

function Welcome() {
  return (
    <ThreadPrimitive.Empty>
      <div className="flex flex-1 flex-col items-center justify-center px-4 py-12 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--accent-soft)] ring-1 ring-[var(--accent-border)]">
          <Stethoscope className="h-6 w-6 text-[var(--accent-hover)]" />
        </div>
        <h2 className="mt-4 text-xl font-semibold tracking-tight text-[var(--text)]">
          Describe the symptoms
        </h2>
        <p className="mt-2 max-w-md text-sm leading-relaxed text-[var(--text-muted)]">
          The engine identifies a likely condition, ranks medications associated with it in
          patient-reported data, and cites supporting evidence you can open and check.
        </p>

        <div className="mt-6 grid w-full max-w-2xl gap-2 sm:grid-cols-2">
          {SUGGESTIONS.map((s) => (
            <ThreadPrimitive.Suggestion
              key={s}
              prompt={s}
              method="replace"
              autoSend
              className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3.5 py-3 text-left text-[13px] leading-snug text-[var(--text-muted)] shadow-[var(--shadow-sm)] transition-all hover:border-[var(--accent-border)] hover:bg-[var(--accent-soft)] hover:text-[var(--text)]"
            >
              {s}
            </ThreadPrimitive.Suggestion>
          ))}
        </div>
      </div>
    </ThreadPrimitive.Empty>
  );
}

function Composer() {
  return (
    <ComposerPrimitive.Root className="flex items-end gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-2 shadow-[var(--shadow)] transition-colors focus-within:border-[var(--accent-border)]">
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        placeholder="Describe symptoms in your own words…"
        className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-2 py-2 text-[14px] leading-relaxed text-[var(--text)] outline-none placeholder:text-[var(--text-subtle)]"
      />
      <ThreadPrimitive.If running={false}>
        <ComposerPrimitive.Send asChild>
          <Button size="icon" className="h-9 w-9" aria-label="Send">
            <Send className="h-4 w-4" />
          </Button>
        </ComposerPrimitive.Send>
      </ThreadPrimitive.If>
      <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel asChild>
          <Button size="icon" variant="secondary" className="h-9 w-9" aria-label="Stop">
            <Square className="h-3.5 w-3.5" />
          </Button>
        </ComposerPrimitive.Cancel>
      </ThreadPrimitive.If>
    </ComposerPrimitive.Root>
  );
}

export function Thread() {
  return (
    <ThreadPrimitive.Root className="flex h-full flex-col bg-[var(--background)]">
      <ThreadPrimitive.Viewport
        autoScroll
        className="relative flex flex-1 flex-col overflow-y-auto px-4 pt-6"
      >
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col">
          <Welcome />
          <ThreadPrimitive.Messages
            components={{ UserMessage, AssistantMessage }}
          />
          <ThreadPrimitive.If running>
            <div className="mb-6 flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--accent-soft)] ring-1 ring-[var(--accent-border)]">
                <Stethoscope className="h-4 w-4 text-[var(--accent-hover)]" />
              </div>
              <div className="flex items-center gap-1.5 rounded-xl border border-[var(--border)] bg-[var(--surface)] px-4 py-3.5">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="typing-dot h-1.5 w-1.5 rounded-full bg-[var(--accent)]"
                    style={{ animationDelay: `${i * 0.16}s` }}
                  />
                ))}
              </div>
            </div>
          </ThreadPrimitive.If>
          <div className="min-h-6 flex-grow" />
        </div>

        <ThreadPrimitive.ScrollToBottom asChild>
          <Button
            variant="secondary"
            size="icon"
            aria-label="Scroll to bottom"
            className={cn(
              "sticky bottom-2 left-1/2 z-10 -translate-x-1/2 rounded-full shadow-[var(--shadow-lg)]",
              "disabled:invisible",
            )}
          >
            <ArrowDown className="h-4 w-4" />
          </Button>
        </ThreadPrimitive.ScrollToBottom>
      </ThreadPrimitive.Viewport>

      <div className="border-t border-[var(--border)] bg-[var(--background)] px-4 pb-4 pt-3">
        <div className="mx-auto w-full max-w-3xl">
          <Composer />
          <p className="mt-2 text-center text-[11px] text-[var(--text-subtle)]">
            Research and educational decision support only — not a prescription. Always consult a
            qualified clinician.
          </p>
        </div>
      </div>
    </ThreadPrimitive.Root>
  );
}
