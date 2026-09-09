import type { ChatModelAdapter } from "@assistant-ui/react";
import { useSyncExternalStore } from "react";
import { pushRecommendation } from "./store";
import { API_BASE, type Recommendation } from "./utils";

/**
 * Structured payloads keyed by the assistant text they accompany.
 *
 * assistant-ui's message parts carry text, so the rich recommendation object
 * travels alongside rather than inside the thread. Keying on the user's query
 * lets the message renderer pair them back up without threading extra state
 * through the runtime.
 */
export const recommendationStore = new Map<string, Recommendation>();

/** Subscribers notified whenever a structured payload lands. */
const listeners = new Set<() => void>();
let latest: Recommendation | undefined;

function publish(rec: Recommendation) {
  latest = rec;
  listeners.forEach((fn) => fn());
}

/**
 * The rendered recommendation for the current turn.
 *
 * useSyncExternalStore keeps React in step with the plain Map the streaming
 * adapter writes to, so the card appears the moment the payload arrives
 * without depending on version-specific message-state APIs.
 */
export function useLatestRecommendation(): Recommendation | undefined {
  return useSyncExternalStore(
    (onChange) => {
      listeners.add(onChange);
      return () => listeners.delete(onChange);
    },
    () => latest,
    () => undefined, // server render has no payload yet
  );
}

export const prescriptionAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    const query =
      lastUser?.content
        .filter((c): c is { type: "text"; text: string } => c.type === "text")
        .map((c) => c.text)
        .join(" ")
        .trim() ?? "";

    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: abortSignal,
      body: JSON.stringify({
        messages: messages.map((m) => ({
          role: m.role,
          content: m.content
            .filter((c): c is { type: "text"; text: string } => c.type === "text")
            .map((c) => c.text)
            .join(" "),
        })),
      }),
    });

    if (!response.ok || !response.body) {
      const detail = await response.text().catch(() => response.statusText);
      throw new Error(`Backend returned ${response.status}: ${detail}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // The backend emits newline-delimited JSON; the final element is kept in
      // the buffer because it may be a partial line.
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.trim()) continue;
        let frame: { type: string; text?: string; payload?: Recommendation };
        try {
          frame = JSON.parse(line);
        } catch {
          continue; // skip a malformed frame rather than killing the stream
        }

        if (frame.type === "text" && frame.text) {
          text += frame.text;
          yield { content: [{ type: "text", text }] };
        } else if (frame.type === "data" && frame.payload) {
          recommendationStore.set(query, frame.payload);
          publish(frame.payload);
          // Share with the other pages so Prescription can report on it and
          // Evidence can highlight its condition and drug names.
          pushRecommendation(frame.payload);
          // Re-yield so the renderer re-runs now that the payload has landed.
          yield { content: [{ type: "text", text }] };
        }
      }
    }
  },
};
