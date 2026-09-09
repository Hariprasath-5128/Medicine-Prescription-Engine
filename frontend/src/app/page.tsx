"use client";

import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import { Header } from "@/components/Header";
import { Thread } from "@/components/Thread";
import { prescriptionAdapter } from "@/lib/runtime";

export default function Home() {
  // LocalRuntime keeps thread state in the browser and delegates each turn to
  // our adapter, which streams from the FastAPI backend.
  const runtime = useLocalRuntime(prescriptionAdapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-dvh flex-col">
        <Header />
        <main className="min-h-0 flex-1">
          <Thread />
        </main>
      </div>
    </AssistantRuntimeProvider>
  );
}
