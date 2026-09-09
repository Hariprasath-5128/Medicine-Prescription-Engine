"use client";

import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import { Thread } from "@/components/Thread";
import { prescriptionAdapter } from "@/lib/runtime";

export default function ChatPage() {
  // LocalRuntime keeps thread state in the browser and delegates each turn to
  // our adapter, which streams from the FastAPI backend.
  const runtime = useLocalRuntime(prescriptionAdapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex min-h-0 flex-1 flex-col">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
}
