"use client";

import { useSyncExternalStore } from "react";
import type { Recommendation } from "./utils";

/**
 * Cross-page state.
 *
 * The three pages are separate routes but share one workflow: a recommendation
 * produced in Chat or Prescription is what the Report prints and what the
 * Evidence page inspects. Keeping it in a tiny external store (rather than a
 * context high in the tree) means any page can read the latest result without
 * the others having to re-fetch it.
 *
 * sessionStorage persists across navigation and reload, but deliberately not
 * across browser sessions - this is patient-shaped data and should not linger.
 */
const KEY = "mpe.session";

export interface SessionState {
  latest?: Recommendation;
  history: Recommendation[];
}

let state: SessionState = { history: [] };
let hydrated = false;
const listeners = new Set<() => void>();

function hydrate() {
  if (hydrated || typeof window === "undefined") return;
  hydrated = true;
  try {
    const raw = sessionStorage.getItem(KEY);
    if (raw) state = JSON.parse(raw) as SessionState;
  } catch {
    // Corrupt or unavailable storage: start clean rather than crash the page.
  }
}

function persist() {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // Quota or private mode - in-memory state still works for this session.
  }
}

function emit() {
  persist();
  listeners.forEach((fn) => fn());
}

export function pushRecommendation(rec: Recommendation) {
  hydrate();
  // Cap the history so a long session cannot exhaust sessionStorage.
  state = { latest: rec, history: [rec, ...state.history].slice(0, 20) };
  emit();
}

export function clearSession() {
  state = { history: [] };
  emit();
}

function subscribe(fn: () => void) {
  hydrate();
  listeners.add(fn);
  return () => listeners.delete(fn);
}

const SERVER_SNAPSHOT: SessionState = { history: [] };

export function useSession(): SessionState {
  return useSyncExternalStore(
    subscribe,
    () => {
      hydrate();
      return state;
    },
    () => SERVER_SNAPSHOT,
  );
}
