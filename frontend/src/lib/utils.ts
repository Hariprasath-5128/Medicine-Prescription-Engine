import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

/** One retrieved chunk, mirroring `Evidence.to_dict()` in src/rag/retriever.py. */
export interface Evidence {
  chunk_id: string;
  text: string;
  distance: number;
  relevance: number;
  document_id: string;
  section: string;
  question_focus: string;
  umls_semantic_group: string;
  source_url: string;
}

/** Mirrors `Recommendation.to_dict()` in src/rag/pipeline.py. */
export interface Recommendation {
  query: string;
  condition: string;
  condition_confidence: number;
  drugs: { name: string; score: number }[];
  alternatives: { name: string; score: number }[];
  evidence: Evidence[];
  grounded: boolean;
  red_flag: boolean;
  red_flag_terms: string[];
  disclaimer: string;
}

export interface HealthStatus {
  status: string;
  evidence_db: boolean;
  recommender_trained: boolean;
  chunks?: number;
  error?: string;
}

export const percent = (n: number) => `${(n * 100).toFixed(1)}%`;
