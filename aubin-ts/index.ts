// AUBIN TypeScript / Node / browser client — talks to `aubin serve` (Laya/Jev-compatible /v1/systemone).
//   import { Aubin } from "./index";
//   const ai = new Aubin("http://127.0.0.1:8009");
//   const r = await ai.predict("billed twice", { dept: { type: "choice", instructions: "Which team?", criteria: { billing: "...", tech: "..." } } });

export type Question =
  | { type: "choice"; instructions: string; criteria: Record<string, string> }
  | { type: "score"; instructions: string; criteria: string[] }
  | { type: "noul"; instructions: string; criteria?: { true?: string; false?: string } };

export interface Answer {
  choice?: string | null; score?: number | null; noul?: number; label?: string;
  confidence: number | null; probabilities?: Record<string, number>; tier?: string; abstained?: boolean;
}
export interface Prediction { answers: Record<string, Answer>; routing: { per_question: Record<string, string>; tier_ms: Record<string, number>; total_ms: number } }

export class Aubin {
  constructor(private baseUrl = "http://127.0.0.1:8009", private apiKey?: string) {}

  private async post<T>(path: string, body: unknown): Promise<T> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    const res = await fetch(this.baseUrl.replace(/\/$/, "") + path, { method: "POST", headers, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(`AUBIN ${res.status}: ${await res.text()}`);
    return res.json() as Promise<T>;
  }

  predict(state: string | object, questions: Record<string, Question>, minConfidence?: number) {
    return this.post<Prediction>("/v1/systemone", { state, questions, min_confidence: minConfidence });
  }
  batch(items: { state: string | object; questions: Record<string, Question> }[]) {
    return this.post<{ results: Prediction[] }>("/v1/systemone/batch", { items });
  }
  learn(state: string | object, questions: Record<string, Question>, answers: Record<string, string>) {
    return this.post<{ learned_ms: number }>("/learn", { state, questions, answers });
  }
}
