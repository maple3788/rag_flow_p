"use client";

import { FormEvent, useEffect, useState } from "react";

import {
  getEvaluationHistory,
  getEvaluationSummary,
  listDatasets,
  streamChat,
  type ChatResponse,
  type Dataset,
  type EvaluationHistoryItem,
  type EvaluationScores,
  type EvaluationSummaryPoint,
  type LlmModel,
} from "@/lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  rewrittenQuery?: string | null;
  sources?: ChatResponse["sources"];
  evaluation?: EvaluationScores;
};

export default function ChatPage() {
  const [sessionScope, setSessionScope] = useState<"current" | "all">("current");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [model, setModel] = useState<LlmModel>("qwen3:8b");
  const [enableQueryRewrite, setEnableQueryRewrite] = useState(false);
  const [enableRerank, setEnableRerank] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [history, setHistory] = useState<EvaluationHistoryItem[]>([]);
  const [summary, setSummary] = useState<EvaluationSummaryPoint[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("chat_conversation_id");
    if (saved) {
      setConversationId(saved);
    }
  }, []);

  useEffect(() => {
    void loadDatasets();
  }, []);

  useEffect(() => {
    void loadHistory();
    void loadSummary();
  }, [conversationId, sessionScope]);

  async function loadHistory() {
    setHistoryLoading(true);
    try {
      const filterConversationId =
        sessionScope === "current" ? conversationId : undefined;
      const items = await getEvaluationHistory(40, filterConversationId);
      setHistory(items);
    } catch {
      // Keep chat usable even if history fetch fails.
    } finally {
      setHistoryLoading(false);
    }
  }

  async function loadSummary() {
    try {
      const filterConversationId =
        sessionScope === "current" ? conversationId : undefined;
      const points = await getEvaluationSummary(200, filterConversationId);
      setSummary(points);
    } catch {
      // Keep chat usable even if summary fetch fails.
    }
  }

  async function loadDatasets() {
    try {
      const items = await listDatasets();
      setDatasets(items);
      if (items.length > 0) {
        setSelectedDatasetId(String(items[0].id));
      }
    } catch {
      // Keep chat usable even if dataset fetch fails.
    }
  }

  function startNewSession() {
    const nextId = crypto.randomUUID().replace(/-/g, "");
    localStorage.setItem("chat_conversation_id", nextId);
    setConversationId(nextId);
    setMessages([]);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || loading) return;

    setError("");
    setLoading(true);
    const userId = `user-${Date.now()}`;
    const assistantId = `assistant-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: trimmed },
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setQuery("");

    try {
      await streamChat(
        trimmed,
        undefined,
        model,
        conversationId,
        {
          enableQueryRewrite,
          enableRerank,
          datasetId: selectedDatasetId ? Number(selectedDatasetId) : undefined,
        },
        {
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? { ...message, content: `${message.content}${token}` }
                : message
            )
          );
        },
        onDone: (result) => {
          if (!conversationId && result.conversation_id) {
            localStorage.setItem("chat_conversation_id", result.conversation_id);
            setConversationId(result.conversation_id);
          }
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    rewrittenQuery: result.rewritten_query,
                    content: result.answer,
                    sources: result.sources,
                    evaluation: result.evaluation,
                  }
                : message
            )
          );
        },
        }
      );
      await loadHistory();
      await loadSummary();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="chat-wrap">
      <div className="history-header">
        <h2>Chat with Documents</h2>
        <button className="button secondary" onClick={startNewSession} disabled={loading}>
          New Session
        </button>
      </div>
      <p className="muted">Session ID: {conversationId ?? "auto-create on first response"}</p>
      <div className="history-header">
        <label className="muted" htmlFor="session-scope-select">
          Panel scope
        </label>
        <select
          id="session-scope-select"
          className="model-select"
          value={sessionScope}
          onChange={(event) => setSessionScope(event.target.value as "current" | "all")}
        >
          <option value="current">Current session</option>
          <option value="all">All sessions</option>
        </select>
      </div>
      <div className="chat-box">
        {messages.length === 0 && <p className="muted">Ask a question to start.</p>}
        {messages.map((message, idx) => (
          <article key={message.id ?? `${message.role}-${idx}`} className={`message ${message.role}`}>
            <p className="message-role">{message.role === "user" ? "You" : "Assistant"}</p>
            {message.rewrittenQuery && (
              <p className="muted">Rewritten query: {message.rewrittenQuery}</p>
            )}
            <p>{message.content || (message.role === "assistant" && loading ? "..." : "")}</p>
            {message.sources && message.sources.length > 0 && (
              <div className="sources">
                <p className="message-role">Sources</p>
                {message.sources.map((source) => (
                  <div key={source.chunk_id} className="source-item">
                    <p>
                      <strong>{source.filename}</strong> (chunk {source.chunk_id}, score{" "}
                      {source.score.toFixed(4)})
                    </p>
                    <p>{source.content}</p>
                  </div>
                ))}
              </div>
            )}
            {message.evaluation && (
              <div className="eval-panel">
                <p className="message-role">Evaluation</p>
                <ScoreRow label="Faithfulness" value={message.evaluation.faithfulness} />
                <ScoreRow label="Relevance" value={message.evaluation.relevance} />
                <ScoreRow
                  label="Context Precision"
                  value={message.evaluation.context_precision}
                />
                {message.evaluation.rationale && (
                  <p className="muted">Judge note: {message.evaluation.rationale}</p>
                )}
              </div>
            )}
          </article>
        ))}
      </div>

      <form className="chat-form" onSubmit={onSubmit}>
        <select
          className="model-select"
          value={selectedDatasetId}
          onChange={(event) => setSelectedDatasetId(event.target.value)}
          disabled={loading || datasets.length === 0}
        >
          {datasets.length === 0 && <option value="">No datasets</option>}
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.name}
            </option>
          ))}
        </select>
        <select
          className="model-select"
          value={model}
          onChange={(event) => setModel(event.target.value as LlmModel)}
          disabled={loading}
        >
          <option value="qwen3:8b">qwen3:8b</option>
          <option value="llama3.2:latest">llama3.2:latest</option>
        </select>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Ask a question about uploaded files..."
          disabled={loading}
        />
        <button className="button" type="submit" disabled={loading || !query.trim()}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
      {datasets.length === 0 && (
        <p className="muted">No datasets found. Create one in the Datasets page first.</p>
      )}
      <div className="history-scores">
        <label className="inspector-checkbox">
          <input
            type="checkbox"
            checked={enableQueryRewrite}
            onChange={(event) => setEnableQueryRewrite(event.target.checked)}
            disabled={loading}
          />
          Rewrite query before retrieval
        </label>
        <label className="inspector-checkbox">
          <input
            type="checkbox"
            checked={enableRerank}
            onChange={(event) => setEnableRerank(event.target.checked)}
            disabled={loading}
          />
          Rerank retrieved chunks
        </label>
      </div>

      {error && <p className="error">{error}</p>}

      <section className="history-panel">
        <div className="history-header">
          <h3>Average Evaluation Scores Over Time</h3>
          <span className="muted">
            Scope: {sessionScope === "current" ? "Current session" : "All sessions"}
          </span>
        </div>
        {summary.length === 0 && <p className="muted">No summary data yet.</p>}
        {summary.length > 0 && (
          <div className="trend-list">
            {summary.map((point) => (
              <article key={point.period} className="history-item">
                <p>
                  <strong>{point.period}</strong> ({point.count} run{point.count > 1 ? "s" : ""})
                </p>
                <MiniTrendRow label="F" value={point.faithfulness} />
                <MiniTrendRow label="R" value={point.relevance} />
                <MiniTrendRow label="CP" value={point.context_precision} />
              </article>
            ))}
          </div>
        )}
      </section>

      <section className="history-panel">
        <div className="history-header">
          <h3>Past Evaluations by Session</h3>
          <button
            className="button secondary"
            onClick={async () => {
              await loadHistory();
              await loadSummary();
            }}
            disabled={historyLoading}
          >
            {historyLoading ? "Loading..." : "Refresh"}
          </button>
        </div>
        {history.length === 0 && <p className="muted">No evaluations yet.</p>}
        {groupByConversation(history).map(([sessionId, items]) => (
          <div key={sessionId} className="session-group">
            <p className="message-role">Session: {sessionId}</p>
            {items.map((item) => (
              <article key={item.id} className="history-item">
                <p className="muted">{new Date(item.created_at).toLocaleString()}</p>
                <p>
                  <strong>Q:</strong> {item.query}
                </p>
                <div className="history-scores">
                  <span className="score-badge">F {item.scores.faithfulness.toFixed(2)}</span>
                  <span className="score-badge">R {item.scores.relevance.toFixed(2)}</span>
                  <span className="score-badge">
                    CP {item.scores.context_precision.toFixed(2)}
                  </span>
                </div>
              </article>
            ))}
          </div>
        ))}
      </section>
    </section>
  );
}

function groupByConversation(items: EvaluationHistoryItem[]): [string, EvaluationHistoryItem[]][] {
  const map = new Map<string, EvaluationHistoryItem[]>();
  for (const item of items) {
    const key = item.conversation_id || "unknown";
    if (!map.has(key)) map.set(key, []);
    map.get(key)?.push(item);
  }
  return Array.from(map.entries());
}

function ScoreRow({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="score-row">
      <span>{label}</span>
      <span className="score-badge">{value.toFixed(2)}</span>
      <div className="score-bar">
        <div className="score-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function MiniTrendRow({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="score-row mini">
      <span>{label}</span>
      <span className="score-badge">{value.toFixed(2)}</span>
      <div className="score-bar">
        <div className="score-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
