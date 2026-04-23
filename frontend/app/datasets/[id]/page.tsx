"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import {
  debugDatasetRetrieval,
  deleteDataset,
  deleteDatasetFile,
  getDataset,
  listDatasetChunks,
  listDatasetFiles,
  updateDataset,
  uploadDatasetFile,
  type Dataset,
  type DatasetChunk,
  type RetrievalDebugResponse,
  type DatasetFile,
} from "@/lib/api";

type TabId = "files" | "chunks" | "config" | "retrieval";

export default function DatasetDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const datasetId = useMemo(() => Number(params?.id), [params?.id]);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [files, setFiles] = useState<DatasetFile[]>([]);
  const [chunks, setChunks] = useState<DatasetChunk[]>([]);
  const [selectedFileId, setSelectedFileId] = useState<number | "">("");
  const [tab, setTab] = useState<TabId>("files");
  const [uploadFileValue, setUploadFileValue] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [isDeletingDataset, setIsDeletingDataset] = useState(false);
  const [deletingFileId, setDeletingFileId] = useState<number | null>(null);
  const [descriptionDraft, setDescriptionDraft] = useState("");
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [enableQueryRewrite, setEnableQueryRewrite] = useState(false);
  const [rerankEnabled, setRerankEnabled] = useState(true);
  const [rerankModel, setRerankModel] = useState("cross-encoder/ms-marco-MiniLM-L-6-v2");
  const [testQuery, setTestQuery] = useState("");
  const [testTopKBm25, setTestTopKBm25] = useState(10);
  const [testTopKDense, setTestTopKDense] = useState(10);
  const [testFinalTopK, setTestFinalTopK] = useState(5);
  const [testRewrite, setTestRewrite] = useState(true);
  const [isRunningTest, setIsRunningTest] = useState(false);
  const [testResult, setTestResult] = useState<RetrievalDebugResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!Number.isFinite(datasetId)) return;
    void loadAll();
  }, [datasetId]);

  async function loadAll() {
    setError("");
    try {
      const [datasetRes, filesRes] = await Promise.all([
        getDataset(datasetId),
        listDatasetFiles(datasetId),
      ]);
      setDataset(datasetRes);
      setDescriptionDraft(datasetRes.description || "");
      setChunkSize(_configNumber(datasetRes.config, "chunk_size", 500));
      setChunkOverlap(_configNumber(datasetRes.config, "chunk_overlap", 50));
      setEnableQueryRewrite(_configBool(datasetRes.config, "enable_query_rewrite", false));
      setRerankEnabled(_configBool(datasetRes.config, "rerank_enabled", true));
      setRerankModel(
        _configString(
          datasetRes.config,
          "rerank_model",
          "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
      );
      setTestTopKBm25(10);
      setTestTopKDense(10);
      setTestFinalTopK(5);
      setFiles(filesRes);
      setChunks(await listDatasetChunks(datasetId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dataset");
    }
  }

  async function onUpload(event: FormEvent) {
    event.preventDefault();
    if (!uploadFileValue) return;
    setIsUploading(true);
    setError("");
    try {
      await uploadDatasetFile(datasetId, uploadFileValue);
      setUploadFileValue(null);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  async function refreshChunks(fileId?: number) {
    setError("");
    try {
      setChunks(await listDatasetChunks(datasetId, fileId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load chunks");
    }
  }

  async function onSaveConfig(event: FormEvent) {
    event.preventDefault();
    setIsSavingConfig(true);
    setError("");
    try {
      const updated = await updateDataset(datasetId, {
        description: descriptionDraft,
        config: {
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
          enable_query_rewrite: enableQueryRewrite,
          rerank_enabled: rerankEnabled,
          rerank_model: rerankModel,
        },
      });
      setDataset(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save config");
    } finally {
      setIsSavingConfig(false);
    }
  }

  async function onRunRetrievalTest(event: FormEvent) {
    event.preventDefault();
    if (!testQuery.trim()) return;
    setIsRunningTest(true);
    setError("");
    try {
      const result = await debugDatasetRetrieval(datasetId, {
        query: testQuery.trim(),
        top_k_bm25: testTopKBm25,
        top_k_dense: testTopKDense,
        final_top_k: testFinalTopK,
        enable_query_rewrite: testRewrite,
      });
      setTestResult(result);
      setTab("retrieval");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run retrieval debug");
    } finally {
      setIsRunningTest(false);
    }
  }

  async function onDeleteDataset() {
    const confirmed = window.confirm("Delete this dataset and all its files/chunks?");
    if (!confirmed) return;
    setIsDeletingDataset(true);
    setError("");
    try {
      await deleteDataset(datasetId);
      router.push("/datasets");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete dataset");
      setIsDeletingDataset(false);
    }
  }

  async function onDeleteFile(fileId: number) {
    const confirmed = window.confirm("Delete this file and all chunks generated from it?");
    if (!confirmed) return;
    setDeletingFileId(fileId);
    setError("");
    try {
      await deleteDatasetFile(datasetId, fileId);
      if (selectedFileId === fileId) {
        setSelectedFileId("");
        await refreshChunks();
      } else {
        await refreshChunks(selectedFileId === "" ? undefined : selectedFileId);
      }
      setFiles((prev) => prev.filter((file) => file.id !== fileId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete file");
    } finally {
      setDeletingFileId(null);
    }
  }

  async function onInspectFileChunks(fileId: number) {
    setSelectedFileId(fileId);
    setTab("chunks");
    await refreshChunks(fileId);
  }

  return (
    <section className="card">
      <h2>Dataset Detail</h2>
      <p className="muted">
        {dataset ? `${dataset.name} (#${dataset.id})` : "Loading dataset..."}
      </p>
      <p className="muted">Files attached: {files.length}</p>
      <button className="button" onClick={() => void onDeleteDataset()} disabled={isDeletingDataset}>
        {isDeletingDataset ? "Deleting..." : "Delete Dataset"}
      </button>

      <div className="tabs">
        <button className={`button ${tab === "files" ? "" : "secondary"}`} onClick={() => setTab("files")}>
          Files
        </button>
        <button className={`button ${tab === "chunks" ? "" : "secondary"}`} onClick={() => setTab("chunks")}>
          Chunks Explorer
        </button>
        <button className={`button ${tab === "config" ? "" : "secondary"}`} onClick={() => setTab("config")}>
          Config Editor
        </button>
        <button
          className={`button ${tab === "retrieval" ? "" : "secondary"}`}
          onClick={() => setTab("retrieval")}
        >
          Retrieval Test
        </button>
      </div>

      {tab === "files" && (
        <section className="history-panel">
          <form className="dataset-create-form" onSubmit={onUpload}>
            <input
              type="file"
              onChange={(event) => setUploadFileValue(event.target.files?.[0] ?? null)}
              disabled={isUploading}
            />
            <button className="button" type="submit" disabled={isUploading || !uploadFileValue}>
              {isUploading ? "Uploading..." : "Upload File"}
            </button>
          </form>
          <div className="trend-list">
            {files.map((file) => (
              <article className="history-item" key={file.id}>
                <p>
                  <strong>{file.filename}</strong> (file #{file.id})
                </p>
                <p className="muted">Text length: {file.raw_text.length}</p>
                <div className="actions">
                  <button
                    className="button secondary"
                    onClick={() => void onInspectFileChunks(file.id)}
                  >
                    Inspect Chunks
                  </button>
                  <button
                    className="button"
                    onClick={() => void onDeleteFile(file.id)}
                    disabled={deletingFileId === file.id}
                  >
                    {deletingFileId === file.id ? "Deleting..." : "Delete File"}
                  </button>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}

      {tab === "chunks" && (
        <section className="history-panel">
          <div className="history-header">
            <select
              className="model-select"
              value={selectedFileId}
              onChange={(event) => {
                const nextValue = event.target.value ? Number(event.target.value) : "";
                setSelectedFileId(nextValue);
                void refreshChunks(nextValue === "" ? undefined : nextValue);
              }}
            >
              <option value="">All files</option>
              {files.map((file) => (
                <option key={file.id} value={file.id}>
                  {file.filename} (#{file.id})
                </option>
              ))}
            </select>
          </div>
          <div className="trend-list">
            {chunks.map((chunk) => (
              <article key={chunk.id} className="history-item">
                <p>
                  <strong>Chunk #{chunk.id}</strong> from file #{chunk.file_id}
                </p>
                <p>{chunk.content}</p>
              </article>
            ))}
          </div>
        </section>
      )}

      {tab === "config" && (
        <section className="history-panel">
          <h3>Config Editor</h3>
          <form className="dataset-create-form" onSubmit={onSaveConfig}>
            <textarea
              className="json-box"
              value={descriptionDraft}
              onChange={(event) => setDescriptionDraft(event.target.value)}
              placeholder="Dataset description"
            />
            <label className="muted">
              Chunk size
              <input
                className="inspector-input"
                type="number"
                min={100}
                max={4000}
                value={chunkSize}
                onChange={(event) => setChunkSize(Number(event.target.value || 500))}
              />
            </label>
            <label className="muted">
              Chunk overlap
              <input
                className="inspector-input"
                type="number"
                min={0}
                max={1000}
                value={chunkOverlap}
                onChange={(event) => setChunkOverlap(Number(event.target.value || 50))}
              />
            </label>
            <p className="muted">Fusion method: rrf (fixed)</p>
            <label className="inspector-checkbox">
              <input
                type="checkbox"
                checked={enableQueryRewrite}
                onChange={(event) => setEnableQueryRewrite(event.target.checked)}
              />
              Enable query rewrite by default
            </label>
            <label className="inspector-checkbox">
              <input
                type="checkbox"
                checked={rerankEnabled}
                onChange={(event) => setRerankEnabled(event.target.checked)}
              />
              Enable rerank
            </label>
            <label className="muted">
              Rerank model
              <input
                className="inspector-input"
                value={rerankModel}
                onChange={(event) => setRerankModel(event.target.value)}
              />
            </label>
            <button className="button" type="submit" disabled={isSavingConfig}>
              {isSavingConfig ? "Saving..." : "Save Config"}
            </button>
          </form>
        </section>
      )}
      {tab === "retrieval" && (
        <section className="history-panel">
          <h3>Retrieval Testbench</h3>
          <form className="dataset-create-form" onSubmit={onRunRetrievalTest}>
            <textarea
              className="json-box"
              value={testQuery}
              onChange={(event) => setTestQuery(event.target.value)}
              placeholder="Enter a query to inspect sparse, dense, fusion, and rerank stages"
              disabled={isRunningTest}
            />
            <label className="muted">
              Top-k BM25
              <input
                className="inspector-input"
                type="number"
                min={1}
                max={200}
                value={testTopKBm25}
                onChange={(event) => setTestTopKBm25(Number(event.target.value || 10))}
                disabled={isRunningTest}
              />
            </label>
            <label className="muted">
              Top-k Dense
              <input
                className="inspector-input"
                type="number"
                min={1}
                max={200}
                value={testTopKDense}
                onChange={(event) => setTestTopKDense(Number(event.target.value || 10))}
                disabled={isRunningTest}
              />
            </label>
            <label className="muted">
              Final top-k
              <input
                className="inspector-input"
                type="number"
                min={1}
                max={50}
                value={testFinalTopK}
                onChange={(event) => setTestFinalTopK(Number(event.target.value || 5))}
                disabled={isRunningTest}
              />
            </label>
            <p className="muted">Fusion method: rrf (fixed)</p>
            <label className="inspector-checkbox">
              <input
                type="checkbox"
                checked={testRewrite}
                onChange={(event) => setTestRewrite(event.target.checked)}
                disabled={isRunningTest}
              />
              Apply rewrite
            </label>
            <button className="button" type="submit" disabled={isRunningTest || !testQuery.trim()}>
              {isRunningTest ? "Running..." : "Run Retrieval Debug"}
            </button>
          </form>

          {!testResult && <p className="muted">Run a query to inspect each retrieval stage.</p>}
          {testResult && (
            <div className="trend-list">
              <article className="history-item">
                <p><strong>Original query:</strong> {testResult.original_query}</p>
                {testResult.rewritten_query && (
                  <p><strong>Rewritten query:</strong> {testResult.rewritten_query}</p>
                )}
                <p><strong>Used query:</strong> {testResult.used_query}</p>
              </article>
              <DebugStage title="Sparse (BM25)" hits={testResult.bm25_hits} />
              <DebugStage title="Dense (FAISS)" hits={testResult.dense_hits} />
              <DebugStage title="Fusion (RRF)" hits={testResult.fused_hits} />
              <DebugStage title="Rerank" hits={testResult.reranked_hits} />
              <article className="history-item">
                <p><strong>Final selected chunks ({testResult.final_sources.length})</strong></p>
                {testResult.final_sources.map((source) => (
                  <div key={source.chunk_id} className="source-item">
                    <p>
                      <strong>{source.filename}</strong> chunk #{source.chunk_id} - score {source.score.toFixed(4)}
                    </p>
                    <p>{source.content}</p>
                  </div>
                ))}
              </article>
            </div>
          )}
        </section>
      )}
      {error && <p className="error">{error}</p>}
    </section>
  );
}

function DebugStage({
  title,
  hits,
}: {
  title: string;
  hits: { rank: number; chunk_id: number; score: number }[];
}) {
  return (
    <article className="history-item">
      <p><strong>{title}</strong></p>
      {hits.length === 0 && <p className="muted">No results.</p>}
      {hits.length > 0 && (
        <div className="history-scores">
          {hits.map((hit) => (
            <span key={`${title}-${hit.rank}-${hit.chunk_id}`} className="score-badge">
              #{hit.rank} chunk {hit.chunk_id} ({hit.score.toFixed(4)})
            </span>
          ))}
        </div>
      )}
    </article>
  );
}

function _configNumber(config: Record<string, unknown>, key: string, fallback: number): number {
  const raw = config[key];
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  if (typeof raw === "string") {
    const parsed = Number(raw);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function _configBool(config: Record<string, unknown>, key: string, fallback: boolean): boolean {
  const raw = config[key];
  if (typeof raw === "boolean") return raw;
  if (typeof raw === "string") {
    if (raw.toLowerCase() === "true") return true;
    if (raw.toLowerCase() === "false") return false;
  }
  return fallback;
}

function _configString(config: Record<string, unknown>, key: string, fallback: string): string {
  const raw = config[key];
  if (typeof raw === "string" && raw.trim()) return raw.trim();
  return fallback;
}

