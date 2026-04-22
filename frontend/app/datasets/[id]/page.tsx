"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import {
  deleteDataset,
  deleteDatasetFile,
  getDataset,
  listDatasetChunks,
  listDatasetFiles,
  updateDataset,
  uploadDatasetFile,
  type Dataset,
  type DatasetChunk,
  type DatasetFile,
} from "@/lib/api";

type TabId = "files" | "chunks" | "config";

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
  const [finalK, setFinalK] = useState(5);
  const [topKBm25, setTopKBm25] = useState(10);
  const [topKDense, setTopKDense] = useState(10);
  const [fusionMethod, setFusionMethod] = useState("rrf");
  const [enableQueryRewrite, setEnableQueryRewrite] = useState(false);
  const [rerankEnabled, setRerankEnabled] = useState(true);
  const [rerankModel, setRerankModel] = useState("cross-encoder/ms-marco-MiniLM-L-6-v2");
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
      setFinalK(_configNumber(datasetRes.config, "final_k", 5));
      setTopKBm25(_configNumber(datasetRes.config, "top_k_bm25", 10));
      setTopKDense(_configNumber(datasetRes.config, "top_k_dense", 10));
      setFusionMethod(_configString(datasetRes.config, "fusion_method", "rrf"));
      setEnableQueryRewrite(_configBool(datasetRes.config, "enable_query_rewrite", false));
      setRerankEnabled(_configBool(datasetRes.config, "rerank_enabled", true));
      setRerankModel(
        _configString(
          datasetRes.config,
          "rerank_model",
          "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
      );
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
          final_k: finalK,
          top_k_bm25: topKBm25,
          top_k_dense: topKDense,
          fusion_method: fusionMethod,
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
              Final top-k (to LLM)
              <input
                className="inspector-input"
                type="number"
                min={1}
                max={50}
                value={finalK}
                onChange={(event) => setFinalK(Number(event.target.value || 5))}
              />
            </label>
            <label className="muted">
              Top-k BM25
              <input
                className="inspector-input"
                type="number"
                min={1}
                max={200}
                value={topKBm25}
                onChange={(event) => setTopKBm25(Number(event.target.value || 10))}
              />
            </label>
            <label className="muted">
              Top-k Dense
              <input
                className="inspector-input"
                type="number"
                min={1}
                max={200}
                value={topKDense}
                onChange={(event) => setTopKDense(Number(event.target.value || 10))}
              />
            </label>
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
            <label className="muted">
              Fusion method
              <select
                className="model-select"
                value={fusionMethod}
                onChange={(event) => setFusionMethod(event.target.value)}
              >
                <option value="rrf">rrf</option>
              </select>
            </label>
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
      {error && <p className="error">{error}</p>}
    </section>
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
