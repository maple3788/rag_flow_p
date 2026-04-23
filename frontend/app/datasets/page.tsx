"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

import { createDataset, deleteDataset, listDatasets, type Dataset } from "@/lib/api";

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [enableQueryRewrite, setEnableQueryRewrite] = useState(false);
  const [rerankEnabled, setRerankEnabled] = useState(true);
  const [rerankModel, setRerankModel] = useState("cross-encoder/ms-marco-MiniLM-L-6-v2");
  const [isLoading, setIsLoading] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [deletingDatasetId, setDeletingDatasetId] = useState<number | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    void refreshDatasets();
  }, []);

  async function refreshDatasets() {
    setIsLoading(true);
    setError("");
    try {
      setDatasets(await listDatasets());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load datasets");
    } finally {
      setIsLoading(false);
    }
  }

  async function onCreate(event: FormEvent) {
    event.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName) return;
    setIsCreating(true);
    setError("");
    try {
      await createDataset({
        name: trimmedName,
        description: description.trim(),
        config: {
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
          enable_query_rewrite: enableQueryRewrite,
          rerank_enabled: rerankEnabled,
          rerank_model: rerankModel,
        },
      });
      setName("");
      setDescription("");
      setChunkSize(500);
      setChunkOverlap(50);
      setEnableQueryRewrite(false);
      setRerankEnabled(true);
      setRerankModel("cross-encoder/ms-marco-MiniLM-L-6-v2");
      await refreshDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create dataset");
    } finally {
      setIsCreating(false);
    }
  }

  async function onDeleteDataset(datasetId: number) {
    const confirmed = window.confirm("Delete this dataset and all its files/chunks?");
    if (!confirmed) return;
    setDeletingDatasetId(datasetId);
    setError("");
    try {
      await deleteDataset(datasetId);
      await refreshDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete dataset");
    } finally {
      setDeletingDatasetId(null);
    }
  }

  return (
    <section className="card">
      <h2>Datasets</h2>
      <p className="muted">Create dataset-level retrieval spaces and manage their files/chunks.</p>

      <form className="dataset-create-form" onSubmit={onCreate}>
        <input
          className="inspector-input"
          placeholder="Dataset name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          disabled={isCreating}
        />
        <textarea
          className="json-box"
          placeholder="Description"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          disabled={isCreating}
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
            disabled={isCreating}
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
            disabled={isCreating}
          />
        </label>
        <p className="muted">Fusion method: rrf (fixed)</p>
        <label className="inspector-checkbox">
          <input
            type="checkbox"
            checked={enableQueryRewrite}
            onChange={(event) => setEnableQueryRewrite(event.target.checked)}
            disabled={isCreating}
          />
          Enable query rewrite by default
        </label>
        <label className="inspector-checkbox">
          <input
            type="checkbox"
            checked={rerankEnabled}
            onChange={(event) => setRerankEnabled(event.target.checked)}
            disabled={isCreating}
          />
          Enable rerank
        </label>
        <label className="muted">
          Rerank model
          <input
            className="inspector-input"
            value={rerankModel}
            onChange={(event) => setRerankModel(event.target.value)}
            disabled={isCreating}
          />
        </label>
        <button className="button" type="submit" disabled={isCreating || !name.trim()}>
          {isCreating ? "Creating..." : "Create Dataset"}
        </button>
      </form>

      <div className="history-header">
        <h3>All Datasets</h3>
        <button className="button secondary" onClick={() => void refreshDatasets()} disabled={isLoading}>
          {isLoading ? "Loading..." : "Refresh"}
        </button>
      </div>
      {datasets.length === 0 && !isLoading && <p className="muted">No datasets yet.</p>}
      <div className="trend-list">
        {datasets.map((dataset) => (
          <article key={dataset.id} className="history-item">
            <p>
              <strong>{dataset.name}</strong>
            </p>
            <p className="muted">{dataset.description || "No description"}</p>
            <p className="muted">Created: {new Date(dataset.created_at).toLocaleString()}</p>
            <div className="actions">
              <Link className="button secondary" href={`/datasets/${dataset.id}`}>
                Open Dataset
              </Link>
              <button
                className="button"
                onClick={() => void onDeleteDataset(dataset.id)}
                disabled={deletingDatasetId === dataset.id}
              >
                {deletingDatasetId === dataset.id ? "Deleting..." : "Delete"}
              </button>
            </div>
          </article>
        ))}
      </div>
      {error && <p className="error">{error}</p>}
    </section>
  );
}
