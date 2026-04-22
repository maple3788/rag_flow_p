"use client";

import { DragEvent, useEffect, useMemo, useState } from "react";

import { listDatasets, uploadDatasetFile, type Dataset } from "@/lib/api";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [isUploading, setIsUploading] = useState(false);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");

  const accepted = useMemo(() => ".pdf,.docx,.txt", []);

  useEffect(() => {
    void loadDatasets();
  }, []);

  async function loadDatasets() {
    setIsLoadingDatasets(true);
    setError("");
    try {
      const rows = await listDatasets();
      setDatasets(rows);
      if (rows.length > 0) {
        setSelectedDatasetId(String(rows[0].id));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load datasets");
    } finally {
      setIsLoadingDatasets(false);
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      setFile(dropped);
      setMessage("");
      setError("");
    }
  }

  async function handleUpload() {
    if (!file || !selectedDatasetId) return;
    setIsUploading(true);
    setError("");
    setMessage("");
    try {
      const result = await uploadDatasetFile(Number(selectedDatasetId), file);
      setMessage(
        `Indexed "${result.filename}" into dataset #${result.dataset_id} (file #${result.id}).`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <section className="card">
      <h2>Upload Documents</h2>
      <p>Accepted formats: PDF, DOCX, TXT.</p>
      <div className="history-header">
        <label className="muted" htmlFor="dataset-select">
          Target dataset
        </label>
        <select
          id="dataset-select"
          className="model-select"
          value={selectedDatasetId}
          onChange={(event) => setSelectedDatasetId(event.target.value)}
          disabled={isUploading || isLoadingDatasets || datasets.length === 0}
        >
          {datasets.length === 0 && <option value="">No datasets available</option>}
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.name} (#{dataset.id})
            </option>
          ))}
        </select>
      </div>
      {datasets.length === 0 && !isLoadingDatasets && (
        <p className="muted">Create a dataset first in the Datasets page.</p>
      )}

      <div
        className="dropzone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
      >
        <p>{file ? `Selected: ${file.name}` : "Drag and drop file here"}</p>
        <label className="button secondary" htmlFor="file-input">
          Choose File
        </label>
        <input
          id="file-input"
          type="file"
          accept={accepted}
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          hidden
        />
      </div>

      <button
        className="button"
        disabled={!file || isUploading || !selectedDatasetId}
        onClick={handleUpload}
      >
        {isUploading ? "Uploading..." : "Upload & Index"}
      </button>

      {message && <p className="success">{message}</p>}
      {error && <p className="error">{error}</p>}
    </section>
  );
}
