import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { API_BASE } from "../config";
import { useWorkspace } from "../hooks/useWorkspace";

const TRIAGE_URL = `${API_BASE}/triage`;

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function formatPathology(name) {
  return name.replace(/_/g, " ");
}

function formatTriageLevel(level) {
  const labels = {
    critical: "Critical",
    high: "High",
    medium: "Medium",
    low: "Low",
  };
  return labels[level] ?? level;
}

export default function HomePage() {
  const { patchExamFromUpload, setExamHeatmap } = useWorkspace();
  const [fileName, setFileName] = useState("");
  const [originalDataUrl, setOriginalDataUrl] = useState("");
  const [heatmapBase64, setHeatmapBase64] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [triageLevel, setTriageLevel] = useState("");
  const [highRiskFindings, setHighRiskFindings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [alpha, setAlpha] = useState(0.35);
  const [dragOver, setDragOver] = useState(false);
  const [useMask, setUseMask] = useState(false);
  const [appliedMask, setAppliedMask] = useState(false);

  const heatmapDataUrl = useMemo(
    () => (heatmapBase64 ? `data:image/png;base64,${heatmapBase64}` : ""),
    [heatmapBase64]
  );

  const sortedPredictions = useMemo(
    () => [...predictions].sort((a, b) => b.probability - a.probability),
    [predictions]
  );

  const highRiskSet = useMemo(
    () => new Set(highRiskFindings.map((f) => f.pathology)),
    [highRiskFindings]
  );

  async function runPredictionForImage(dataUrl, maskEnabled = useMask) {
    setLoading(true);
    setError("");

    try {
      const base64 = dataUrl.split(",")[1];

      const { data } = await axios.post(TRIAGE_URL, {
        base_64_image: base64,
      }, {
        params: { use_mask: maskEnabled },
      });

      setPredictions(data.predictions || []);
      setTriageLevel(data.triage_level || "");
      setHighRiskFindings(data.high_risk_findings || []);
      const hm = data.base_64_heatmap || "";
      setHeatmapBase64(hm);
      setExamHeatmap(hm);
      setAppliedMask(maskEnabled);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleUseMaskChange(checked) {
    setUseMask(checked);
    if (originalDataUrl && !loading) {
      await runPredictionForImage(originalDataUrl, checked);
    }
  }

  async function handleSelectedFile(selected) {
    if (!selected) return;
    setFileName(selected.name);
    setError("");
    setHeatmapBase64("");
    setPredictions([]);
    setTriageLevel("");
    setHighRiskFindings([]);
    setAppliedMask(false);

    const dataUrl = await fileToDataUrl(selected);
    setOriginalDataUrl(dataUrl);
    patchExamFromUpload(selected, dataUrl);
    await runPredictionForImage(dataUrl);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files?.[0];
    handleSelectedFile(dropped);
  }

  function onDragOver(e) {
    e.preventDefault();
    setDragOver(true);
  }

  function onDragLeave(e) {
    e.preventDefault();
    setDragOver(false);
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Chest X-ray Diagnostics</h1>
        <p className="app-subtitle">
          Upload a study, review triage and class probabilities (per-class model thresholds), and
          inspect the overlay attention map.
        </p>
      </header>

      <section className="card card-upload nav-actions-card" aria-label="Workflow shortcuts">
        <h2 className="nav-actions-title">Documentation</h2>
        <p className="nav-actions-sub">
          The study and heatmap from this page are carried into description and labeling. Upload an X-ray
          here first, then open Add description or Add Label.
        </p>
        <div className="nav-actions">
          <Link className="btn btn-primary" to="/description">
            Add description
          </Link>
          <Link className="btn btn-primary" to="/label">
            Add Label
          </Link>
        </div>
      </section>

      <section className="card analysis-options-card" aria-label="Analysis options">
        <h2 className="analysis-options-title">Segmentation</h2>
        <label className="checkbox-row analysis-checkbox">
          <input
            type="checkbox"
            checked={useMask}
            onChange={(e) => handleUseMaskChange(e.target.checked)}
            disabled={loading}
          />
          <span>
            <strong>Use thoracic mask</strong>
            <span className="analysis-options-hint">
              {" "}
              Segment lungs and heart, predict on the cropped region, and map the heatmap back
              onto the full image.
            </span>
          </span>
        </label>
        {originalDataUrl && !loading && heatmapBase64 && (
          <p className="analysis-mode-note">
            Current results:{" "}
            <span className={`mode-badge mode-badge--${appliedMask ? "on" : "off"}`}>
              {appliedMask ? "with segmentation" : "full image"}
            </span>
          </p>
        )}
      </section>

      <section className="previews card" aria-label="Preview">
        <div className="preview-pane">
          <div className="preview-pane-head preview-pane-head--upload">
            <h2 className="preview-card-title">Original</h2>
            <label className="file-pick">
              <span>Select photo</span>
              <input
                type="file"
                accept="image/*"
                className="visually-hidden"
                onChange={(e) => handleSelectedFile(e.target.files?.[0] ?? null)}
              />
            </label>
          </div>
          <div
            className={`dropzone preview-frame${dragOver ? " dropzone--active" : ""}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            {originalDataUrl ? (
              <img className="panel-img" src={originalDataUrl} alt="Original X-ray" />
            ) : (
              <div className="dropzone-inner">
                <p className="dropzone-title">Drop an image here</p>
                <p className="dropzone-hint">
                  or use &quot;Select photo&quot; to choose from your device
                </p>
              </div>
            )}
          </div>
          {fileName && (
            <div className="file-chip" title={fileName}>
              <span className="file-chip-name">{fileName}</span>
            </div>
          )}
          {loading && <p className="predicting-note">Running prediction...</p>}
          {error && (
            <div className="alert alert-error" role="alert">
              {typeof error === "string"
                ? error
                : Array.isArray(error)
                  ? error.map((x) => x.msg ?? x).join(" ")
                  : "Request failed"}
            </div>
          )}
        </div>

        <div className="preview-pane">
          <div className="preview-pane-head">
            <h2 className="preview-card-title">
              With heatmap{appliedMask ? " (segmented)" : ""}
            </h2>
          </div>
          <div className="preview-frame overlay-wrap">
            {originalDataUrl && heatmapDataUrl ? (
              <>
                <img
                  className="base-img"
                  src={originalDataUrl}
                  alt=""
                  aria-hidden="true"
                />
                <img
                  className="heatmap-img"
                  src={heatmapDataUrl}
                  alt="Heatmap overlaid on X-ray"
                  style={{ opacity: alpha }}
                />
              </>
            ) : (
              <p className="overlay-placeholder">
                {originalDataUrl
                  ? "Generating overlay for the selected image..."
                  : "Select or drop an image to preview the heatmap overlay."}
              </p>
            )}
          </div>
          <div className={`alpha-strip${!heatmapDataUrl ? " alpha-strip--disabled" : ""}`}>
            <div className="alpha-strip-label">
              <span>Heatmap blend</span>
              <span className="alpha-value">{alpha.toFixed(2)}</span>
            </div>
            <input
              id="alpha-range"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
              className="range-input"
              disabled={!heatmapDataUrl}
            />
          </div>
        </div>
      </section>

      {triageLevel && (
        <section
          className={`card triage-banner triage-banner--${triageLevel}`}
          aria-label="Triage assessment"
        >
          <div className="triage-banner-main">
            <span className="triage-banner-label">Overall triage</span>
            <span className={`triage-banner-level triage-banner-level--${triageLevel}`}>
              {formatTriageLevel(triageLevel)}
            </span>
          </div>
          <p className="triage-banner-sub">
            {highRiskFindings.length > 0
              ? `${highRiskFindings.length} finding(s) at or above the model threshold.`
              : "No findings above the per-class model threshold."}
          </p>
        </section>
      )}

      {highRiskFindings.length > 0 && (
        <section className="card card-probs card-probs--compact" aria-label="High-risk findings">
          <div className="card-probs-head">
            <h2 className="card-probs-title">High-risk findings</h2>
            <p className="card-probs-sub">Classes meeting or exceeding their trained threshold.</p>
          </div>
          <ul className="prob-list">
            {highRiskFindings.map((p) => {
              const pct = p.probability * 100;
              return (
                <li key={p.pathology} className="prob-row">
                  <div className="prob-row-top">
                    <span className="prob-name">{formatPathology(p.pathology)}</span>
                    <span className="prob-badge prob-badge--positive" aria-label="Above threshold">
                      Above threshold
                    </span>
                    <span className="prob-pct">{pct.toFixed(1)}%</span>
                  </div>
                  <div className="prob-track prob-track--high" role="presentation">
                    <div className="prob-fill" style={{ width: `${Math.min(100, pct)}%` }} />
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {sortedPredictions.length > 0 && (
        <section className="card card-probs" aria-label="Probabilities">
          <div className="card-probs-head">
            <h2 className="card-probs-title">All classes</h2>
            <p className="card-probs-sub">
              Sorted by estimated probability. Badges use per-class thresholds from the model, not
              fixed cutoffs.
            </p>
          </div>
          <ul className="prob-list">
            {sortedPredictions.map((p) => {
              const aboveThreshold = highRiskSet.has(p.pathology);
              const tier = aboveThreshold ? "positive" : "below";
              const pct = p.probability * 100;
              return (
                <li key={p.pathology} className="prob-row">
                  <div className="prob-row-top">
                    <span className="prob-name">{formatPathology(p.pathology)}</span>
                    <span
                      className={`prob-badge prob-badge--${tier}`}
                      aria-label={aboveThreshold ? "Above threshold" : "Below threshold"}
                    >
                      {aboveThreshold ? "Above threshold" : "Below threshold"}
                    </span>
                    <span className="prob-pct">{pct.toFixed(1)}%</span>
                  </div>
                  <div
                    className={`prob-track prob-track--${aboveThreshold ? "high" : "low"}`}
                    role="presentation"
                  >
                    <div
                      className="prob-fill"
                      style={{ width: `${Math.min(100, pct)}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
