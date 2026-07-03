import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { API_BASE } from "../config";
import { useWorkspace } from "../hooks/useWorkspace";
import { pngBase64ToFile } from "../utils/images";

const PROJECTIONS = [
  { value: "", label: "—" },
  { value: "AP", label: "AP" },
  { value: "PA", label: "PA" },
  { value: "L", label: "L" },
];

export default function DescriptionPage() {
  const { exam, hasExam } = useWorkspace();

  const [firstName, setFirstName] = useState("");
  const [surname, setSurname] = useState("");
  const [dateOfBirth, setDateOfBirth] = useState("");
  const [studyName, setStudyName] = useState("");
  const [projection, setProjection] = useState("");
  const [studyDate, setStudyDate] = useState("");
  const [technicalDescription, setTechnicalDescription] = useState("");
  const [conclusions, setConclusions] = useState("");
  const [additionalNotes, setAdditionalNotes] = useState("");

  const [previewAlpha, setPreviewAlpha] = useState(0.35);
  /** PDF attachment pages: none | original | composite | both */
  const [pdfFigures, setPdfFigures] = useState("both");

  const [imageId, setImageId] = useState("");
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const heatmapDataUrl = useMemo(
    () => (exam.heatmapBase64 ? `data:image/png;base64,${exam.heatmapBase64}` : ""),
    [exam.heatmapBase64]
  );

  const canSave =
    Boolean(
      technicalDescription.trim() ||
        conclusions.trim() ||
        additionalNotes.trim()
    );

  async function saveDescription() {
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const fd = new FormData();
      fd.append("technical_description", technicalDescription);
      fd.append("conclusions", conclusions);
      fd.append("description", additionalNotes);
      fd.append("first_name", firstName);
      fd.append("surname", surname);
      if (dateOfBirth) fd.append("date_of_birth", dateOfBirth);
      fd.append("study_name", studyName);
      if (projection) fd.append("projection", projection);
      if (studyDate) fd.append("study_date", studyDate);
      fd.append("heatmap_overlay_alpha", String(previewAlpha));

      if (exam.studySessionId) {
        fd.append("image_id", exam.studySessionId);
      }
      if (exam.studyFile) {
        fd.append("original", exam.studyFile, exam.studyFileName || "study.jpg");
      }
      if (exam.heatmapBase64) {
        fd.append(
          "heatmap",
          pngBase64ToFile(exam.heatmapBase64, "heatmap.png"),
          "heatmap.png"
        );
      }

      const { data } = await axios.post(`${API_BASE}/annotations/descriptions`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setImageId(data.image_id);
      setMessage("Description saved.");
    } catch (e) {
      setError(
        e?.response?.data?.detail
          ? String(e.response.data.detail)
          : e.message || "Save failed"
      );
    } finally {
      setSaving(false);
    }
  }

  async function exportPdf() {
    if (!imageId) {
      setError("Save the description first.");
      return;
    }
    setExporting(true);
    setError("");
    try {
      const res = await axios.get(`${API_BASE}/annotations/descriptions/${imageId}/pdf`, {
        responseType: "blob",
        params: { figures: pdfFigures },
      });
      const blob = new Blob([res.data], { type: "application/pdf" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `description-${imageId}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
      setMessage("PDF downloaded.");
    } catch (e) {
      setError(e?.response?.status === 404 ? "Description not found." : e.message || "Export failed");
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="app">
      <div className="page-shell page-shell--wide">
        <p className="back-row">
          <Link className="back-link" to="/">
            ← Back to main
          </Link>
        </p>

        <header className="app-header">
          <h1 className="app-title">Clinical description</h1>
          <p className="app-subtitle">
            Structured report fields, technical text, and conclusions. The study and heatmap from the main
            page attach on save. When exporting PDF, choose whether to include the raw study, the blended
            overlay, both pages, or text only.
          </p>
        </header>

        {hasExam && (
          <section className="previews card desc-previews-card" aria-label="Study previews">
            <div className="preview-pane">
              <div className="preview-pane-head">
                <h2 className="preview-card-title">Original</h2>
              </div>
              <div className="preview-frame">
                <img className="panel-img" src={exam.originalDataUrl} alt="Original study" />
              </div>
            </div>
            <div className="preview-pane">
              <div className="preview-pane-head">
                <h2 className="preview-card-title">With heatmap</h2>
              </div>
              <div className="preview-frame overlay-wrap desc-overlay-wrap">
                {heatmapDataUrl ? (
                  <>
                    <img className="base-img" src={exam.originalDataUrl} alt="" aria-hidden />
                    <img
                      className="heatmap-img"
                      src={heatmapDataUrl}
                      alt="Heatmap overlay"
                      style={{ opacity: previewAlpha }}
                    />
                  </>
                ) : (
                  <p className="overlay-placeholder desc-placeholder">
                    Heatmap will appear once prediction finishes on the main page.
                  </p>
                )}
              </div>
              <div
                className={`alpha-strip desc-alpha-strip${!heatmapDataUrl ? " alpha-strip--disabled" : ""}`}
              >
                <div className="alpha-strip-label">
                  <span>Heatmap blend (used for PDF composite)</span>
                  <span className="alpha-value">{previewAlpha.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={previewAlpha}
                  onChange={(e) => setPreviewAlpha(Number(e.target.value))}
                  className="range-input"
                  disabled={!heatmapDataUrl}
                />
              </div>
            </div>
          </section>
        )}

        {!hasExam && (
          <div className="card form-card desc-hint-banner" role="note">
            <p className="desc-hint-text">
              No study loaded yet. Go to{" "}
              <Link className="inline-crumb" to="/">
                main
              </Link>{" "}
              and upload an X-ray so attachments match your session.
            </p>
          </div>
        )}

        <section className="card form-card desc-form-grid" aria-label="Patient and study">
          <h2 className="form-section-title">Patient and study</h2>
          <div className="field-grid-2">
            <div>
              <label className="field-label" htmlFor="fn">
                First name
              </label>
              <input
                id="fn"
                type="text"
                className="field-input"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
              />
            </div>
            <div>
              <label className="field-label" htmlFor="sn">
                Surname
              </label>
              <input
                id="sn"
                type="text"
                className="field-input"
                value={surname}
                onChange={(e) => setSurname(e.target.value)}
              />
            </div>
            <div>
              <label className="field-label" htmlFor="dob">
                Date of birth
              </label>
              <input
                id="dob"
                type="date"
                className="field-input"
                value={dateOfBirth}
                onChange={(e) => setDateOfBirth(e.target.value)}
              />
            </div>
            <div>
              <label className="field-label" htmlFor="study-name">
                Study name
              </label>
              <input
                id="study-name"
                type="text"
                className="field-input"
                value={studyName}
                onChange={(e) => setStudyName(e.target.value)}
                placeholder="e.g. Ward CXR #12"
              />
            </div>
            <div>
              <label className="field-label" htmlFor="proj">
                Image projection
              </label>
              <select
                id="proj"
                className="field-select"
                value={projection}
                onChange={(e) => setProjection(e.target.value)}
              >
                {PROJECTIONS.map((p) => (
                  <option key={p.value || "none"} value={p.value}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="field-label" htmlFor="study-date">
                Date of study
              </label>
              <input
                id="study-date"
                type="date"
                className="field-input"
                value={studyDate}
                onChange={(e) => setStudyDate(e.target.value)}
              />
            </div>
          </div>
        </section>

        <section className="card form-card" aria-label="Report body">
          <label className="field-label" htmlFor="tech">
            Technical description
          </label>
          <textarea
            id="tech"
            className="field-textarea"
            rows={5}
            value={technicalDescription}
            onChange={(e) => setTechnicalDescription(e.target.value)}
            placeholder="Technique, findings, measurements…"
          />

          <label className="field-label field-label--spaced" htmlFor="conc">
            Conclusions
          </label>
          <textarea
            id="conc"
            className="field-textarea"
            rows={4}
            value={conclusions}
            onChange={(e) => setConclusions(e.target.value)}
            placeholder="Summary and recommendations…"
          />

          <label className="field-label field-label--spaced" htmlFor="notes">
            Additional notes (optional)
          </label>
          <textarea
            id="notes"
            className="field-textarea"
            rows={3}
            value={additionalNotes}
            onChange={(e) => setAdditionalNotes(e.target.value)}
            placeholder="Extra narrative stored as legacy notes in the PDF"
          />

          <p className="field-hint desc-attach-summary">
            {hasExam ? (
              <>
                Attachments on save:{" "}
                <strong>{exam.studyFileName || "study image"}</strong>
                {exam.heatmapBase64 ? " + heatmap" : " (heatmap pending)"}.
              </>
            ) : (
              <>No study file in workspace — save stores report text only.</>
            )}
          </p>

          <fieldset className="pdf-figures-fieldset">
            <legend className="field-label">Images in PDF export</legend>
            <div className="radio-stack">
              <label className="radio-option">
                <input
                  type="radio"
                  name="pdf-figures"
                  value="none"
                  checked={pdfFigures === "none"}
                  onChange={() => setPdfFigures("none")}
                />
                <span>Text only (no images)</span>
              </label>
              <label className="radio-option">
                <input
                  type="radio"
                  name="pdf-figures"
                  value="original"
                  checked={pdfFigures === "original"}
                  onChange={() => setPdfFigures("original")}
                />
                <span>Original study image only</span>
              </label>
              <label className="radio-option">
                <input
                  type="radio"
                  name="pdf-figures"
                  value="composite"
                  checked={pdfFigures === "composite"}
                  onChange={() => setPdfFigures("composite")}
                />
                <span>Study with heatmap overlay only (blended)</span>
              </label>
              <label className="radio-option">
                <input
                  type="radio"
                  name="pdf-figures"
                  value="both"
                  checked={pdfFigures === "both"}
                  onChange={() => setPdfFigures("both")}
                />
                <span>Both — original page and overlay page</span>
              </label>
            </div>
          </fieldset>

          {imageId && (
            <p className="field-hint">
              Saved record ID: <code className="inline-code">{imageId}</code>
            </p>
          )}

          {message && (
            <div className="alert alert-success" role="status">
              {message}
            </div>
          )}
          {error && (
            <div className="alert alert-error" role="alert">
              {error}
            </div>
          )}

          <div className="form-actions">
            <button
              type="button"
              className="btn btn-primary"
              disabled={saving || !canSave}
              onClick={saveDescription}
            >
              {saving ? "Saving…" : "Save description"}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={exporting || !imageId}
              onClick={exportPdf}
            >
              {exporting ? "Exporting…" : "Export to PDF"}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
