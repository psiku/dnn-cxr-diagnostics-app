import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { API_BASE } from "../config";
import { useWorkspace } from "../hooks/useWorkspace";
import { usePathologies } from "../hooks/usePathologies";

function dispRectToNatural(img, left, top, rw, rh) {
  const sx = img.naturalWidth / img.clientWidth;
  const sy = img.naturalHeight / img.clientHeight;
  return {
    tlx: Math.round(left * sx),
    tly: Math.round(top * sy),
    nw: Math.round(rw * sx),
    nh: Math.round(rh * sy),
  };
}

function naturalToDisp(img, tlx, tly, nw, nh) {
  const sx = img.clientWidth / img.naturalWidth;
  const sy = img.clientHeight / img.naturalHeight;
  return { x: tlx * sx, y: tly * sy, w: nw * sx, h: nh * sy };
}

/**
 * Keyed by parent so state resets when workspace study or manual override changes.
 */
function LabelAnnotatorBody({
  imageUrlInitial,
  imageFileInitial,
  labelImageIdInitial,
  onManualFile,
  pathologies,
  pathologiesLoading,
}) {
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);
  const draftRef = useRef(null);

  const [imageUrl] = useState(imageUrlInitial);
  const [imageFile] = useState(imageFileInitial);
  const [imageId] = useState(labelImageIdInitial);
  const [boxes, setBoxes] = useState([]);
  const [draft, setDraft] = useState(null);
  const [selectedPathology, setSelectedPathology] = useState("");
  const [customLabel, setCustomLabel] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [resizeTick, setResizeTick] = useState(0);

  const resolvedLabel = useCallback(() => {
    const c = customLabel.trim();
    if (c) return c;
    return selectedPathology.trim();
  }, [customLabel, selectedPathology]);

  const syncCanvasToImage = useCallback(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !img.naturalWidth) return;
    const w = img.clientWidth;
    const h = img.clientHeight;
    canvas.width = w;
    canvas.height = h;
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.naturalWidth) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "rgba(34, 197, 94, 0.95)";
    ctx.lineWidth = 2;
    for (const b of boxes) {
      const r = naturalToDisp(img, b.tlx, b.tly, b.nw, b.nh);
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
      ctx.fillRect(r.x, r.y - 18, Math.min(r.w, Math.max(120, b.label.length * 7)), 18);
      ctx.fillStyle = "#f8fafc";
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(b.label.slice(0, 48), r.x + 4, r.y - 5);
    }

    const d = draftRef.current;
    if (d) {
      const left = Math.min(d.x1, d.x2);
      const top = Math.min(d.y1, d.y2);
      const rw = Math.abs(d.x2 - d.x1);
      const rh = Math.abs(d.y2 - d.y1);
      ctx.strokeStyle = "rgba(239, 68, 68, 0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect(left, top, rw, rh);
    }
  }, [boxes]);

  useEffect(() => {
    redraw();
  }, [redraw, draft, resizeTick, imageUrl]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img || !imageUrl) return undefined;
    const ro = new ResizeObserver(() => {
      syncCanvasToImage();
      setResizeTick((t) => t + 1);
    });
    ro.observe(img);
    return () => ro.disconnect();
  }, [imageUrl, syncCanvasToImage]);

  function eventPos(e) {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    return { x, y };
  }

  function handleFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    onManualFile(f);
    e.target.value = "";
  }

  function onImgLoad() {
    syncCanvasToImage();
    setResizeTick((t) => t + 1);
  }

  function handlePointerDown(e) {
    const label = resolvedLabel();
    if (!label) {
      setError("Select a pathology from the list or type a custom label before drawing.");
      return;
    }
    setError("");
    const pos = eventPos(e);
    drawingRef.current = true;
    draftRef.current = { x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y };
    setDraft({ ...draftRef.current });
    e.currentTarget.setPointerCapture(e.pointerId);
  }

  function handlePointerMove(e) {
    if (!drawingRef.current || !draftRef.current) return;
    const pos = eventPos(e);
    draftRef.current = { ...draftRef.current, x2: pos.x, y2: pos.y };
    setDraft({ ...draftRef.current });
  }

  function handlePointerUp(e) {
    if (!drawingRef.current) {
      return;
    }
    drawingRef.current = false;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }

    const d = draftRef.current;
    draftRef.current = null;
    setDraft(null);
    if (!d) return;

    const left = Math.min(d.x1, d.x2);
    const top = Math.min(d.y1, d.y2);
    const rw = Math.abs(d.x2 - d.x1);
    const rh = Math.abs(d.y2 - d.y1);
    const img = imgRef.current;
    const label = resolvedLabel();
    if (!img || !label || rw < 4 || rh < 4) return;

    const nat = dispRectToNatural(img, left, top, rw, rh);
    if (nat.nw < 2 || nat.nh < 2) return;

    setBoxes((prev) => [...prev, { id: crypto.randomUUID(), label, ...nat }]);
  }

  function removeBox(id) {
    setBoxes((prev) => prev.filter((b) => b.id !== id));
  }

  async function saveAnnotation() {
    if (!imageFile || !imageId) {
      setError("Choose an image first.");
      return;
    }
    if (boxes.length === 0) {
      setError("Draw at least one bounding box.");
      return;
    }

    setSaving(true);
    setError("");
    setMessage("");

    const labeled = {
      image_id: imageId,
      bounding_boxes: boxes.map((b) => ({
        label: b.label,
        x: Math.round(b.tlx + b.nw / 2),
        y: Math.round(b.tly + b.nh / 2),
        width: b.nw,
        height: b.nh,
      })),
      timestamp: new Date().toISOString(),
    };

    try {
      const fd = new FormData();
      fd.append("payload", JSON.stringify(labeled));
      fd.append("image", imageFile, imageFile.name);

      await axios.post(`${API_BASE}/annotations/labels`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMessage("Annotation saved on the server.");
    } catch (err) {
      setError(
        err?.response?.data?.detail ? String(err.response.data.detail) : err.message || "Save failed"
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <section className="card form-card label-controls" aria-label="Label settings">
        <div className="field-row field-row--split">
          <div>
            <label className="field-label" htmlFor="pathology-select">
              Pathology (preset)
            </label>
            <select
              id="pathology-select"
              className="field-select"
              value={selectedPathology}
              onChange={(e) => setSelectedPathology(e.target.value)}
              disabled={pathologiesLoading}
            >
              <option value="">
                {pathologiesLoading ? "Loading pathologies…" : "— Optional if you use custom —"}
              </option>
              {pathologies.map((p) => (
                <option key={p} value={p}>
                  {p.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="field-label" htmlFor="custom-label">
              Custom label
            </label>
            <input
              id="custom-label"
              type="text"
              className="field-input"
              value={customLabel}
              onChange={(e) => setCustomLabel(e.target.value)}
              placeholder="Overrides preset when filled"
            />
          </div>
        </div>

        <div className="field-row">
          <span className="field-label" id="label-image-heading">
            Study image (override)
          </span>
          <label className="file-pick">
            <span>Select image</span>
            <input
              id="label-image"
              type="file"
              accept="image/*"
              className="visually-hidden"
              onChange={handleFileChange}
              aria-labelledby="label-image-heading"
            />
          </label>
        </div>

        {imageId && (
          <p className="field-hint">
            Image ID for this session: <code className="inline-code">{imageId}</code>
          </p>
        )}

        {error && (
          <div className="alert alert-error" role="alert">
            {error}
          </div>
        )}
        {message && (
          <div className="alert alert-success" role="status">
            {message}
          </div>
        )}
      </section>

      {imageUrl ? (
        <section className="card label-viewport-card" aria-label="Drawing area">
          <div className="label-viewport">
            <img
              ref={imgRef}
              className="label-img"
              src={imageUrl}
              alt="Study to label"
              onLoad={onImgLoad}
              draggable={false}
            />
            <canvas
              ref={canvasRef}
              className="label-canvas"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
            />
          </div>
        </section>
      ) : (
        <p className="field-hint label-empty-hint">Upload a study or load one from the main page.</p>
      )}

      {boxes.length > 0 && (
        <section className="card form-card" aria-label="Boxes">
          <h2 className="form-section-title">Boxes ({boxes.length})</h2>
          <ul className="box-list">
            {boxes.map((b) => (
              <li key={b.id} className="box-list-item">
                <span className="box-list-label">{b.label}</span>
                <span className="box-list-meta">
                  {b.tlx},{b.tly} · {b.nw}×{b.nh}px
                </span>
                <button type="button" className="btn-text" onClick={() => removeBox(b.id)}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="form-actions form-actions--solo">
        <button
          type="button"
          className="btn btn-primary"
          disabled={saving || !imageFile || boxes.length === 0}
          onClick={saveAnnotation}
        >
          {saving ? "Saving…" : "Save annotation"}
        </button>
      </div>
    </>
  );
}

export default function LabelPage() {
  const { exam, hasExam } = useWorkspace();
  const { pathologies, loading: pathologiesLoading } = usePathologies();

  const wsBundle = useMemo(() => {
    if (!(hasExam && exam.originalDataUrl && exam.studyFile && exam.studySessionId)) return null;
    return {
      key: exam.studySessionId,
      imageUrlInitial: exam.originalDataUrl,
      imageFileInitial: exam.studyFile,
      labelImageIdInitial: exam.studySessionId,
    };
  }, [hasExam, exam.originalDataUrl, exam.studyFile, exam.studySessionId]);

  const [manual, setManual] = useState(null);

  useEffect(() => {
    // Reset manual override when a new study is uploaded on the main page.
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional sync to workspace identity
    setManual((prev) => {
      if (prev?.blobUrl) URL.revokeObjectURL(prev.blobUrl);
      return null;
    });
  }, [exam.studySessionId]);

  useEffect(() => {
    return () => {
      if (manual?.blobUrl) URL.revokeObjectURL(manual.blobUrl);
    };
  }, [manual]);

  function replaceWithManualFile(file) {
    setManual((prev) => {
      if (prev?.blobUrl) URL.revokeObjectURL(prev.blobUrl);
      const blobUrl = URL.createObjectURL(file);
      return {
        key: crypto.randomUUID(),
        blobUrl,
        imageUrlInitial: blobUrl,
        imageFileInitial: file,
        labelImageIdInitial: crypto.randomUUID(),
      };
    });
  }

  const active = manual ?? wsBundle ?? {
    key: "__pending__",
    imageUrlInitial: "",
    imageFileInitial: null,
    labelImageIdInitial: "",
  };

  return (
    <div className="app">
      <div className="page-shell page-shell--wide">
        <p className="back-row">
          <Link className="back-link" to="/">
            ← Back to main
          </Link>
        </p>

        <header className="app-header">
          <h1 className="app-title">Image labeling</h1>
          <p className="app-subtitle">
            {hasExam ? (
              <>
                Study from the main page is loaded below. Pick a label (or type your own), then drag to
                draw a box. Use <em>Study image</em> only if you need a different image.
              </>
            ) : (
              <>
                Pick a label (or type your own), then drag on the image to draw a rectangle, or upload a
                study below. You can also load a study first on the{" "}
                <Link className="inline-crumb" to="/">
                  main page
                </Link>
                .
              </>
            )}
          </p>
        </header>

        <LabelAnnotatorBody
          key={active.key}
          imageUrlInitial={active.imageUrlInitial}
          imageFileInitial={active.imageFileInitial}
          labelImageIdInitial={active.labelImageIdInitial}
          onManualFile={replaceWithManualFile}
          pathologies={pathologies}
          pathologiesLoading={pathologiesLoading}
        />
      </div>
    </div>
  );
}
