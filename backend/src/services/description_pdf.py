"""Render a clinical description and optional images into a PDF (fpdf2)."""

from __future__ import annotations

import inspect
import os
import tempfile
from typing import Literal
from functools import lru_cache
from pathlib import Path

from fpdf import FPDF
from PIL import Image

from src.domain.annotations.description_manifest import DescriptionManifest
from src.domain.annotations.description_model import Description

PdfFiguresMode = Literal["none", "original", "composite", "both"]


def _usable_width(pdf: FPDF) -> float:
    return pdf.w - pdf.l_margin - pdf.r_margin


@lru_cache(maxsize=1)
def _advance_line_kwargs() -> dict[str, object]:
    """
    fpdf2 versions differ: newer uses new_x/new_y; older uses ln.
    Avoid importing fpdf.enums (not present in some installs). String enums work on newer fpdf2.
    """
    sig = inspect.signature(FPDF.multi_cell)
    if "new_x" in sig.parameters:
        return {"new_x": "LMARGIN", "new_y": "NEXT"}
    return {"ln": 1}


def _dejavu_paths_from_matplotlib() -> tuple[Path | None, Path | None]:
    """DejaVu Sans ships with matplotlib (already a backend dependency). Full Unicode coverage."""
    try:
        import matplotlib

        base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        regular = base / "DejaVuSans.ttf"
        bold = base / "DejaVuSans-Bold.ttf"
        if regular.is_file() and bold.is_file():
            return regular, bold
    except Exception:
        pass
    return None, None


def register_unicode_font(pdf: FPDF) -> str:
    """
    Load a Unicode-capable font so Polish (ę, ą, …) and other UTF-8 text work.

    Priority:
    1) FPDF_UNICODE_FONT_REGULAR + FPDF_UNICODE_FONT_BOLD env paths (both .ttf)
    2) DejaVu Sans from matplotlib's mpl-data (DejaVuSans.ttf + DejaVuSans-Bold.ttf)
    3) Helvetica (Latin-1 only — may raise on non-Latin characters)

    Other common choices if you add files yourself: Noto Sans, Liberation Sans, Arial Unicode MS.
    """
    reg_env = os.environ.get("FPDF_UNICODE_FONT_REGULAR")
    bold_env = os.environ.get("FPDF_UNICODE_FONT_BOLD")
    if reg_env and bold_env:
        rp, bp = Path(reg_env), Path(bold_env)
        if rp.is_file() and bp.is_file():
            pdf.add_font("ReportUni", "", str(rp))
            pdf.add_font("ReportUni", "B", str(bp))
            return "ReportUni"

    regular, bold = _dejavu_paths_from_matplotlib()
    if regular and bold:
        pdf.add_font("DejaVu", "", str(regular))
        pdf.add_font("DejaVu", "B", str(bold))
        return "DejaVu"

    return "Helvetica"


def _composite_original_heatmap(original: Path, heatmap: Path, alpha: float) -> Path:
    """Blend heatmap over original (same strategy as frontend overlay). Returns temp PNG path."""
    base = Image.open(original).convert("RGB")
    over = Image.open(heatmap).convert("RGB")
    if over.size != base.size:
        over = over.resize(base.size, Image.Resampling.LANCZOS)
    blended = Image.blend(base, over, alpha)
    fd, path_str = tempfile.mkstemp(suffix=".png", prefix="cxr-composite-")
    os.close(fd)
    out = Path(path_str)
    blended.save(out, format="PNG")
    return out


def _write_report_text(pdf: FPDF, d: Description, *, font_family: str) -> None:
    uw = _usable_width(pdf)
    line_h = 7
    kw = _advance_line_kwargs()

    def line(label: str, value: str | None) -> None:
        if value is None or str(value).strip() == "":
            return
        pdf.set_font(font_family, "B", 11)
        pdf.multi_cell(uw, line_h, label, **kw)
        pdf.set_font(font_family, "", 11)
        pdf.multi_cell(uw, line_h, str(value).strip(), **kw)
        pdf.ln(1)

    pdf.set_font(font_family, "B", 14)
    pdf.cell(uw, 10, "Chest X-ray report", **kw)
    pdf.ln(2)

    patient_parts = [p for p in (d.first_name.strip(), d.surname.strip()) if p]
    if patient_parts:
        pdf.set_font(font_family, "B", 11)
        pdf.cell(uw, line_h, "Patient", **kw)
        pdf.set_font(font_family, "", 11)
        pdf.cell(uw, line_h, " ".join(patient_parts), **kw)
        pdf.ln(2)

    line("Date of birth", d.date_of_birth.isoformat() if d.date_of_birth else None)
    line("Study name", d.study_name or None)
    line("Projection", d.projection or None)
    line("Study date", d.study_date.isoformat() if d.study_date else None)

    pdf.ln(2)

    if d.technical_description.strip():
        pdf.set_font(font_family, "B", 12)
        pdf.multi_cell(uw, 8, "Technical description", **kw)
        pdf.set_font(font_family, "", 11)
        pdf.multi_cell(uw, 7, d.technical_description.strip(), **kw)
        pdf.ln(3)

    if d.conclusions.strip():
        pdf.set_font(font_family, "B", 12)
        pdf.multi_cell(uw, 8, "Conclusions", **kw)
        pdf.set_font(font_family, "", 11)
        pdf.multi_cell(uw, 7, d.conclusions.strip(), **kw)
        pdf.ln(3)

    if d.description.strip():
        pdf.set_font(font_family, "B", 12)
        pdf.multi_cell(uw, 8, "Additional notes", **kw)
        pdf.set_font(font_family, "", 11)
        pdf.multi_cell(uw, 7, d.description.strip(), **kw)


def _append_figure_pages(
    pdf: FPDF,
    *,
    font_family: str,
    manifest: DescriptionManifest,
    attachment_paths: dict[str, Path],
    figures: PdfFiguresMode,
) -> list[Path]:
    """Append image pages; returns temp composite PNG paths to delete after PDF generation."""
    uw = _usable_width(pdf)
    to_cleanup: list[Path] = []

    original = attachment_paths.get("original")
    heatmap = attachment_paths.get("heatmap")
    has_orig = bool(original and original.is_file())
    has_hm = bool(heatmap and heatmap.is_file())
    can_composite = has_orig and has_hm
    alpha = manifest.description.heatmap_overlay_alpha

    def add_figure_page(title: str, image_path: Path) -> None:
        pdf.add_page()
        pdf.set_font(font_family, "", 11)
        pdf.cell(0, 8, title)
        pdf.ln(6)
        pdf.image(str(image_path), w=min(uw, 180))

    if figures == "none":
        return to_cleanup

    if figures == "original":
        if has_orig:
            add_figure_page("Study image", original)
        return to_cleanup

    if figures == "composite":
        if can_composite:
            comp = _composite_original_heatmap(original, heatmap, alpha)
            to_cleanup.append(comp)
            add_figure_page("Study with heatmap overlay", comp)
        elif has_hm and not has_orig:
            add_figure_page("Heatmap", heatmap)
        elif has_orig:
            add_figure_page("Study image", original)
        return to_cleanup

    # figures == "both"
    if has_orig:
        add_figure_page("Study image", original)
    if can_composite:
        comp = _composite_original_heatmap(original, heatmap, alpha)
        to_cleanup.append(comp)
        add_figure_page("Study with heatmap overlay", comp)
    elif has_hm and not has_orig:
        add_figure_page("Heatmap", heatmap)

    return to_cleanup


class DescriptionPdfBuilder:
    """Builds a PDF report from stored description metadata."""

    def build(
        self,
        manifest: DescriptionManifest,
        attachment_paths: dict[str, Path],
        *,
        figures: PdfFiguresMode = "both",
    ) -> bytes:
        pdf = FPDF()
        font_family = register_unicode_font(pdf)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        _write_report_text(pdf, manifest.description, font_family=font_family)

        temp_files = _append_figure_pages(
            pdf,
            font_family=font_family,
            manifest=manifest,
            attachment_paths=attachment_paths,
            figures=figures,
        )

        try:
            output = pdf.output(dest="S")
            if isinstance(output, str):
                return output.encode("latin-1")
            return bytes(output)
        finally:
            for p in temp_files:
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass
