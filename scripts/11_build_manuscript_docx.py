from __future__ import annotations

from pathlib import Path
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


STUDY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = STUDY_ROOT / "manuscript_package"


def set_document_defaults(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for style_name, size in [("Title", 16), ("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11), ("List Bullet", 11)]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        if style_name != "List Bullet":
            style.font.bold = True


def finalize_paragraph(paragraph, *, alignment: WD_ALIGN_PARAGRAPH | None = None, first_line_indent: Inches | None = Inches(0.25)) -> None:
    if alignment is not None:
        paragraph.alignment = alignment
    elif paragraph.style.name == "Normal":
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if paragraph.style.name == "Normal":
        paragraph.paragraph_format.first_line_indent = first_line_indent
    else:
        paragraph.paragraph_format.first_line_indent = None


def add_markdown_block(document: Document, line: str) -> None:
    stripped = line.strip()
    if not stripped:
        document.add_paragraph("")
        return
    stripped = stripped.replace("`", "")
    stripped = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
    if stripped.startswith("# "):
        p = document.add_paragraph(stripped[2:], style="Title")
        finalize_paragraph(p, alignment=WD_ALIGN_PARAGRAPH.CENTER, first_line_indent=None)
        return
    if stripped.startswith("## "):
        p = document.add_paragraph(stripped[3:], style="Heading 1")
        finalize_paragraph(p, alignment=WD_ALIGN_PARAGRAPH.LEFT, first_line_indent=None)
        return
    if stripped.startswith("### "):
        p = document.add_paragraph(stripped[4:], style="Heading 2")
        finalize_paragraph(p, alignment=WD_ALIGN_PARAGRAPH.LEFT, first_line_indent=None)
        return
    if stripped.startswith("- "):
        p = document.add_paragraph(stripped[2:], style="List Bullet")
        finalize_paragraph(p, alignment=WD_ALIGN_PARAGRAPH.LEFT, first_line_indent=None)
        return
    if stripped.startswith("**Keywords:**"):
        p = document.add_paragraph()
        run = p.add_run("Keywords: ")
        run.bold = True
        p.add_run(stripped.split("**Keywords:**", 1)[1].strip())
        finalize_paragraph(p, alignment=WD_ALIGN_PARAGRAPH.LEFT, first_line_indent=None)
        return
    p = document.add_paragraph(stripped)
    finalize_paragraph(p)


def markdown_to_docx(markdown_path: Path, output_path: Path) -> None:
    document = Document()
    set_document_defaults(document)
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        add_markdown_block(document, line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)


def text_to_docx(text_path: Path, title: str, output_path: Path, bullets: bool = False) -> None:
    document = Document()
    set_document_defaults(document)
    title_p = document.add_paragraph(title, style="Title")
    finalize_paragraph(title_p, alignment=WD_ALIGN_PARAGRAPH.CENTER, first_line_indent=None)
    for line in text_path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if bullets:
            if cleaned.startswith("- "):
                cleaned = cleaned[2:]
            p = document.add_paragraph(cleaned, style="List Bullet")
            finalize_paragraph(p, alignment=WD_ALIGN_PARAGRAPH.LEFT, first_line_indent=None)
        else:
            p = document.add_paragraph(cleaned)
            finalize_paragraph(p)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)


def main() -> int:
    markdown_to_docx(PACKAGE_DIR / "main_manuscript" / "manuscript.md", PACKAGE_DIR / "main_manuscript" / "manuscript.docx")
    markdown_to_docx(PACKAGE_DIR / "supplementary_information" / "SI.md", PACKAGE_DIR / "supplementary_information" / "SI.docx")
    markdown_to_docx(PACKAGE_DIR / "figures" / "figure_captions.md", PACKAGE_DIR / "figures" / "figure_captions.docx")
    markdown_to_docx(PACKAGE_DIR / "tables" / "table_captions.md", PACKAGE_DIR / "tables" / "table_captions.docx")
    markdown_to_docx(PACKAGE_DIR / "main_manuscript" / "submission_notes.md", PACKAGE_DIR / "main_manuscript" / "submission_notes.docx")
    text_to_docx(PACKAGE_DIR / "main_manuscript" / "highlights.txt", "Highlights", PACKAGE_DIR / "main_manuscript" / "highlights.docx", bullets=True)
    text_to_docx(PACKAGE_DIR / "main_manuscript" / "keywords.txt", "Keywords", PACKAGE_DIR / "main_manuscript" / "keywords.docx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
