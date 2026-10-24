"""Build the Word document from the measured results.

This file holds only the mechanics of writing a .docx. The numbers come from
report_facts.py and the wording from the three section modules, so changing how the
document looks never risks changing what it says.

Run it last: it asserts that every chart it references already exists.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

import config as cfg
import report_facts
import section_conclusion as sc
import section_discussion as sd
import section_eda as se
import section_errors as serr
import section_methods as sm
import section_models as smo
import section_robustness as srob


def _results(F: dict) -> list[tuple]:
    """Four parts: what the data looks like, how accurate the models were, where they
    failed, and whether the findings survive different choices."""
    return se.blocks(F) + smo.blocks(F) + serr.blocks(F) + srob.blocks(F)


SECTIONS = [
    ("Data and Methodology", sm.blocks),
    ("Data Analysis and Results", _results),
    ("Discussion of Findings", sd.discussion_blocks),
    ("Conclusion and Recommendations", sc.conclusion_blocks),
]


def _topic(doc: Document, text: str) -> int:
    """A bold topic sentence. Returns its word count."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.add_run(text).bold = True
    return len(text.split())


def _bullets(doc: Document, items: list[str]) -> int:
    """Short bullets beneath a topic sentence. Returns their total word count."""
    words = 0
    for item in items:
        doc.add_paragraph(item, style="List Bullet")
        words += len(item.split())
    return words


def _caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def _table(doc: Document, frame: pd.DataFrame, caption: str) -> None:
    table = doc.add_table(rows=1, cols=len(frame.columns))
    table.style = "Light Grid Accent 1"
    for cell, name in zip(table.rows[0].cells, frame.columns):
        cell.text = str(name)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for _, record in frame.iterrows():
        for cell, value in zip(table.add_row().cells, record):
            cell.text = str(value)
    _caption(doc, caption)


def _figure(doc: Document, filename: str, caption: str) -> None:
    path = cfg.FIG_DIR / filename
    assert path.exists() and path.stat().st_size > 0, (
        f"chart missing or empty: {filename} - run the figure steps first"
    )
    doc.add_picture(str(path), width=Inches(6.0))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    _caption(doc, caption)


def _save(doc: Document):
    """Save the document, working around it being open in Word.

    Word holds an exclusive lock on an open file, so overwriting it fails. Without this
    the whole pipeline run would be lost at the final step just because the previous
    version was on screen - so a dated copy is written alongside instead.
    """
    target = cfg.DOC_DIR / "Thesis_Sections.docx"
    try:
        doc.save(target)
        return target
    except PermissionError:
        spare = cfg.DOC_DIR / f"Thesis_Sections_{datetime.now():%Y%m%d_%H%M}.docx"
        doc.save(spare)
        print(f"NOTE: {target.name} is open in Word and could not be replaced.")
        print(f"      Written alongside it as {spare.name}. Close Word and re-run")
        print("      report.py if you would rather have the original filename.")
        return spare


def main() -> None:
    cfg.ensure_dirs()
    facts, tables = report_facts.gather()

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    tally: list[tuple[str, int, int, int]] = []
    for title, builder in SECTIONS:
        doc.add_heading(title, level=1)
        words = n_tables = n_figures = 0
        for block in builder(facts):
            kind = block[0]
            if kind == "topic":
                words += _topic(doc, block[1])
            elif kind == "bullets":
                words += _bullets(doc, block[1])
            elif kind == "table":
                _table(doc, tables[block[1]], block[2])
                n_tables += 1
            elif kind == "figure":
                _figure(doc, block[1], block[2])
                n_figures += 1
            else:
                raise ValueError(f"unknown block type: {kind}")
        tally.append((title, words, n_tables, n_figures))

    out = _save(doc)
    print(f"Saved {out}\n")
    print(f"{'Section':34s} {'Words':>7s} {'Tables':>7s} {'Figures':>8s}")
    for title, words, n_tables, n_figures in tally:
        print(f"{title:34s} {words:>7,} {n_tables:>7} {n_figures:>8}")
    totals = [sum(c[i] for c in tally) for i in (1, 2, 3)]
    print(f"{'TOTAL':34s} {totals[0]:>7,} {totals[1]:>7} {totals[2]:>8}")
    print("\nWord count covers topic sentences and bullets only. Headings, table "
          "contents and figure captions are excluded, as most institutions exclude "
          "them - check your own regulations.")


if __name__ == "__main__":
    main()
