#!/usr/bin/env python3
"""Convert writeup.md to .docx (headings, bullets, inline styles, links, inline figures)."""

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).parent
FIGS = {
    "1": HERE / "figures/fig1_layer_sweep.png",
    "2": HERE / "figures/fig2_pattern_baseline.png",
    "3": HERE / "figures/fig3_J_dose_response.png",
    "4": HERE / "figures/fig4_M_decay.png",
    "5": HERE / "figures/fig5_N_peak_vs_slot.png",
    "6": HERE / "figures/fig6_N_heatmaps.png",
    "7": HERE / "figures/fig4_distractor_examples.png",
}
TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def add_hyperlink(par, text, url):
    part = par.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                          is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1155CC")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(color)
    rpr.append(u)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    run.append(t)
    link.append(run)
    par._p.append(link)


def add_inline(par, text, italic=False):
    for piece in TOKEN.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            add_inline_run(par, piece[2:-2], bold=True, italic=italic)
        elif piece.startswith("`") and piece.endswith("`"):
            r = par.add_run(piece[1:-1])
            r.font.name = "Courier New"
            r.font.size = Pt(10)
        elif piece.startswith("[") and "](" in piece:
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", piece)
            add_hyperlink(par, m.group(1), m.group(2))
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            add_inline(par, piece[1:-1], italic=True)
        else:
            add_inline_run(par, piece, italic=italic)


def add_inline_run(par, text, bold=False, italic=False):
    # nested italics inside bold, e.g. **... *x* ...**
    for sub in re.split(r"(\*[^*]+\*)", text):
        if not sub:
            continue
        if sub.startswith("*") and sub.endswith("*") and len(sub) > 2:
            r = par.add_run(sub[1:-1])
            r.italic = True
        else:
            r = par.add_run(sub)
            r.italic = italic
        r.bold = bold


def main(md_path: Path, out_path: Path) -> None:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Georgia"
    st.font.size = Pt(11)
    lines = md_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):  # fenced code block -> one monospace paragraph with line breaks
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            par = doc.add_paragraph()
            for k, cl in enumerate(code):
                run = par.add_run(cl)
                run.font.name = "Courier New"
                run.font.size = Pt(10)
                if k < len(code) - 1:
                    run.add_break()
        elif line.startswith("### "):
            add_inline(doc.add_heading(level=3), line[4:])
        elif line.startswith("## "):
            add_inline(doc.add_heading(level=2), line[3:])
        elif line.startswith("# "):
            add_inline(doc.add_heading(level=1), line[2:])
        elif line.strip() == "---":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run("* * *")
        elif m := re.match(r"^\(Figure (\d): (.*)\)$", line):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            # full text width (6.0 in) for wide figures; stacked two-panel figures at 5.75 in so they fit a page
            from PIL import Image
            w, h = Image.open(FIGS[m.group(1)]).size
            p.add_run().add_picture(str(FIGS[m.group(1)]), width=Inches(5.75 if h > w else 6.0))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cap.add_run(f"Figure {m.group(1)}: {m.group(2)}")
            r.italic = True
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i].rstrip())
                i += 1
            i -= 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in block
                    if not re.fullmatch(r"\|[\s:|-]+\|", r)]
            t = doc.add_table(rows=len(rows), cols=len(rows[0]))
            t.style = "Table Grid"
            for ri, r in enumerate(rows):
                for ci, c in enumerate(r):
                    cell = t.cell(ri, ci)
                    cell.text = ""
                    add_inline(cell.paragraphs[0], f"**{c}**" if ri == 0 else c)
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(9.5)
            doc.add_paragraph()
        elif line.startswith("- "):
            add_inline(doc.add_paragraph(style="List Bullet"), line[2:])
        else:
            para = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(
                    r"^(#|- |\||\(Figure \d|---)", lines[i + 1]):
                i += 1
                para.append(lines[i].rstrip())
            add_inline(doc.add_paragraph(), " ".join(para))
        i += 1
    doc.core_properties.title = "cue-channels — write-up (CoT faithfulness probing)"
    doc.core_properties.author = "CCaptainX"
    doc.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
