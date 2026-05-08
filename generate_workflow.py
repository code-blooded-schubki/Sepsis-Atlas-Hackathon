import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 22))
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# ── helpers ──────────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, color, text_lines, title_size=13,
                body_size=10.5, title_color='white', body_color='white',
                radius=0.35, alpha=0.93):
    """Draw a rounded rectangle with a title (first line) and body lines."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.6, edgecolor='white',
        facecolor=color, alpha=alpha, zorder=3
    )
    ax.add_patch(box)

    # optional light inner shadow
    shadow = FancyBboxPatch(
        (x + 0.07, y - 0.07), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, edgecolor='none',
        facecolor='#00000018', alpha=1, zorder=2
    )
    ax.add_patch(shadow)

    if not text_lines:
        return

    title = text_lines[0]
    body  = text_lines[1:]

    title_y = y + h - 0.42
    ax.text(x + w / 2, title_y, title,
            ha='center', va='top', fontsize=title_size,
            fontweight='bold', color=title_color, zorder=4,
            wrap=False)

    if body:
        body_start_y = title_y - 0.42
        line_gap = (h - 0.85) / max(len(body), 1)
        line_gap = min(line_gap, 0.42)
        for i, line in enumerate(body):
            ax.text(x + w / 2, body_start_y - i * line_gap, line,
                    ha='center', va='top', fontsize=body_size,
                    color=body_color, zorder=4,
                    fontstyle='italic' if line.startswith('•') else 'normal')


def arrow(ax, x1, y1, x2, y2, color='#37474F', lw=2.2, style='-|>'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color, lw=lw,
                    connectionstyle='arc3,rad=0.0',
                    mutation_scale=20
                ), zorder=5)


def label_arrow(ax, x, y, text, color='#37474F'):
    ax.text(x, y, text, ha='center', va='center', fontsize=8.5,
            color=color, fontstyle='italic', zorder=6,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='#FFFFFFCC',
                      edgecolor='none'))

# ── colour palette ────────────────────────────────────────────────────────────
C_BLUE    = '#2196F3'
C_GREEN   = '#388E3C'
C_ORANGE  = '#E65100'
C_PURPLE  = '#7B1FA2'
C_DARK    = '#1565C0'
C_TEAL    = '#00796B'

# ── layout constants ─────────────────────────────────────────────────────────
CX  = 3.0          # left edge of centre column boxes
CW  = 10.0         # width  of centre column boxes
MID = CX + CW / 2  # horizontal midpoint = 8.0

# y-positions (top of each box)
Y1_TOP = 21.0   # INPUT
Y2_TOP = 18.6   # PDF EXTRACTION
Y3_TOP = 15.4   # LLM EXTRACTION
Y4_TOP = 11.6   # parallel pair  (CONFIDENCE + VERIFIABILITY)
Y5_TOP =  8.2   # SQLITE DATABASE
Y6_TOP =  5.0   # NL QUERY
Y7_TOP =  1.8   # STREAMLIT DASHBOARD

# box heights
H1 = 1.6
H2 = 2.6
H3 = 3.2
H4 = 3.0
H5 = 2.6
H6 = 2.6
H7 = 1.6

# ── 1  INPUT ─────────────────────────────────────────────────────────────────
rounded_box(ax, CX, Y1_TOP - H1, CW, H1, C_BLUE,
            ['INPUT',
             '29 Sepsis Research PDFs  (heterogeneous formats, tables, prose)'],
            title_size=14, body_size=11)

# ── 2  PDF EXTRACTION ────────────────────────────────────────────────────────
rounded_box(ax, CX, Y2_TOP - H2, CW, H2, C_GREEN,
            ['PDF TEXT EXTRACTION',
             'pdfplumber  +  PyMuPDF',
             '• Full prose text extracted page by page',
             '• Structured tables preserved  →  col | col | col'],
            title_size=13, body_size=10.5)

# ── 3  LLM EXTRACTION ────────────────────────────────────────────────────────
rounded_box(ax, CX, Y3_TOP - H3, CW, H3, C_GREEN,
            ['LLM STRUCTURED EXTRACTION   (Claude Sonnet 4.5)',
             'Each field returned as JSON object with three sub-fields:',
             '  value                   —  extracted answer',
             '  source_sentence    —  verbatim quote from PDF',
             '  confidence              —  self-graded 0.0 – 1.0',
             '',
             'Fields:  metadata  ·  sepsis_definition  ·  cohorts[ ]  ·  findings[ ]'],
            title_size=13, body_size=10)

# ── 4  PARALLEL BOXES ────────────────────────────────────────────────────────
GAP  = 0.3
PW   = (CW - GAP) / 2      # width of each parallel box
LX   = CX                  # left  box x
RX   = CX + PW + GAP       # right box x

rounded_box(ax, LX, Y4_TOP - H4, PW, H4, C_ORANGE,
            ['CONFIDENCE SCORING',
             '(LLM self-grading)',
             '',
             '≥ 0.9  —  explicitly stated in paper',
             '~ 0.6  —  inferred / implied',
             '  0.0  —  information not found'],
            title_size=12, body_size=10)

rounded_box(ax, RX, Y4_TOP - H4, PW, H4, C_ORANGE,
            ['VERIFIABILITY CHECK',
             '(deterministic — no LLM)',
             '',
             '✓  source_sentence found verbatim in PDF?',
             '✓  numeric values match original source?'],
            title_size=12, body_size=10)

# ── 5  SQLITE ────────────────────────────────────────────────────────────────
rounded_box(ax, CX, Y5_TOP - H5, CW, H5, C_PURPLE,
            ['SQLite DATABASE',
             'papers   —  one row per study  (title, authors, year, DOI, …)',
             'cohorts  —  one row per sub-population  (n, age, SOFA, mortality, …)',
             'findings —  one row per predictor → outcome association'],
            title_size=13, body_size=10.5)

# ── 6  NL QUERY ──────────────────────────────────────────────────────────────
rounded_box(ax, CX, Y6_TOP - H6, CW, H6, C_DARK,
            ['NATURAL-LANGUAGE QUERY INTERFACE',
             'User question',
             '→  LLM generates SQL  (schema-aware prompt)',
             '→  Query executed on SQLite',
             '→  Structured result table returned'],
            title_size=13, body_size=10.5)

# ── 7  DASHBOARD ─────────────────────────────────────────────────────────────
rounded_box(ax, CX, Y7_TOP - H7, CW, H7, C_TEAL,
            ['STREAMLIT DASHBOARD',
             'Evidence Query  ·  Browse Database  ·  Export CSV'],
            title_size=14, body_size=11)

# ── ARROWS (centre-to-centre vertical) ───────────────────────────────────────
arrow_color = '#263238'
aw = 2.5

# 1 → 2
arrow(ax, MID, Y1_TOP - H1, MID, Y2_TOP, color=arrow_color, lw=aw)
# 2 → 3
arrow(ax, MID, Y2_TOP - H2, MID, Y3_TOP, color=arrow_color, lw=aw)

# 3 → left parallel box  (angled)
arrow(ax, MID, Y3_TOP - H3, LX + PW / 2, Y4_TOP, color=arrow_color, lw=aw)
# 3 → right parallel box (angled)
arrow(ax, MID, Y3_TOP - H3, RX + PW / 2, Y4_TOP, color=arrow_color, lw=aw)

# left  parallel → 5
arrow(ax, LX + PW / 2, Y4_TOP - H4, MID, Y5_TOP, color=arrow_color, lw=aw)
# right parallel → 5
arrow(ax, RX + PW / 2, Y4_TOP - H4, MID, Y5_TOP, color=arrow_color, lw=aw)

# 5 → 6
arrow(ax, MID, Y5_TOP - H5, MID, Y6_TOP, color=arrow_color, lw=aw)
# 6 → 7
arrow(ax, MID, Y6_TOP - H6, MID, Y7_TOP, color=arrow_color, lw=aw)

# ── TITLE ─────────────────────────────────────────────────────────────────────
ax.text(MID, 21.75,
        'Sepsis Atlas  —  End-to-End Pipeline',
        ha='center', va='center', fontsize=17, fontweight='bold',
        color='#1A237E', zorder=6)
ax.text(MID, 21.38,
        'Automated extraction, verification and querying of sepsis research',
        ha='center', va='center', fontsize=11, color='#455A64', zorder=6)

# ── SAVE ──────────────────────────────────────────────────────────────────────
out = '/Users/pratularyan/Desktop/hackathon/Sepsis-Atlas-Hackathon/workflow.png'
plt.tight_layout(pad=0.4)
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f'Saved → {out}')
