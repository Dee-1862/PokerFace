"""Generate MODULES.pdf from MODULES.md using reportlab."""
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
)

W, H = A4
MARGIN = 20 * mm

styles = getSampleStyleSheet()
styles.add(ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=6))
styles.add(ParagraphStyle('H1', fontSize=20, leading=24, spaceAfter=10, spaceBefore=16,
                           fontName='Helvetica-Bold', textColor=HexColor('#111111')))
styles.add(ParagraphStyle('H2', fontSize=15, leading=19, spaceAfter=8, spaceBefore=14,
                           fontName='Helvetica-Bold', textColor=HexColor('#222222')))
styles.add(ParagraphStyle('H3', fontSize=12, leading=16, spaceAfter=6, spaceBefore=10,
                           fontName='Helvetica-Bold', textColor=HexColor('#333333')))
styles.add(ParagraphStyle('CodeBlock', fontName='Courier', fontSize=8.5, leading=11,
                           backColor=HexColor('#f0f0f0'), spaceAfter=8, spaceBefore=4,
                           leftIndent=10, rightIndent=10))
styles.add(ParagraphStyle('BulletItem', parent=styles['Body'], bulletIndent=10, leftIndent=24))
styles.add(ParagraphStyle('BoldLine', parent=styles['Body'], fontName='Helvetica-Bold'))
styles.add(ParagraphStyle('Cell', fontSize=9, leading=12))
styles.add(ParagraphStyle('CellBold', fontSize=9, leading=12, fontName='Helvetica-Bold'))

TABLE_STYLE = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e0e0e0')),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#888888')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f5f5f5')]),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
])


def fmt(text):
    """Markdown inline formatting to reportlab tags."""
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<font face="Courier" size="9">\1</font>', text)
    return text


def is_separator_row(line):
    """Check if a pipe-delimited line is a --- separator row."""
    return bool(re.match(r'^\s*\|[\s\-:|]+\|\s*$', line))


def parse_table_row(line):
    """Extract cells from a pipe-delimited row."""
    return [c.strip() for c in line.strip().strip('|').split('|')]


def build_table(rows):
    """Build a reportlab Table from row data (first row = header)."""
    avail = W - 2 * MARGIN
    ncols = len(rows[0])
    data = []
    for ri, row in enumerate(rows):
        st = styles['CellBold'] if ri == 0 else styles['Cell']
        data.append([Paragraph(fmt(c), st) for c in row])

    if ncols == 2:
        widths = [avail * 0.35, avail * 0.65]
    elif ncols == 3:
        widths = [avail * 0.15, avail * 0.55, avail * 0.30]
    else:
        widths = [avail / ncols] * ncols

    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TABLE_STYLE)
    return t


def parse(text):
    lines = text.split('\n')
    elements = []
    i = 0
    in_code = False
    code_buf = []

    while i < len(lines):
        line = lines[i]

        # Code fences
        if line.strip().startswith('```'):
            if in_code:
                if code_buf:
                    elements.append(Preformatted('\n'.join(code_buf), styles['CodeBlock']))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Markdown table: header row followed by separator row
        if '|' in line and i + 1 < len(lines) and is_separator_row(lines[i + 1]):
            rows = []
            rows.append(parse_table_row(line))  # header
            i += 2  # skip header + separator
            while i < len(lines) and '|' in lines[i] and not is_separator_row(lines[i]):
                rows.append(parse_table_row(lines[i]))
                i += 1
            elements.append(Spacer(1, 3 * mm))
            elements.append(build_table(rows))
            elements.append(Spacer(1, 3 * mm))
            continue

        # Headings
        if line.startswith('# '):
            elements.append(Paragraph(fmt(line[2:].strip()), styles['H1']))
            i += 1
            continue
        if line.startswith('## '):
            elements.append(Paragraph(fmt(line[3:].strip()), styles['H2']))
            i += 1
            continue
        if line.startswith('### '):
            elements.append(Paragraph(fmt(line[4:].strip()), styles['H3']))
            i += 1
            continue

        # Bullet
        if line.strip().startswith('- '):
            elements.append(Paragraph(fmt(line.strip()[2:]), styles['BulletItem'], bulletText='\u2022'))
            i += 1
            continue

        # Bold-only line
        m = re.match(r'^\*\*(.+)\*\*$', line.strip())
        if m:
            elements.append(Paragraph(fmt(m.group(1)), styles['BoldLine']))
            i += 1
            continue

        # Horizontal rule
        if line.strip() in ('---', '***', '___'):
            elements.append(Spacer(1, 6 * mm))
            i += 1
            continue

        # Empty
        if not line.strip():
            elements.append(Spacer(1, 2 * mm))
            i += 1
            continue

        # Normal paragraph
        elements.append(Paragraph(fmt(line), styles['Body']))
        i += 1

    return elements


if __name__ == '__main__':
    md = open('MODULES.md', encoding='utf-8').read()
    doc = SimpleDocTemplate(
        'MODULES.pdf', pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    doc.build(parse(md))
    print('MODULES.pdf generated')
