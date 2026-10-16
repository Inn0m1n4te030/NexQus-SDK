"""Dependency-free circuit diagrams; drawing never executes a circuit."""
from html import escape
from pathlib import Path
import numpy as np
from .parameters import Expression
from .circuit import positive_integer


class SVGDiagram(str):
    """SVG source that displays directly in a Jupyter notebook."""
    def _repr_svg_(self):
        return str(self)


def _expression(value):
    if not isinstance(value, Expression):
        return format(value, '.4g')
    if value.operation == 'parameter':
        return value.args[0]
    symbol = {'add': '+', 'mul': '*', 'div': '/'}[value.operation]
    return '(' + symbol.join(_expression(a) for a in value.args) + ')'


def _label(op):
    if op.kind == 'parameterized':
        return f'{op.name}({_expression(op.data[1])})'
    if op.kind == 'measure':
        return 'M'
    if op.kind == 'reset':
        return '|0>'
    if op.kind == 'noise':
        return 'noise: ' + str(op.name)
    return str(op.name)


def _symbol(op):
    # User-defined gates may reuse built-in names. Verify their data before
    # displaying a special symbol rather than trusting a display name.
    if len(op.wires) != 2:
        return None
    if op.kind == 'unitary':
        cx = np.eye(4); cx[2:, 2:] = [[0, 1], [1, 0]]
        swap = np.eye(4)[[0, 2, 1, 3]]
        if op.name in ('CX', 'CX†') and np.array_equal(op.data, cx):
            return 'cx'
        if op.name in ('SWAP', 'SWAP†') and np.array_equal(op.data, swap):
            return 'swap'
    if op.kind == 'diagonal' and op.name in ('CZ', 'CZ†') and np.array_equal(op.data, [1, 1, 1, -1]):
        return 'cz'
    return None


def _clean(value):
    # Keep labels on one line, including untrusted user-supplied gate/key names.
    return ''.join(c if c.isprintable() else ' ' for c in str(value))


def draw(circuit, output='text', *, filename=None, fold=12):
    """Draw in operation order, returning text or notebook-displayable SVG.

    ``fold`` is the maximum number of operations per panel (None disables it).
    Optional filename writes UTF-8 output. Multiwire boxes include operand
    indices, preserving wire order even for nonadjacent/reversed wires.
    Numeric gates display their stored names; symbolic gates include expressions.
    """
    if output not in ('text', 'svg'):
        raise ValueError("output must be 'text' or 'svg'")
    if fold is not None:
        fold = positive_integer(fold, 'fold')
    operations = list(circuit.operations)
    keys = circuit.classical_keys
    labels = [f'q{i}' for i in range(circuit.n_qubits)] + ['c:' + _clean(k) for k in keys]
    key_rows = {key: circuit.n_qubits + i for i, key in enumerate(keys)}
    step = fold or max(1, len(operations))
    panels = [operations[i:i+step] for i in range(0, len(operations), step)] or [[]]
    rendered = _text(panels, labels, key_rows) if output == 'text' else _svg(panels, labels, key_rows)
    if filename is not None:
        Path(filename).write_text(rendered, encoding='utf-8')
    return rendered


def _cells(op, key_rows):
    symbol = _symbol(op)
    label = _clean(_label(op))
    cells = {}
    for index, wire in enumerate(op.wires):
        if symbol == 'cx': text = '●' if index == 0 else '⊕'
        elif symbol == 'cz': text = '●'
        elif symbol == 'swap': text = '×'
        else: text = '[' + label + (f':{index}' if len(op.wires) > 1 else '') + ']'
        cells[wire] = text
    if op.kind == 'measure':
        cells[key_rows[op.data]] = '↓'
    if op.condition is not None:
        key, value = op.condition
        cells[key_rows[key]] = f'={value}'
    return cells


def _text(panels, labels, key_rows):
    blocks = []
    margin = max(map(len, labels)) + 2
    classical = set(key_rows.values())
    for panel_index, panel in enumerate(panels):
        rows = [label.ljust(margin) + ('═' if i in classical else '─') * 2 for i, label in enumerate(labels)]
        for op in panel:
            cells = _cells(op, key_rows)
            width = max(map(len, cells.values())) + 4
            low, high = min(cells), max(cells)
            for i in range(len(rows)):
                fill = '═' if i in classical else '─'
                cell = cells.get(i, '│' if low < i < high else '')
                left = (width - len(cell)) // 2
                rows[i] += fill * left + cell + fill * (width - len(cell) - left)
        prefix = f'Panel {panel_index + 1}/{len(panels)}\n' if len(panels) > 1 else ''
        blocks.append(prefix + '\n'.join(rows))
    return '\n\n'.join(blocks)


def _svg(panels, labels, key_rows):
    row_gap, top = 58, 56
    panel_height = top + row_gap * len(labels) + 18
    margin = max(90, max(map(len, labels)) * 8 + 28)
    widths = [[max(70, max(len(v) for v in _cells(op, key_rows).values()) * 8 + 24)
               for op in panel] for panel in panels]
    width = max(margin + sum(ws) + 30 for ws in widths)
    height = len(panels) * panel_height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
             '<title>NexQus quantum circuit</title>',
             '<desc>Wires run left to right in operation order. Multiwire gate suffixes identify operand indices.</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>']

    def line(x1, y1, x2, y2, color='#64748b'):
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5"/>')

    def text(x, y, value, anchor='middle', size=13):
        parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" font-family="monospace" font-size="{size}" fill="#172554">{escape(value)}</text>')

    for panel_index, (panel, ws) in enumerate(zip(panels, widths)):
        offset = panel_index * panel_height
        text(20, offset + 20, f'NexQus · panel {panel_index + 1}/{len(panels)}', 'start', 12)
        right = margin + sum(ws) + 12
        for row, label in enumerate(labels):
            y = offset + top + row * row_gap
            text(margin - 14, y, label, 'end')
            if row in key_rows.values():
                line(margin, y-2, right, y-2); line(margin, y+2, right, y+2)
            else:
                line(margin, y, right, y)
        start = margin
        for op, w in zip(panel, ws):
            x = start + w / 2
            cells = _cells(op, key_rows)
            symbol = _symbol(op)
            low, high = min(cells), max(cells)
            line(x, offset+top+low*row_gap, x, offset+top+high*row_gap, '#2563eb')
            parts.append('<g><title>' + escape(_clean(_label(op))) + '</title>')
            for row, label in cells.items():
                y = offset + top + row * row_gap
                if row in op.wires and symbol:
                    if label == '●':
                        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#1d4ed8"/>')
                    elif label == '⊕':
                        parts.append(f'<circle cx="{x}" cy="{y}" r="13" fill="white" stroke="#1d4ed8" stroke-width="2"/>')
                        line(x-9, y, x+9, y, '#1d4ed8'); line(x, y-9, x, y+9, '#1d4ed8')
                    else:
                        line(x-8,y-8,x+8,y+8,'#1d4ed8'); line(x-8,y+8,x+8,y-8,'#1d4ed8')
                else:
                    color = '#fff7ed' if op.kind == 'noise' else '#eff6ff'
                    box_width = len(label) * 8 + 12
                    parts.append(f'<rect x="{x-box_width/2}" y="{y-17}" width="{box_width}" height="34" rx="5" fill="{color}" stroke="#2563eb"/>')
                    text(x, y, label)
            parts.append('</g>')
            start += w
    parts.append('</svg>')
    return SVGDiagram(''.join(parts))
