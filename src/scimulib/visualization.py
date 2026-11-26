"""3D visualization utilities for Pymatgen structures and molecules using py3Dmol."""

from collections.abc import Mapping, Sequence
import html
import json
from typing import Any, Literal
import numpy as np
from pymatgen.core import Molecule, Structure

RenderStyle = Literal["ball_and_stick", "spacefill", "stick"]
AxesMode = Literal["cartesian", "cell"]
LabelMode = Literal["symbol", "index", "symbol_index"]


class ColorbarView:
    """Wrapper pairing a py3Dmol viewer with a right-side HTML/CSS color scale.

    Delegates unknown attribute access to the underlying `py3Dmol.view` instance
    while rendering a side-by-side flex layout in Jupyter and VS Code notebooks.
    """

    def __init__(self, view: Any, colorbar_html: str) -> None:
        self.view = view
        self._colorbar_html = colorbar_html

    def _repr_html_(self) -> str:
        """Returns combined HTML for rich notebook rendering."""
        viewer_html = self.view._make_html()
        return (
            '<div style="display: inline-flex; flex-direction: row; '
            'align-items: center; gap: 14px; font-family: sans-serif;">'
            f"<div>{viewer_html}</div>"
            f"{self._colorbar_html}"
            "</div>"
        )

    def show(self) -> None:
        """Displays the viewer and color scale in an IPython/Jupyter environment."""
        from IPython.display import HTML, display

        display(HTML(self._repr_html_()))

    def __getattr__(self, name: str) -> Any:
        """Forwards attribute and method lookups to the underlying py3Dmol view."""
        return getattr(self.view, name)


def _map_values_to_hex(
    values: np.ndarray,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric_clim: bool = False,
) -> tuple[list[str], float, float, list[str]]:
    """Maps 1D per-atom scalar values to hex colors and builds colorbar gradient stops.

    Args:
        values: 1D float array of per-atom property values.
        cmap: Matplotlib colormap name (e.g., `'viridis'`, `'coolwarm'`, `'bwr'`).
        vmin: Explicit lower bound for color normalization. Defaults to data minimum.
        vmax: Explicit upper bound for color normalization. Defaults to data maximum.
        symmetric_clim: If True and both `vmin` and `vmax` are None, centers the color
            limits symmetrically around zero (`[-max(|v|), +max(|v|)]`).

    Returns:
        A tuple `(atom_hex_colors, cmin, cmax, gradient_stops)` where
        `atom_hex_colors` has one hex color per site and `gradient_stops` contains
        32 evenly sampled hex colors from `cmin` to `cmax`.

    Raises:
        ImportError: If `matplotlib` is not installed.
        ValueError: If `vmin > vmax` or if `cmap` is invalid.
    """
    try:
        import matplotlib as mpl
        from matplotlib.colors import Normalize, to_hex
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to map per-atom properties to a colormap. "
            "Install it with: pip install matplotlib"
        ) from exc

    colormap = mpl.colormaps[cmap]
    finite_vals = values[np.isfinite(values)]

    if len(finite_vals) == 0:
        cmin, cmax = 0.0, 1.0
    else:
        data_min = float(np.min(finite_vals))
        data_max = float(np.max(finite_vals))
        if symmetric_clim and vmin is None and vmax is None:
            bound = max(abs(data_min), abs(data_max))
            cmin, cmax = -bound, bound
        else:
            cmin = float(vmin) if vmin is not None else data_min
            cmax = float(vmax) if vmax is not None else data_max

    if cmin > cmax:
        raise ValueError(f"'vmin' ({cmin}) cannot be greater than 'vmax' ({cmax}).")

    if np.isclose(cmin, cmax):
        # Constant property across all sites: map finite values to the colormap midpoint
        atom_colors = [
            to_hex(colormap(0.5)) if np.isfinite(val) else "#CCCCCC"
            for val in values
        ]
    else:
        norm = Normalize(vmin=cmin, vmax=cmax, clip=True)
        atom_colors = [
            to_hex(colormap(norm(val))) if np.isfinite(val) else "#CCCCCC"
            for val in values
        ]

    gradient_stops = [to_hex(colormap(t)) for t in np.linspace(0.0, 1.0, 32)]
    return atom_colors, cmin, cmax, gradient_stops


def _build_colorbar_html(
    gradient_stops: list[str],
    cmin: float,
    cmax: float,
    *,
    height: int = 240,
    property_name: str | None = None,
    value_format: str = ".2f",
) -> str:
    """Builds a vertical HTML/CSS colorbar positioned to the right of the 3D canvas."""
    bar_height = max(140, int(height * 0.65))
    ticks = np.linspace(cmax, cmin, 5)
    tick_spans = "".join(
        f'<span style="line-height: 1; white-space: nowrap;">{tick:{value_format}}</span>'
        for tick in ticks
    )

    title_html = ""
    if property_name:
        safe_title = html.escape(property_name)
        title_html = (
            f'<div style="font-size: 12px; font-weight: 600; margin-bottom: 8px; '
            f'text-align: center; max-width: 110px; word-wrap: break-word;">'
            f"{safe_title}</div>"
        )

    gradient_css = ", ".join(gradient_stops)
    return (
        '<div style="display: flex; flex-direction: column; align-items: center; '
        'user-select: none; color: inherit;">'
        f"{title_html}"
        '<div style="display: flex; flex-direction: row; align-items: stretch; gap: 6px;">'
        f'<div style="width: 16px; height: {bar_height}px; border-radius: 2px; '
        'border: 1px solid rgba(128, 128, 128, 0.5); '
        f'background: linear-gradient(to top, {gradient_css});"></div>'
        '<div style="display: flex; flex-direction: column; '
        'justify-content: space-between; font-size: 11px; font-variant-numeric: tabular-nums;">'
        f"{tick_spans}"
        "</div>"
        "</div>"
        "</div>"
    )


def _add_axes_gizmo(
    view: Any,
    target: Structure | Molecule,
    mode: AxesMode,
    length: float = 2.0,
) -> None:
    """Adds an RGB reference axes overlay fixed to the viewer's lower-right corner.

    Args:
        view: Active `py3Dmol.view` instance.
        target: Periodic `Structure` or non-periodic `Molecule` being rendered.
        mode: Coordinate system for the axes triad. `'cartesian'` draws orthogonal
            (x, y, z) vectors; `'cell'` draws normalized lattice vectors (a, b, c).
        length: Scale factor for the overlay arrows; 2.0 is the default size.

    Raises:
        ValueError: If `mode` is `'cell'` for a non-periodic `Molecule`, or if an
            unsupported `mode` value is provided.
    """
    if mode == "cell":
        if not isinstance(target, Structure):
            raise ValueError(
                "Cell axes ('cell') can only be drawn for periodic Structure objects."
            )
        raw_vectors = target.lattice.matrix
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        vectors = (raw_vectors / norms) * length
        labels = ("a", "b", "c")
    elif mode == "cartesian":
        vectors = np.eye(3) * length
        labels = ("x", "y", "z")
    else:
        raise ValueError(
            f"Unsupported axes mode '{mode}'. Choose from: 'cartesian', 'cell', or None."
        )

    axes_data = json.dumps(
        [
            {"vector": vec.tolist(), "label": label, "color": color}
            for vec, label, color in zip(
                vectors, labels, ("#D9383A", "#2EA043", "#1F6FEB"), strict=True
            )
        ]
    )
    # py3Dmol queues JSON arguments only, so a browser-side callback must be
    # appended to its initialization script.
    view.startjs += """
(function() {
    const viewer = viewer_UNIQUEID;
    const host = document.getElementById("3dmolviewer_UNIQUEID");
    const axes = AXES_DATA;
    const namespace = "http://www.w3.org/2000/svg";
    const radius = Math.max(...axes.map(axis => Math.hypot(...axis.vector))) * 14;
    const size = 2 * (radius + 22);
    const center = size / 2;
    const svg = document.createElementNS(namespace, "svg");
    svg.setAttribute("width", size);
    svg.setAttribute("height", size);
    svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
    svg.setAttribute("aria-label", "Coordinate axes");
    svg.style.cssText = "position:absolute;right:0;bottom:0;pointer-events:none;overflow:visible";
    host.appendChild(svg);

    function updateAxes(view) {
        const rotation = new $3Dmol.Quaternion(view[4], view[5], view[6], view[7]);
        const projected = axes.map(axis => {
            const vector = new $3Dmol.Vector3(...axis.vector).applyQuaternion(rotation);
            return {axis, vector};
        }).sort((a, b) => a.vector.z - b.vector.z);
        svg.replaceChildren();
        for (const {axis, vector} of projected) {
            const x = center + vector.x * 14;
            const y = center - vector.y * 14;
            const dx = x - center;
            const dy = y - center;
            const distance = Math.hypot(dx, dy);
            const group = document.createElementNS(namespace, "g");
            const line = document.createElementNS(namespace, "line");
            line.setAttribute("x1", center);
            line.setAttribute("y1", center);
            line.setAttribute("x2", x);
            line.setAttribute("y2", y);
            line.setAttribute("stroke", axis.color);
            line.setAttribute("stroke-width", "3");
            group.appendChild(line);
            if (distance > 1) {
                const ux = dx / distance;
                const uy = dy / distance;
                const head = document.createElementNS(namespace, "polygon");
                head.setAttribute("points",
                    `${x},${y} ${x - ux * 7 - uy * 3},${y - uy * 7 + ux * 3} ` +
                    `${x - ux * 7 + uy * 3},${y - uy * 7 - ux * 3}`);
                head.setAttribute("fill", axis.color);
                group.appendChild(head);
            }
            const label = document.createElementNS(namespace, "text");
            const offset = distance > 1 ? 10 / distance : 0;
            label.setAttribute("x", x + dx * offset);
            label.setAttribute("y", y + dy * offset);
            label.setAttribute("fill", axis.color);
            label.setAttribute("font-size", "13");
            label.setAttribute("font-weight", "bold");
            label.setAttribute("text-anchor", "middle");
            label.setAttribute("dominant-baseline", "central");
            label.textContent = axis.label;
            group.appendChild(label);
            svg.appendChild(group);
        }
    }
    viewer.setViewChangeCallback(updateAxes);
    updateAxes(viewer.getView());
})();
""".replace("AXES_DATA", axes_data)


def _add_hover_labels(
    view: Any,
    mode: LabelMode,
    index_base: Literal[0, 1] = 0,
    hover_duration_ms: int = 100,
    property_strings: list[str] | None = None,
    property_name: str | None = None,
) -> None:
    """Configures interactive mouseover tooltips displaying atom info and optional values.

    Args:
        view: Active `py3Dmol.view` instance where hover callbacks will be bound.
        mode: Formatting mode for the atom identifier (`'symbol'`, `'index'`, or
            `'symbol_index'`).
        index_base: Starting integer for site numbering. Use `0` for Python/Pymatgen
            indexing or `1` for VASP/VESTA indexing. Defaults to 0.
        hover_duration_ms: Delay in milliseconds before the tooltip appears on hover.
            Defaults to 100.
        property_strings: Optional list of preformatted per-atom property strings
            indexed by 0-based model site index.
        property_name: Optional label prefix shown before the property value in the
            tooltip (e.g., `'Bader q'`).

    Raises:
        ValueError: If `index_base` is not 0 or 1, or if an unsupported `mode` is
            provided.
    """
    if index_base not in (0, 1):
        raise ValueError(f"'index_base' must be 0 or 1, got {index_base!r}.")

    if mode == "symbol":
        header_expr = "atom.elem"
    elif mode == "index":
        header_expr = f"String(atom.index + {index_base})"
    elif mode == "symbol_index":
        header_expr = f"atom.elem + (atom.index + {index_base})"
    else:
        raise ValueError(
            f"Unsupported hover label mode '{mode}'. "
            "Choose from: 'symbol', 'index', 'symbol_index', or bool."
        )

    if property_strings is not None:
        props_json = json.dumps(property_strings)
        prefix_json = json.dumps(f"{property_name}: " if property_name else "")
        text_logic = (
            f"var propValues = {props_json};"
            f"var prefix = {prefix_json};"
            f"var header = {header_expr};"
            "var val = propValues[atom.index];"
            "var text = (val !== undefined) ? (header + ' | ' + prefix + val) : header;"
        )
    else:
        text_logic = f"var text = {header_expr};"

    hover_callback = (
        "function(atom, viewer, event, container) {"
        "  if (!atom.label) {"
        f"    {text_logic}"
        "    atom.label = viewer.addLabel(text, {"
        "      position: atom,"
        "      backgroundColor: '#1F2328',"
        "      backgroundOpacity: 0.88,"
        "      fontColor: '#FFFFFF',"
        "      fontSize: 12,"
        "      borderThickness: 0,"
        "      inFront: true"
        "    });"
        "  }"
        "}"
    )

    unhover_callback = (
        "function(atom, viewer) {"
        "  if (atom.label) {"
        "    viewer.removeLabel(atom.label);"
        "    delete atom.label;"
        "  }"
        "}"
    )

    view.setHoverDuration(hover_duration_ms)
    view.setHoverable({}, True, hover_callback, unhover_callback)


def build_py3dmol_view(
    structure: Structure | Molecule,
    *,
    width: int = 600,
    height: int = 450,
    style: RenderStyle = "ball_and_stick",
    repeat: tuple[int, int, int] = (1, 1, 1),
    show_unit_cell: bool = True,
    element_colors: Mapping[str, str] | None = None,
    axes: AxesMode | None = "cartesian",
    axes_length: float = 2.0,
    hover_labels: bool | LabelMode = True,
    index_base: Literal[0, 1] = 0,
    property_values: Sequence[float] | np.ndarray | None = None,
    property_name: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric_clim: bool = False,
    value_format: str = ".2f",
) -> Any:
    """Builds an interactive py3Dmol viewer with optional per-atom property coloring.

    Args:
        structure: Periodic `Structure` or non-periodic `Molecule` to render.
        width: Viewer canvas width in pixels. Defaults to 600.
        height: Viewer canvas height in pixels. Defaults to 450.
        style: Molecular rendering preset (`'ball_and_stick'`, `'spacefill'`, or
            `'stick'`). Defaults to `'ball_and_stick'`.
        repeat: Supercell replication factors `(nx, ny, nz)` applied when `structure`
            is a periodic `Structure`. Ignored for `Molecule` objects. Defaults to
            `(1, 1, 1)`.
        show_unit_cell: Whether to draw the unit cell bounding box for periodic
            `Structure` objects. Defaults to True.
        element_colors: Optional mapping from element symbols to CSS or hex color
            strings. Ignored when `property_values` is provided.
        axes: Reference axes fixed in the viewer's lower-right corner (`'cartesian'`,
            `'cell'`, or `None` to hide). Defaults to `'cartesian'`.
        axes_length: Scale factor for the overlay arrows. Defaults to 2.0.
        hover_labels: Controls interactive mouseover tooltips on atoms. Pass `True`
            (equivalent to `'symbol_index'`) for symbol and index, `'symbol'` for
            element symbols only, `'index'` for site indices only, or `False` to
            disable hover tooltips. When `property_values` is provided, tooltips
            automatically append the hovered atom's property value. Defaults to True.
        index_base: Starting index for site numbering when `hover_labels` displays
            indices (`0` for Python/Pymatgen, `1` for VASP/VESTA). Defaults to 0.
        property_values: Optional 1D sequence or NumPy array of scalar values (one
            per site in `structure`) used to color atoms via `cmap` and display a
            right-side color scale.
        property_name: Display title for the color scale and hover tooltip prefix
            (e.g., `'Net Bader charge (e)'`).
        cmap: Matplotlib colormap name used when `property_values` is provided.
            Defaults to `'viridis'`.
        vmin: Minimum value for color scale normalization. Defaults to data minimum.
        vmax: Maximum value for color scale normalization. Defaults to data maximum.
        symmetric_clim: If True and `vmin`/`vmax` are None, sets symmetric color
            limits `[-max(|v|), +max(|v|)]` around zero. Useful for signed properties
            such as Bader charges or magnetic moments. Defaults to False.
        value_format: Python format specification for property numbers in tooltips
            and colorbar ticks (e.g., `'.2f'`, `'+.3f'`). Defaults to `'.2f'`.

    Returns:
        A `ColorbarView` (wrapping `py3Dmol.view` with a right-side color scale) when
        `property_values` is provided, or a plain `py3Dmol.view` instance otherwise.

    Raises:
        ImportError: If `py3Dmol` (or `matplotlib` when `property_values` is given)
            is not installed.
        ValueError: If `property_values` length does not match `len(structure)`, or
            if invalid rendering options are supplied.
    """
    try:
        import py3Dmol
    except ImportError as exc:
        raise ImportError(
            "py3Dmol is required for 3D visualization. "
            "Install it with: pip install py3Dmol"
        ) from exc

    expanded_values: np.ndarray | None = None
    if property_values is not None:
        raw_values = np.asarray(property_values, dtype=float)
        if raw_values.ndim != 1 or len(raw_values) != len(structure):
            raise ValueError(
                f"'property_values' must be a 1D array of length {len(structure)}, "
                f"got shape {raw_values.shape}."
            )
        if isinstance(structure, Structure) and repeat != (1, 1, 1):
            # Pymatgen Structure.__mul__ replicates each original site contiguously
            expanded_values = np.repeat(raw_values, int(np.prod(repeat)))
        else:
            expanded_values = raw_values

    view = py3Dmol.view(width=width, height=height)

    if isinstance(structure, Structure):
        target: Structure | Molecule = (
            structure * repeat if repeat != (1, 1, 1) else structure
        )
        view.addModel(target.to(fmt="cif"), "cif")
        if show_unit_cell:
            view.addUnitCell()
    else:
        target = structure
        view.addModel(target.to(fmt="xyz"), "xyz")

    styles_map: dict[str, dict[str, Any]] = {
        "ball_and_stick": {
            "sphere": {"scale": 0.30, "colorscheme": "Jmol"},
            "stick": {"radius": 0.14, "colorscheme": "Jmol"},
        },
        "spacefill": {
            "sphere": {"scale": 0.85, "colorscheme": "Jmol"},
        },
        "stick": {
            "stick": {"radius": 0.16, "colorscheme": "Jmol"},
        },
    }

    if style not in styles_map:
        raise ValueError(
            f"Unsupported style '{style}'. Choose from: {list(styles_map.keys())}."
        )

    base_style = styles_map[style]
    view.setStyle({}, base_style)

    colorbar_html: str | None = None
    property_strings: list[str] | None = None

    if expanded_values is not None:
        atom_colors, cmin, cmax, gradient_stops = _map_values_to_hex(
            expanded_values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            symmetric_clim=symmetric_clim,
        )
        for idx, hex_color in enumerate(atom_colors):
            site_style = {
                geom_type: {k: v for k, v in props.items() if k != "colorscheme"}
                | {"color": hex_color}
                for geom_type, props in base_style.items()
            }
            view.setStyle({"index": idx}, site_style)

        property_strings = [
            "N/A" if not np.isfinite(val) else f"{val:{value_format}}"
            for val in expanded_values
        ]
        colorbar_html = _build_colorbar_html(
            gradient_stops=gradient_stops,
            cmin=cmin,
            cmax=cmax,
            height=height,
            property_name=property_name,
            value_format=value_format,
        )
    elif element_colors:
        for symbol, color in element_colors.items():
            custom_style = {
                geom_type: {k: v for k, v in props.items() if k != "colorscheme"}
                | {"color": color}
                for geom_type, props in base_style.items()
            }
            view.setStyle({"elem": symbol}, custom_style)

    if axes is not None:
        _add_axes_gizmo(view, target=target, mode=axes, length=axes_length)

    if hover_labels or property_strings is not None:
        label_mode: LabelMode = (
            "symbol_index"
            if (hover_labels is True or hover_labels is False)
            else hover_labels
        )
        _add_hover_labels(
            view,
            mode=label_mode,
            index_base=index_base,
            property_strings=property_strings,
            property_name=property_name,
        )

    view.zoomTo()
    if colorbar_html is not None:
        return ColorbarView(view=view, colorbar_html=colorbar_html)
    return view