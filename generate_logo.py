#!/usr/bin/env python3
"""
Generate the AMJax logo.

Run from the repository root with:

    uv run --group dev python logo.py

Outputs:
    assets/logo.svg
    assets/logo.png

Dependencies:
    numpy
    scipy
    matplotlib
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "amjax-logo"

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.path import Path as MplPath
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from scipy.spatial import Delaunay


def pick_font():
    available = {f.name for f in fontManager.ttflist}
    preferences = [
        "Clear Sans",
        "Noto Sans Display",
        "Noto Sans",
        "Liberation Sans",
        "Nimbus Sans",
        "DejaVu Sans",
    ]
    for name in preferences:
        if name in available:
            return name
    return "DejaVu Sans"


def text_path_raw(text, font_name, weight="bold"):
    fp = FontProperties(family=font_name, weight=weight)
    return TextPath((0, 0), text, size=1.0, prop=fp, usetex=False)


def path_width_height(path):
    v = path.vertices
    xmin, ymin = v.min(axis=0)
    xmax, ymax = v.max(axis=0)
    return xmax - xmin, ymax - ymin


def make_letter_paths(text="AMJAX", height=3.25, gap=0.08, font_name=None):
    """
    Build separate paths for each letter, aligned as one word.
    Returning individual paths lets us fill AM/JAX differently and clip
    a mesh inside the A letters.
    """
    if font_name is None:
        font_name = pick_font()

    raws = [text_path_raw(ch, font_name, weight="bold") for ch in text]
    dims = [path_width_height(p) for p in raws]
    raw_heights = [h for _, h in dims]
    scale = height / max(raw_heights)

    widths = [w * scale for w, _ in dims]
    gap_abs = gap * height
    total_w = sum(widths) + gap_abs * (len(text) - 1)

    paths = []
    x_cursor = -total_w / 2
    for ch, raw, w in zip(text, raws, widths):
        verts = raw.vertices.copy()
        codes = raw.codes.copy()

        xmin, ymin = verts.min(axis=0)
        xmax, ymax = verts.max(axis=0)

        # Center vertically and place horizontally.
        verts[:, 0] = (verts[:, 0] - xmin) * scale + x_cursor
        verts[:, 1] = (verts[:, 1] - (ymin + ymax) / 2) * scale

        paths.append(MplPath(verts, codes))
        x_cursor += w + gap_abs

    full = MplPath.make_compound_path(*paths)
    return paths, full, font_name


def path_to_polylines(path, samples_per_curve=18):
    dense = path.interpolated(samples_per_curve)
    polys = []
    current = []
    for vert, code in zip(dense.vertices, dense.codes):
        if code == MplPath.MOVETO:
            if len(current) > 1:
                polys.append(np.array(current))
            current = [vert]
        elif code == MplPath.LINETO:
            current.append(vert)
        elif code == MplPath.CLOSEPOLY:
            if len(current) > 1:
                current.append(current[0])
                polys.append(np.array(current))
            current = []
        else:
            current.append(vert)
    if len(current) > 1:
        polys.append(np.array(current))
    cleaned = []
    for p in polys:
        if len(p) >= 4:
            span = np.ptp(p, axis=0)
            if span[0] > 0.015 and span[1] > 0.015:
                cleaned.append(p)
    return cleaned


def sample_polyline(poly, spacing):
    pts = []
    for a, b in zip(poly[:-1], poly[1:]):
        L = np.linalg.norm(b - a)
        n = max(2, int(np.ceil(L / spacing)))
        t = np.linspace(0, 1, n, endpoint=False)
        pts.append(a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None])
    return np.vstack(pts) if pts else np.empty((0, 2))


def sample_rectangle_boundary(xmin, xmax, ymin, ymax, spacing):
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)
    return np.vstack([
        np.c_[xs, np.full_like(xs, ymax)],
        np.c_[xs, np.full_like(xs, ymin)],
        np.c_[np.full_like(ys, xmin), ys],
        np.c_[np.full_like(ys, xmax), ys],
    ])


def distance_to_polylines(P, polylines):
    best = np.full(P.shape[0], np.inf)
    for poly in polylines:
        A = poly[:-1]
        B = poly[1:]
        AB = B - A
        denom = np.sum(AB * AB, axis=1) + 1e-12
        for start in range(0, P.shape[0], 1500):
            Q = P[start:start + 1500]
            AP = Q[:, None, :] - A[None, :, :]
            t = np.sum(AP * AB[None, :, :], axis=2) / denom[None, :]
            t = np.clip(t, 0.0, 1.0)
            C = A[None, :, :] + t[:, :, None] * AB[None, :, :]
            d = np.sqrt(np.min(np.sum((Q[:, None, :] - C) ** 2, axis=2), axis=1))
            best[start:start + 1500] = np.minimum(best[start:start + 1500], d)
    return best


def generate_outer_points(word_path, polylines, bounds, seed=51):
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds

    n_candidates = 11500
    Q = np.c_[
        rng.uniform(xmin, xmax, n_candidates),
        rng.uniform(ymin, ymax, n_candidates),
    ]
    Q = Q[~word_path.contains_points(Q)]
    dist = distance_to_polylines(Q, polylines)

    # Modern/coarser mesh: most density close to the letter boundary.
    keep_prob = 0.030 + 0.46 * np.exp(-(dist / 0.50) ** 2)
    keep_prob += 0.025 * np.exp(-(dist / 1.25) ** 2)
    keep_prob = np.clip(keep_prob, 0.02, 0.55)
    kept = Q[rng.random(Q.shape[0]) < keep_prob]

    # Blue-noise-ish grid thinning.
    order = rng.permutation(kept.shape[0])
    pts = []
    occupied = set()
    cell = 0.155
    for idx in order:
        x, y = kept[idx]
        key = (int(np.floor((x - xmin) / cell)), int(np.floor((y - ymin) / cell)))
        if key in occupied:
            continue
        occupied.add(key)
        pts.append([x, y])
    pts = np.array(pts)

    boundary = np.vstack([sample_polyline(poly, spacing=0.085) for poly in polylines])
    outer = sample_rectangle_boundary(xmin, xmax, ymin, ymax, spacing=0.42)
    P = np.vstack([pts, boundary, outer])
    return np.unique(np.round(P, 4), axis=0)


def build_outer_mesh(P, word_path, polylines, bounds):
    xmin, xmax, ymin, ymax = bounds
    tri = Delaunay(P)
    simplices = tri.simplices
    V = P[simplices]
    centroids = V.mean(axis=1)

    inside_letters = word_path.contains_points(centroids)
    inside_box = (
        (centroids[:, 0] >= xmin) &
        (centroids[:, 0] <= xmax) &
        (centroids[:, 1] >= ymin) &
        (centroids[:, 1] <= ymax)
    )

    edge_lengths = np.stack([
        np.linalg.norm(V[:, 0] - V[:, 1], axis=1),
        np.linalg.norm(V[:, 1] - V[:, 2], axis=1),
        np.linalg.norm(V[:, 2] - V[:, 0], axis=1),
    ], axis=1)
    max_edge = edge_lengths.max(axis=1)

    dist = distance_to_polylines(centroids, polylines)
    threshold = 0.55 + 1.05 * (1.0 - np.exp(-(dist / 1.0) ** 2))
    keep = (~inside_letters) & inside_box & (max_edge < threshold)
    return simplices[keep]


def unique_edges(simplices):
    E = np.vstack([
        simplices[:, [0, 1]],
        simplices[:, [1, 2]],
        simplices[:, [2, 0]],
    ])
    E = np.sort(E, axis=1)
    return np.unique(E, axis=0)


def generate_inner_mesh_for_path(path, seed=0):
    """Triangulate points inside one letter, used for the two A letters."""
    rng = np.random.default_rng(seed)
    v = path.vertices
    xmin, ymin = v.min(axis=0)
    xmax, ymax = v.max(axis=0)

    Q = np.c_[
        rng.uniform(xmin, xmax, 2600),
        rng.uniform(ymin, ymax, 2600),
    ]
    Q = Q[path.contains_points(Q)]

    # Coarse thinning.
    if len(Q) == 0:
        return None, None
    order = rng.permutation(Q.shape[0])
    pts = []
    occupied = set()
    cell = 0.13
    for idx in order:
        x, y = Q[idx]
        key = (int((x - xmin) / cell), int((y - ymin) / cell))
        if key in occupied:
            continue
        occupied.add(key)
        pts.append([x, y])
    pts = np.array(pts)

    # Add letter boundary points.
    polylines = path_to_polylines(path, samples_per_curve=18)
    boundary = np.vstack([sample_polyline(poly, spacing=0.09) for poly in polylines])
    P = np.unique(np.round(np.vstack([pts, boundary]), 4), axis=0)

    if len(P) < 3:
        return None, None

    tri = Delaunay(P)
    simplices = tri.simplices
    V = P[simplices]
    centroids = V.mean(axis=1)
    inside = path.contains_points(centroids)

    lengths = np.stack([
        np.linalg.norm(V[:, 0] - V[:, 1], axis=1),
        np.linalg.norm(V[:, 1] - V[:, 2], axis=1),
        np.linalg.norm(V[:, 2] - V[:, 0], axis=1),
    ], axis=1)
    keep = inside & (lengths.max(axis=1) < 0.55)
    return P, unique_edges(simplices[keep])


def generate_logo(
    output_dir=None,
    name="logo",
    width_px=1800,
    height_px=1050,
    seed=51,
):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "assets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{name}.svg"
    png_path = output_dir / f"{name}.png"

    letter_paths, word_path, font_name = make_letter_paths("AMJAX", height=3.25, gap=0.02)
    polylines = path_to_polylines(word_path, samples_per_curve=18)
    bounds = (-7.25, 7.25, -4.65, 4.00)

    P = generate_outer_points(word_path, polylines, bounds, seed=seed)
    simplices = build_outer_mesh(P, word_path, polylines, bounds)
    edges = unique_edges(simplices)

    # JAX-like colors.
    jax_blue = "#4285F4"
    jax_green = "#34A853"
    purple = "#6F42C1"
    purple_light = "#9C7AE6"
    subtitle_color = "#351A73"

    fig = plt.figure(figsize=(width_px / 200, height_px / 200), dpi=200)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((1, 1, 1, 0))

    # Outer FEM mesh: clean, no filled triangles, modern linework.
    lc = LineCollection(
        P[edges],
        colors=purple,
        linewidths=0.90,
        alpha=0.66,
        capstyle="round",
        joinstyle="round",
        zorder=1,
    )
    ax.add_collection(lc)

    # Sparse nodes close to letter boundary.
    dist_pts = distance_to_polylines(P, polylines)
    node_keep = dist_pts < 0.18
    ax.scatter(
        P[node_keep, 0],
        P[node_keep, 1],
        s=5.0,
        c=purple_light,
        alpha=0.78,
        linewidths=0,
        zorder=2,
    )

    # Filled letters with NO contour.
    for i, path in enumerate(letter_paths):
        fill = jax_blue if i < 2 else jax_green
        ax.add_patch(PathPatch(
            path,
            facecolor=fill,
            edgecolor="none",
            linewidth=0,
            alpha=0.98,
            zorder=4,
        ))

    # Mesh inside every letter, clipped by being triangulated inside each letter.
    # AM uses a light blue mesh; JAX uses a light green mesh.
    inner_colors = ["#BFD7FF", "#BFD7FF", "#BFF0C8", "#BFF0C8", "#BFF0C8"]
    for idx, color in enumerate(inner_colors):
        Pin, Ein = generate_inner_mesh_for_path(letter_paths[idx], seed=seed + idx + 200)
        if Pin is not None and Ein is not None and len(Ein) > 0:
            inner = LineCollection(
                Pin[Ein],
                colors=color,
                linewidths=0.70,
                alpha=0.80,
                capstyle="round",
                joinstyle="round",
                zorder=5,
            )
            ax.add_collection(inner)
            ax.scatter(Pin[:, 0], Pin[:, 1], s=1.9, c=color, alpha=0.54, linewidths=0, zorder=5)

    # Subtitle: modern, more visible, but without heavy box/background.
    fp_subtitle = FontProperties(family=font_name, weight="bold")
    txt = ax.text(
        0.0,
        -4.20,
        "Algebraic multigrid solvers in JAX",
        ha="center",
        va="center",
        color=subtitle_color,
        fontsize=31,
        fontproperties=fp_subtitle,
        alpha=1.0,
        zorder=8,
    )
    txt.set_path_effects([
        pe.Stroke(linewidth=4.5, foreground="white", alpha=0.95),
        pe.Normal(),
    ])

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(svg_path, format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(png_path, format="png", transparent=True)
    plt.close(fig)

    print(f"Using font: {font_name}")
    print(f"Outer mesh points: {len(P)}")
    print(f"Outer mesh triangles: {len(simplices)}")
    return svg_path, png_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AMJax logo assets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
        help="Directory where logo.svg and logo.png are written.",
    )
    parser.add_argument(
        "--name",
        default="logo",
        help="Output basename without extension.",
    )
    parser.add_argument("--width-px", type=int, default=1800)
    parser.add_argument("--height-px", type=int, default=1050)
    parser.add_argument("--seed", type=int, default=51)
    return parser.parse_args()


def main():
    args = parse_args()
    svg, png = generate_logo(
        output_dir=args.output_dir,
        name=args.name,
        width_px=args.width_px,
        height_px=args.height_px,
        seed=args.seed,
    )
    print(svg)
    print(png)


if __name__ == "__main__":
    main()
