from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from manim import *


config.frame_width = 16
config.frame_height = 9
config.pixel_width = 1920
config.pixel_height = 1080


class Animacion1GPPriorKernel(Scene):
    """GP prior samples and RBF length-scale, in a clean mathematical style."""

    X_MIN = -3.2
    X_MAX = 3.2
    Y_MIN = -2.55
    Y_MAX = 2.55
    N_POINTS = 210
    N_SAMPLES = 5

    BG = "#05070d"
    AXIS = "#6f788a"
    MUTED = "#aeb6c6"
    WHITE = "#f2f4fb"
    BLUE = "#58c4dd"
    GREEN = "#83c167"
    YELLOW = "#f4d35e"
    PINK = "#ff6f91"
    PURPLE = "#9a7ff0"

    def construct(self) -> None:
        self.camera.background_color = self.BG

        xs = np.linspace(self.X_MIN, self.X_MAX, self.N_POINTS)
        rng = np.random.default_rng(12)
        normals = rng.normal(size=(self.N_SAMPLES, self.N_POINTS))

        ell_short = 0.30
        ell_long = 1.15
        x0 = 0.85

        axes = self._axes()
        axis_labels = self._axis_labels(axes)
        mean_line = self._mean_line(axes)
        mean_label = self._latex(r"$m(x)=0$", height=0.23, color=self.MUTED)
        mean_label.move_to(axes.c2p(self.X_MIN + 0.45, 0.22)).set_z_index(8)

        prior_label = self._latex(
            r"$f \sim \mathcal{GP}\!\left(0,\,k_\ell\right)$",
            height=0.48,
            color=self.WHITE,
        )
        prior_label.to_corner(UL, buff=0.48)

        ell_label_short = self._latex(
            rf"$\ell={ell_short:.2f}$",
            height=0.48,
            color=self.YELLOW,
        )
        ell_label_long = self._latex(
            rf"$\ell={ell_long:.2f}$",
            height=0.48,
            color=self.YELLOW,
        )
        ell_label_short.to_corner(UR, buff=0.62)
        ell_label_long.move_to(ell_label_short)

        short_curves = self._sample_curves(axes, xs, normals, ell_short)
        long_curves = self._sample_curves(axes, xs, normals, ell_long)

        x0_marker = self._x0_marker(axes, x0)
        bump_short = self._kernel_bump(axes, xs, x0, ell_short)
        bump_long = self._kernel_bump(axes, xs, x0, ell_long)
        bump_label = self._latex(r"$k_\ell(x,x_0)$", height=0.27, color=self.YELLOW)
        bump_label.move_to(axes.c2p(x0 + 0.95, self.Y_MIN + 1.15)).set_z_index(8)

        kernel_formula = self._latex(
            r"$k_\ell(x,x')=\sigma^2\exp\!\left(-\frac{(x-x')^2}{2\ell^2}\right)$",
            height=0.42,
            color=self.MUTED,
        )
        kernel_formula.next_to(prior_label, DOWN, buff=0.18).align_to(prior_label, LEFT)

        conclusion = self._latex(
            r"$\ell \uparrow \quad\Longrightarrow\quad "
            r"\mathrm{correlaci\acute{o}n}\uparrow \quad\Longrightarrow\quad "
            r"\mathrm{suavidad}\uparrow$",
            height=0.38,
            color=self.WHITE,
        )
        conclusion.to_edge(DOWN, buff=0.45)

        self.play(
            FadeIn(prior_label, shift=DOWN * 0.15),
            Create(axes),
            FadeIn(axis_labels),
            run_time=1.3,
        )
        self.play(Create(mean_line), FadeIn(mean_label), run_time=0.75)
        self.play(
            FadeIn(ell_label_short, shift=LEFT * 0.1),
            LaggedStart(*(Create(curve) for curve in short_curves), lag_ratio=0.16),
            run_time=3.0,
        )
        self.wait(0.4)

        self.play(
            short_curves.animate.set_stroke(opacity=0.42),
            FadeIn(kernel_formula, shift=DOWN * 0.1),
            run_time=0.9,
        )
        self.play(
            Create(x0_marker[0]),
            GrowFromCenter(x0_marker[1]),
            FadeIn(x0_marker[2], shift=UP * 0.06),
            FadeIn(bump_short),
            FadeIn(bump_label, shift=UP * 0.08),
            run_time=1.5,
        )
        self.wait(0.4)

        self.play(short_curves.animate.set_stroke(opacity=1.0), run_time=0.45)
        self.play(
            Transform(short_curves, long_curves, run_time=3.4, rate_func=smooth),
            Transform(bump_short, bump_long, run_time=3.4, rate_func=smooth),
            AnimationGroup(
                FadeOut(ell_label_short, shift=UP * 0.08, run_time=0.45),
                FadeIn(ell_label_long, shift=UP * 0.08, run_time=0.45),
                lag_ratio=0,
            ),
        )
        self.wait(0.4)

        self.play(
            FadeOut(bump_label),
            FadeOut(bump_short),
            FadeOut(x0_marker),
            FadeOut(kernel_formula),
            run_time=0.8,
        )
        self.play(FadeIn(conclusion, shift=UP * 0.1), run_time=0.9)
        self.wait(1.6)

    def _axes(self) -> Axes:
        axes = Axes(
            x_range=[self.X_MIN, self.X_MAX, 1],
            y_range=[self.Y_MIN, self.Y_MAX, 1],
            x_length=12.4,
            y_length=5.65,
            axis_config={
                "color": self.AXIS,
                "stroke_width": 2,
                "include_tip": False,
                "include_ticks": True,
                "tick_size": 0.075,
            },
            tips=False,
        )
        axes.move_to(DOWN * 0.05)
        axes.set_z_index(1)
        return axes

    def _axis_labels(self, axes: Axes) -> VGroup:
        x_label = self._latex(r"$x$", height=0.26, color=self.MUTED)
        y_label = self._latex(r"$f(x)$", height=0.27, color=self.MUTED)
        x_label.next_to(axes.x_axis, RIGHT, buff=0.12)
        y_label.next_to(axes.y_axis, UP, buff=0.14)
        return Group(x_label, y_label).set_z_index(5)

    def _mean_line(self, axes: Axes) -> DashedLine:
        line = DashedLine(
            axes.c2p(self.X_MIN, 0),
            axes.c2p(self.X_MAX, 0),
            dash_length=0.12,
            color=self.MUTED,
            stroke_width=2,
        )
        line.set_opacity(0.56)
        line.set_z_index(2)
        return line

    def _sample_curves(
        self,
        axes: Axes,
        xs: np.ndarray,
        normals: np.ndarray,
        length_scale: float,
    ) -> VGroup:
        covariance = self._rbf_kernel(xs, xs, length_scale) + 1e-6 * np.eye(len(xs))
        cholesky = np.linalg.cholesky(covariance)
        samples = normals @ cholesky.T

        colors = [self.BLUE, self.YELLOW, self.GREEN, self.PINK, self.PURPLE]
        curves = VGroup()
        for sample, color in zip(samples, colors):
            sample = np.clip(0.58 * sample, self.Y_MIN + 0.12, self.Y_MAX - 0.12)
            points = [axes.c2p(float(x), float(y)) for x, y in zip(xs, sample)]
            curve = VMobject(color=color, stroke_width=3.1)
            curve.set_points_smoothly(points)
            curve.set_fill(opacity=0)
            curve.set_z_index(4)
            curves.add(curve)
        return curves

    def _x0_marker(self, axes: Axes, x0: float) -> VGroup:
        vertical = DashedLine(
            axes.c2p(x0, self.Y_MIN + 0.2),
            axes.c2p(x0, self.Y_MAX - 0.2),
            dash_length=0.11,
            color=self.YELLOW,
            stroke_width=2,
        )
        vertical.set_opacity(0.72)
        dot = Dot(axes.c2p(x0, 0), radius=0.055, color=self.YELLOW)
        label = self._latex(r"$x_0$", height=0.25, color=self.YELLOW)
        label.next_to(dot, UP + RIGHT, buff=0.12)
        return Group(vertical, dot, label).set_z_index(7)

    def _kernel_bump(
        self,
        axes: Axes,
        xs: np.ndarray,
        x0: float,
        length_scale: float,
    ) -> VGroup:
        y_base = self.Y_MIN + 0.35
        amplitude = 0.86
        values = np.exp(-0.5 * ((xs - x0) / length_scale) ** 2)
        upper = [axes.c2p(float(x), float(y_base + amplitude * v)) for x, v in zip(xs, values)]
        lower = [axes.c2p(float(x), y_base) for x in xs[::-1]]

        fill = Polygon(*upper, *lower, stroke_width=0)
        fill.set_fill(self.YELLOW, opacity=0.17)
        fill.set_z_index(2)

        line = VMobject(color=self.YELLOW, stroke_width=3.0)
        line.set_points_smoothly(upper)
        line.set_z_index(3)

        baseline = Line(
            axes.c2p(self.X_MIN, y_base),
            axes.c2p(self.X_MAX, y_base),
            color=self.YELLOW,
            stroke_width=1.4,
        )
        baseline.set_opacity(0.22)
        baseline.set_z_index(2)

        return VGroup(fill, baseline, line)

    def _latex(self, tex: str, height: float, color: str) -> ImageMobject:
        path = self._latex_png(tex, color=color)
        mob = ImageMobject(str(path))
        mob.set_height(height)
        return mob

    @staticmethod
    def _latex_png(tex: str, color: str) -> Path:
        cache_dir = Path(__file__).resolve().parent / "assets" / "latex_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{tex}|{color}".encode("utf-8")).hexdigest()[:16]
        path = cache_dir / f"{key}.png"
        if path.exists():
            return path

        mpl.rcParams.update(
            {
                "mathtext.fontset": "cm",
                "font.family": "serif",
                "savefig.transparent": True,
            }
        )

        fig = plt.figure(figsize=(4.8, 1.0), dpi=320)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.5, tex, ha="center", va="center", color=color, fontsize=42)
        fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)

        with Image.open(path) as image:
            alpha = image.getchannel("A")
            bbox = alpha.getbbox()
            if bbox:
                padding = 8
                left = max(bbox[0] - padding, 0)
                upper = max(bbox[1] - padding, 0)
                right = min(bbox[2] + padding, image.width)
                lower = min(bbox[3] + padding, image.height)
                image.crop((left, upper, right, lower)).save(path)
        return path

    @staticmethod
    def _rbf_kernel(x_left: np.ndarray, x_right: np.ndarray, length_scale: float) -> np.ndarray:
        distances = x_left[:, None] - x_right[None, :]
        return np.exp(-0.5 * (distances / length_scale) ** 2)
