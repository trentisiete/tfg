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


class Animacion2GPPosterior(Scene):
    """GP posterior update with observations, matching the style of animation 1."""

    X_MIN = -3.2
    X_MAX = 3.2
    Y_MIN = -2.55
    Y_MAX = 2.55
    N_POINTS = 230

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
        ell = 0.85
        sigma_f = 1.0
        noise = 0.06
        x_obs = np.array([-2.55, -1.25, 0.15, 1.25, 2.35])
        y_obs = self._target_function(x_obs)

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

        posterior_label = self._latex(
            r"$f\,|\,\mathcal{D} \sim \mathcal{GP}\!\left(\mu_{\mathcal{D}},\,k_{\mathcal{D}}\right)$",
            height=0.48,
            color=self.WHITE,
        )
        posterior_label.move_to(prior_label).align_to(prior_label, LEFT)

        data_label = self._latex(r"$\mathcal{D}=\{(x_i,y_i)\}$", height=0.34, color=self.YELLOW)
        data_label.to_corner(UR, buff=0.62)

        update_equation = self._latex(
            r"$\sigma_{\mathcal{D}}(x)\downarrow\ \text{cerca de los datos}$",
            height=0.34,
            color=self.MUTED,
        )
        update_equation.to_edge(DOWN, buff=0.42)

        prior_mean, prior_std = self._posterior(xs, np.array([]), np.array([]), ell, sigma_f, noise)
        prior_band = self._band(axes, xs, prior_mean, prior_std, color=self.BLUE, opacity=0.10)
        prior_curve = self._curve(axes, xs, prior_mean, color=self.BLUE, width=2.4, opacity=0.8)
        prior_samples = self._prior_samples(axes, xs, ell, sigma_f)

        states = [self._posterior(xs, x_obs[:i], y_obs[:i], ell, sigma_f, noise) for i in range(1, len(x_obs) + 1)]
        bands = [self._band(axes, xs, mean, std, color=self.BLUE, opacity=0.16) for mean, std in states]
        means = [self._curve(axes, xs, mean, color=self.YELLOW, width=3.3) for mean, _ in states]
        dots = [self._observation_dot(axes, float(x), float(y)) for x, y in zip(x_obs, y_obs)]

        final_label = self._latex(
            r"$\mu_{\mathcal{D}}(x)\ \text{sigue los datos},\qquad "
            r"\sigma_{\mathcal{D}}(x)\ \text{mide lo que falta por conocer}$",
            height=0.34,
            color=self.WHITE,
        )
        final_label.to_edge(DOWN, buff=0.42)

        self.play(
            FadeIn(prior_label, shift=DOWN * 0.15),
            Create(axes),
            FadeIn(axis_labels),
            run_time=1.25,
        )
        self.play(Create(mean_line), FadeIn(mean_label), FadeIn(prior_band), run_time=1.0)
        self.play(
            FadeIn(prior_curve),
            LaggedStart(*(Create(curve) for curve in prior_samples), lag_ratio=0.17),
            run_time=2.5,
        )
        self.wait(0.25)

        self.play(
            prior_samples.animate.set_stroke(opacity=0.16),
            FadeOut(prior_label, shift=UP * 0.08),
            FadeOut(mean_label, shift=DOWN * 0.04),
            FadeIn(posterior_label, shift=UP * 0.08),
            FadeIn(data_label, shift=LEFT * 0.08),
            run_time=1.0,
        )

        current_band = prior_band
        current_mean = prior_curve
        for i, dot in enumerate(dots):
            self.play(GrowFromCenter(dot), run_time=0.45)
            self.play(
                Transform(current_band, bands[i]),
                Transform(current_mean, means[i]),
                run_time=1.25,
                rate_func=smooth,
            )
            if i == 1:
                self.play(FadeIn(update_equation, shift=UP * 0.08), run_time=0.65)
            self.wait(0.15)

        self.play(
            FadeOut(update_equation),
            FadeIn(final_label, shift=UP * 0.08),
            prior_samples.animate.set_stroke(opacity=0.05),
            run_time=0.9,
        )
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

    def _axis_labels(self, axes: Axes) -> Group:
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

    def _prior_samples(self, axes: Axes, xs: np.ndarray, ell: float, sigma_f: float) -> VGroup:
        rng = np.random.default_rng(21)
        normals = rng.normal(size=(4, len(xs)))
        covariance = self._rbf_kernel(xs, xs, ell, sigma_f) + 1e-6 * np.eye(len(xs))
        samples = normals @ np.linalg.cholesky(covariance).T
        colors = [self.BLUE, self.GREEN, self.PINK, self.PURPLE]
        curves = VGroup()
        for sample, color in zip(samples, colors):
            sample = np.clip(0.55 * sample, self.Y_MIN + 0.14, self.Y_MAX - 0.14)
            curve = self._curve(axes, xs, sample, color=color, width=2.6, opacity=0.52)
            curves.add(curve)
        return curves

    def _curve(
        self,
        axes: Axes,
        xs: np.ndarray,
        ys: np.ndarray,
        color: str,
        width: float,
        opacity: float = 1.0,
    ) -> VMobject:
        points = [axes.c2p(float(x), float(y)) for x, y in zip(xs, ys)]
        curve = VMobject(color=color, stroke_width=width)
        curve.set_points_smoothly(points)
        curve.set_fill(opacity=0)
        curve.set_stroke(opacity=opacity)
        curve.set_z_index(5)
        return curve

    def _band(
        self,
        axes: Axes,
        xs: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        color: str,
        opacity: float,
    ) -> Polygon:
        upper = np.clip(mean + 1.96 * std, self.Y_MIN + 0.08, self.Y_MAX - 0.08)
        lower = np.clip(mean - 1.96 * std, self.Y_MIN + 0.08, self.Y_MAX - 0.08)
        points_upper = [axes.c2p(float(x), float(y)) for x, y in zip(xs, upper)]
        points_lower = [axes.c2p(float(x), float(y)) for x, y in zip(xs[::-1], lower[::-1])]
        band = Polygon(*points_upper, *points_lower, stroke_width=0)
        band.set_fill(color, opacity=opacity)
        band.set_z_index(3)
        return band

    def _observation_dot(self, axes: Axes, x: float, y: float) -> Dot:
        dot = Dot(axes.c2p(x, y), radius=0.075, color=self.YELLOW)
        dot.set_stroke(self.WHITE, width=1.2, opacity=0.95)
        dot.set_z_index(8)
        return dot

    def _posterior(
        self,
        xs: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        ell: float,
        sigma_f: float,
        noise: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(x_train) == 0:
            return np.zeros_like(xs), np.full_like(xs, sigma_f)

        k_xx = self._rbf_kernel(x_train, x_train, ell, sigma_f) + noise**2 * np.eye(len(x_train))
        k_xs = self._rbf_kernel(x_train, xs, ell, sigma_f)
        alpha = np.linalg.solve(k_xx, y_train)
        mean = k_xs.T @ alpha
        v = np.linalg.solve(k_xx, k_xs)
        variance = np.maximum(sigma_f**2 - np.sum(k_xs * v, axis=0), 1e-8)
        return mean, np.sqrt(variance)

    @staticmethod
    def _target_function(x: np.ndarray) -> np.ndarray:
        return 0.62 * np.sin(1.35 * x) + 0.22 * np.cos(2.2 * x) - 0.12 * x

    @staticmethod
    def _rbf_kernel(x_left: np.ndarray, x_right: np.ndarray, ell: float, sigma_f: float) -> np.ndarray:
        distances = x_left[:, None] - x_right[None, :]
        return sigma_f**2 * np.exp(-0.5 * (distances / ell) ** 2)

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

        fig = plt.figure(figsize=(6.4, 1.0), dpi=320)
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
