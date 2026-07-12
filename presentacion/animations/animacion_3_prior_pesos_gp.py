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


class Animacion3PriorPesosGP(Scene):
    """Bridge from a Gaussian prior over linear weights to the GP view."""

    BG = "#05070d"
    AXIS = "#6f788a"
    MUTED = "#aeb6c6"
    WHITE = "#f2f4fb"
    BLUE = "#58c4dd"
    GREEN = "#83c167"
    YELLOW = "#f4d35e"
    PINK = "#ff6f91"
    PURPLE = "#9a7ff0"

    W0_MIN = -2.4
    W0_MAX = 3.1
    W1_MIN = -1.75
    W1_MAX = 1.75
    X_MIN = -2.35
    X_MAX = 2.35
    Y_MIN = -3.4
    Y_MAX = 4.35

    def construct(self) -> None:
        self.camera.background_color = self.BG

        rng = np.random.default_rng(4)
        prior_mean = np.array([0.0, 0.0])
        prior_cov = np.array([[0.55, -0.10], [-0.10, 0.28]])
        obs_x = 1.0
        obs_y = 2.0
        noise = 0.18
        xs = np.linspace(self.X_MIN, self.X_MAX, 190)
        colors = [
            self.BLUE,
            self.YELLOW,
            self.GREEN,
            self.PINK,
            self.PURPLE,
            self.BLUE,
            self.GREEN,
            self.YELLOW,
            self.PINK,
            self.PURPLE,
            self.BLUE,
            self.GREEN,
        ]

        weights = rng.multivariate_normal(prior_mean, prior_cov, 12)
        dense_weights = rng.multivariate_normal(prior_mean, prior_cov, 150)
        x_design = np.array([[1.0, obs_x]])
        post_mean, post_cov = self._posterior_weights(prior_mean, prior_cov, x_design, np.array([obs_y]), noise)
        posterior_weights = rng.multivariate_normal(post_mean, post_cov, len(weights))
        posterior_dense = rng.multivariate_normal(post_mean, post_cov, len(dense_weights))

        param_axes = self._param_axes()
        param_labels = self._param_axis_labels(param_axes)
        function_axes = self._function_axes()
        function_labels = self._function_axis_labels(function_axes)

        prior_cloud = self._weight_cloud(param_axes, dense_weights, self.BLUE, opacity=0.30, radius=0.018)
        prior_contours = self._gaussian_contours(param_axes, prior_mean, prior_cov, self.BLUE)
        weight_dots = self._weight_dots(param_axes, weights, colors)
        prior_lines = self._line_family(function_axes, xs, weights, colors, opacity=0.76)

        title = self._latex(r"$f(x)=w_0+w_1x$", height=0.46, color=self.WHITE)
        title.to_edge(UP, buff=0.34)
        prior_label = self._latex(r"$w\sim\mathcal{N}(m_0,\Sigma_0)$", height=0.32, color=self.MUTED)
        prior_label.move_to(param_axes.get_top() + UP * 0.58 + LEFT * 0.78)
        map_arrow = Arrow(
            start=param_axes.get_right() + RIGHT * 0.26,
            end=function_axes.get_left() + LEFT * 0.26,
            color=self.MUTED,
            stroke_width=3.0,
            buff=0.04,
            max_tip_length_to_length_ratio=0.06,
        ).set_opacity(0.72).set_z_index(8)

        self.play(
            FadeIn(title, shift=DOWN * 0.12),
            Create(param_axes),
            FadeIn(param_labels),
            Create(function_axes),
            FadeIn(function_labels),
            run_time=1.25,
        )
        self.play(FadeIn(prior_label, shift=DOWN * 0.08), FadeIn(prior_cloud), Create(prior_contours), run_time=1.1)
        self.play(FadeIn(map_arrow), run_time=0.35)
        self.play(
            LaggedStart(
                *(
                    AnimationGroup(GrowFromCenter(dot), Create(line), lag_ratio=0.0)
                    for dot, line in zip(weight_dots, prior_lines)
                ),
                lag_ratio=0.075,
            ),
            run_time=3.1,
        )
        self.wait(0.35)

        value_formula = self._latex(r"$f(x_*)=[\,1\;\;x_*\,]w$", height=0.43, color=self.WHITE)
        value_formula.move_to(title)
        x_star = 2.0
        x_star_marker = self._vertical_marker(function_axes, x_star, self.YELLOW, r"$x_*$")
        value_dots = self._value_dots(function_axes, weights, x_star, colors, radius=0.046)
        density_axes = self._density_axes()
        density_labels = self._density_axis_labels(density_axes)
        f_star_samples = dense_weights[:, 0] + x_star * dense_weights[:, 1]
        hist_bars = self._histogram_bars(density_axes, f_star_samples, self.YELLOW)
        density_curve = self._density_curve(density_axes, f_star_samples, self.YELLOW)
        gaussian_value = self._latex(
            r"$f(x_*)\sim\mathcal{N}\!\left(\mu(x_*),\sigma^2(x_*)\right)$",
            height=0.27,
            color=self.WHITE,
        )
        gaussian_value.move_to(density_axes.get_top() + UP * 0.52 + LEFT * 0.20)

        self.play(
            FadeOut(title, shift=UP * 0.04),
            FadeIn(value_formula, shift=UP * 0.04),
            FadeOut(prior_label, shift=UP * 0.06),
            FadeOut(map_arrow),
            prior_cloud.animate.set_opacity(0.10),
            prior_contours.animate.set_stroke(opacity=0.18),
            weight_dots.animate.set_opacity(0.20),
            prior_lines.animate.set_stroke(opacity=0.42),
            run_time=0.95,
        )
        title = value_formula
        self.play(
            Create(x_star_marker[0]),
            GrowFromCenter(x_star_marker[1]),
            FadeIn(x_star_marker[2], shift=UP * 0.05),
            LaggedStart(*(GrowFromCenter(dot) for dot in value_dots), lag_ratio=0.045),
            run_time=1.5,
        )
        self.play(
            FadeOut(param_axes),
            FadeOut(param_labels),
            FadeOut(prior_cloud),
            FadeOut(prior_contours),
            FadeOut(weight_dots),
            FadeIn(density_axes),
            FadeIn(density_labels),
            FadeIn(gaussian_value, shift=DOWN * 0.06),
            run_time=0.85,
        )
        self.play(LaggedStart(*(GrowFromEdge(bar, DOWN) for bar in hist_bars), lag_ratio=0.025), run_time=1.35)
        self.play(Create(density_curve), run_time=1.1)
        self.wait(0.55)

        joint_formula = self._latex(
            r"$\mathbf{f}_X=Xw\quad\Longrightarrow\quad"
            r"\mathbf{f}_X\sim\mathcal{N}\!\left(Xm_0,\,X\Sigma_0X^{\top}\right)$",
            height=0.36,
            color=self.WHITE,
        )
        joint_formula.move_to(title)
        x_finite = np.array([0.0, 1.0, 2.0])
        finite_markers = Group(
            *[
                self._vertical_marker(function_axes, float(x), color, rf"$x_{i+1}$")
                for i, (x, color) in enumerate(zip(x_finite, [self.BLUE, self.GREEN, self.YELLOW]))
            ]
        )
        finite_value_dots = self._finite_value_dots(function_axes, weights[:8], x_finite, colors[:8])
        joint_panel = self._joint_vector_panel()

        self.play(
            FadeOut(title, shift=UP * 0.04),
            FadeIn(joint_formula, shift=UP * 0.04),
            FadeOut(gaussian_value),
            FadeOut(hist_bars),
            FadeOut(density_curve),
            FadeOut(density_axes),
            FadeOut(density_labels),
            FadeOut(x_star_marker),
            FadeOut(value_dots),
            run_time=0.85,
        )
        title = joint_formula
        self.play(
            FadeIn(joint_panel, shift=UP * 0.06),
            run_time=1.0,
        )
        self.play(
            LaggedStart(
                *(
                    AnimationGroup(Create(marker[0]), GrowFromCenter(marker[1]), FadeIn(marker[2], shift=UP * 0.04))
                    for marker in finite_markers
                ),
                lag_ratio=0.15,
            ),
            run_time=1.25,
        )
        self.play(LaggedStart(*(FadeIn(group) for group in finite_value_dots), lag_ratio=0.05), run_time=1.35)
        self.wait(0.55)

        likelihood_formula = self._latex(r"$y\,|\,X,w\sim\mathcal{N}\!\left(Xw,\sigma_n^2I\right)$", height=0.42, color=self.WHITE)
        likelihood_formula.move_to(title)
        likelihood_band = self._likelihood_band(param_axes, obs_x, obs_y, noise * 1.7)
        likelihood_label = self._latex(r"$p(y\,|\,x,w)$", height=0.28, color=self.YELLOW)
        likelihood_label.move_to(param_axes.c2p(2.12, 0.36)).set_z_index(20)
        obs_dot = Dot(function_axes.c2p(obs_x, obs_y), radius=0.085, color=self.YELLOW)
        obs_dot.set_stroke(self.WHITE, width=1.2, opacity=0.95).set_z_index(18)
        obs_label = self._latex(r"$(x,y)$", height=0.25, color=self.YELLOW)
        obs_label.next_to(obs_dot, UP + RIGHT, buff=0.10).set_z_index(18)
        line_emphasis = self._line_emphasis(weights, obs_x, obs_y, noise)

        self.play(
            FadeOut(title, shift=UP * 0.04),
            FadeIn(likelihood_formula, shift=UP * 0.04),
            FadeOut(joint_panel),
            FadeOut(finite_markers),
            FadeOut(finite_value_dots),
            FadeIn(param_axes),
            FadeIn(param_labels),
            FadeIn(prior_cloud.set_opacity(0.25)),
            FadeIn(prior_contours.set_stroke(opacity=0.60)),
            FadeIn(weight_dots.set_opacity(0.92)),
            run_time=0.95,
        )
        title = likelihood_formula
        self.play(GrowFromCenter(obs_dot), FadeIn(obs_label, shift=UP * 0.04), run_time=0.55)
        self.play(
            LaggedStart(
                *[
                    line.animate.set_color(color).set_stroke(opacity=opacity, width=width)
                    for line, (color, opacity, width) in zip(prior_lines, line_emphasis)
                ],
                lag_ratio=0.02,
            ),
            run_time=0.9,
        )
        self.play(FadeIn(likelihood_band[0]), Create(likelihood_band[1]), FadeIn(likelihood_label, shift=DOWN * 0.04), run_time=1.25)
        self.wait(0.45)

        posterior_formula = self._latex(r"$p(w\,|\,X,y)\propto p(y\,|\,X,w)\,p(w)$", height=0.42, color=self.WHITE)
        posterior_formula.move_to(title)
        posterior_cloud = self._weight_cloud(param_axes, posterior_dense, self.GREEN, opacity=0.38, radius=0.018)
        posterior_contours = self._gaussian_contours(param_axes, post_mean, post_cov, self.GREEN, levels=(0.85, 1.45, 2.10))
        posterior_dots = self._weight_dots(param_axes, posterior_weights, colors)

        self.play(FadeOut(title, shift=UP * 0.04), FadeIn(posterior_formula, shift=UP * 0.04), run_time=0.6)
        title = posterior_formula
        self.play(
            Transform(prior_cloud, posterior_cloud),
            Transform(prior_contours, posterior_contours),
            Transform(weight_dots, posterior_dots),
            likelihood_band[0].animate.set_opacity(0.20),
            likelihood_band[1].animate.set_stroke(opacity=0.45),
            run_time=2.2,
            rate_func=smooth,
        )
        self.wait(0.50)

        posterior_function_formula = self._latex(r"$w\,|\,\mathcal{D}\quad\Longrightarrow\quad f\,|\,\mathcal{D}$", height=0.42, color=self.WHITE)
        posterior_function_formula.move_to(title)
        posterior_lines = self._line_family(function_axes, xs, posterior_weights, colors, opacity=0.72)
        post_mean_curve, post_band = self._linear_posterior_band(function_axes, xs, post_mean, post_cov)

        self.play(
            FadeOut(title, shift=UP * 0.04),
            FadeIn(posterior_function_formula, shift=UP * 0.04),
            FadeOut(likelihood_label),
            FadeOut(likelihood_band),
            prior_cloud.animate.set_opacity(0.20),
            prior_contours.animate.set_stroke(opacity=0.42),
            run_time=0.8,
        )
        title = posterior_function_formula
        self.play(Transform(prior_lines, posterior_lines), run_time=2.0, rate_func=smooth)
        self.play(FadeIn(post_band), Create(post_mean_curve), run_time=1.35)
        self.wait(0.45)

        feature_formula = self._latex(r"$f(x)=\phi(x)^{\top}w$", height=0.43, color=self.WHITE)
        feature_formula.move_to(title)
        flexible_curves = self._flexible_curves(function_axes, xs, posterior_weights, colors, obs_x)
        kernel_formula = self._latex(r"$k(x,x')=\phi(x)^{\top}\Sigma_p\,\phi(x')$", height=0.36, color=self.MUTED)
        kernel_formula.next_to(title, DOWN, buff=0.20)
        gp_formula = self._latex(r"$f\sim\mathcal{GP}(m,k)$", height=0.46, color=self.WHITE)
        gp_formula.to_edge(DOWN, buff=0.42)

        self.play(
            FadeOut(title, shift=UP * 0.04),
            FadeIn(feature_formula, shift=UP * 0.04),
            FadeOut(param_axes),
            FadeOut(param_labels),
            FadeOut(prior_cloud),
            FadeOut(prior_contours),
            FadeOut(weight_dots),
            FadeOut(post_mean_curve),
            FadeOut(post_band),
            FadeIn(kernel_formula, shift=DOWN * 0.08),
            run_time=1.0,
        )
        title = feature_formula
        self.play(Transform(prior_lines, flexible_curves), run_time=2.6, rate_func=smooth)
        self.play(FadeIn(gp_formula, shift=UP * 0.08), run_time=0.9)
        self.wait(2.0)

    def _param_axes(self) -> Axes:
        axes = Axes(
            x_range=[self.W0_MIN, self.W0_MAX, 1],
            y_range=[self.W1_MIN, self.W1_MAX, 1],
            x_length=5.05,
            y_length=3.85,
            axis_config={
                "color": self.AXIS,
                "stroke_width": 2,
                "include_tip": False,
                "include_ticks": True,
                "tick_size": 0.06,
            },
            tips=False,
        )
        axes.move_to(LEFT * 4.15 + DOWN * 0.38)
        axes.set_z_index(1)
        return axes

    def _function_axes(self) -> Axes:
        axes = Axes(
            x_range=[self.X_MIN, self.X_MAX, 1],
            y_range=[self.Y_MIN, self.Y_MAX, 1],
            x_length=6.35,
            y_length=4.85,
            axis_config={
                "color": self.AXIS,
                "stroke_width": 2,
                "include_tip": False,
                "include_ticks": True,
                "tick_size": 0.065,
            },
            tips=False,
        )
        axes.move_to(RIGHT * 3.45 + DOWN * 0.38)
        axes.set_z_index(1)
        return axes

    def _density_axes(self) -> Axes:
        axes = Axes(
            x_range=[-3.2, 3.2, 1],
            y_range=[0.0, 0.44, 0.1],
            x_length=4.95,
            y_length=2.60,
            axis_config={
                "color": self.AXIS,
                "stroke_width": 2,
                "include_tip": False,
                "include_ticks": True,
                "tick_size": 0.055,
            },
            tips=False,
        )
        axes.move_to(LEFT * 4.15 + DOWN * 0.38)
        axes.set_z_index(1)
        return axes

    def _joint_axes(self) -> Axes:
        axes = Axes(
            x_range=[-3.0, 3.0, 1],
            y_range=[-3.0, 3.0, 1],
            x_length=4.35,
            y_length=3.75,
            axis_config={
                "color": self.AXIS,
                "stroke_width": 2,
                "include_tip": False,
                "include_ticks": True,
                "tick_size": 0.055,
            },
            tips=False,
        )
        axes.move_to(LEFT * 4.15 + DOWN * 0.38)
        axes.set_z_index(1)
        return axes

    def _param_axis_labels(self, axes: Axes) -> Group:
        w0 = self._latex(r"$w_0$", height=0.24, color=self.MUTED)
        w1 = self._latex(r"$w_1$", height=0.24, color=self.MUTED)
        w0.next_to(axes.x_axis, RIGHT, buff=0.11)
        w1.next_to(axes.y_axis, UP, buff=0.12)
        return Group(w0, w1).set_z_index(8)

    def _function_axis_labels(self, axes: Axes) -> Group:
        x_label = self._latex(r"$x$", height=0.25, color=self.MUTED)
        y_label = self._latex(r"$f(x)$", height=0.27, color=self.MUTED)
        x_label.next_to(axes.x_axis, RIGHT, buff=0.11)
        y_label.next_to(axes.y_axis, UP, buff=0.12)
        return Group(x_label, y_label).set_z_index(8)

    def _density_axis_labels(self, axes: Axes) -> Group:
        z = self._latex(r"$f(x_*)$", height=0.24, color=self.MUTED)
        p = self._latex(r"$p$", height=0.22, color=self.MUTED)
        z.next_to(axes.x_axis, RIGHT, buff=0.10)
        p.next_to(axes.y_axis, UP, buff=0.11)
        return Group(z, p).set_z_index(8)

    def _joint_axis_labels(self, axes: Axes) -> Group:
        left = self._latex(r"$f(x_1)$", height=0.23, color=self.MUTED)
        right = self._latex(r"$f(x_3)$", height=0.23, color=self.MUTED)
        left.next_to(axes.x_axis, RIGHT, buff=0.10)
        right.next_to(axes.y_axis, UP, buff=0.11)
        return Group(left, right).set_z_index(8)

    def _joint_vector_panel(self) -> Group:
        center = LEFT * 4.15 + DOWN * 0.38

        vector_label = self._latex(r"$\mathbf{f}_X$", height=0.42, color=self.WHITE)
        vector_label.move_to(center + UP * 1.20 + LEFT * 1.00)

        entry_colors = [self.BLUE, self.GREEN, self.YELLOW]
        entries = Group()
        entry_y = [0.45, 0.00, -0.45]
        for index, (offset_y, color) in enumerate(zip(entry_y, entry_colors), start=1):
            dot = Dot(center + LEFT * 1.36 + UP * offset_y, radius=0.045, color=color)
            dot.set_stroke(self.WHITE, width=0.75, opacity=0.85).set_z_index(10)
            label = self._latex(rf"$f(x_{index})$", height=0.25, color=color)
            label.next_to(dot, RIGHT, buff=0.18).set_z_index(10)
            entries.add(Group(dot, label))

        left_bracket = VGroup(
            Line(center + LEFT * 1.62 + UP * 0.74, center + LEFT * 1.62 + DOWN * 0.74, color=self.MUTED, stroke_width=2.0),
            Line(center + LEFT * 1.62 + UP * 0.74, center + LEFT * 1.48 + UP * 0.74, color=self.MUTED, stroke_width=2.0),
            Line(center + LEFT * 1.62 + DOWN * 0.74, center + LEFT * 1.48 + DOWN * 0.74, color=self.MUTED, stroke_width=2.0),
        ).set_opacity(0.78).set_z_index(9)
        right_bracket = VGroup(
            Line(center + LEFT * 0.42 + UP * 0.74, center + LEFT * 0.42 + DOWN * 0.74, color=self.MUTED, stroke_width=2.0),
            Line(center + LEFT * 0.42 + UP * 0.74, center + LEFT * 0.56 + UP * 0.74, color=self.MUTED, stroke_width=2.0),
            Line(center + LEFT * 0.42 + DOWN * 0.74, center + LEFT * 0.56 + DOWN * 0.74, color=self.MUTED, stroke_width=2.0),
        ).set_opacity(0.78).set_z_index(9)

        k_label = self._latex(r"$K=X\Sigma_0X^{\top}$", height=0.30, color=self.MUTED)
        k_label.move_to(center + RIGHT * 0.95 + UP * 1.18)

        grid = VGroup()
        cell_size = 0.30
        cov_colors = [
            [self.BLUE, self.PURPLE, self.PURPLE],
            [self.PURPLE, self.GREEN, self.PURPLE],
            [self.PURPLE, self.PURPLE, self.YELLOW],
        ]
        opacities = [
            [0.50, 0.24, 0.18],
            [0.24, 0.50, 0.26],
            [0.18, 0.26, 0.50],
        ]
        grid_center = center + RIGHT * 0.95 + DOWN * 0.05
        for row in range(3):
            for col in range(3):
                square = Square(side_length=cell_size)
                square.set_stroke(self.AXIS, width=1.1, opacity=0.55)
                square.set_fill(cov_colors[row][col], opacity=opacities[row][col])
                square.move_to(grid_center + RIGHT * ((col - 1) * cell_size) + DOWN * ((row - 1) * cell_size))
                square.set_z_index(5)
                grid.add(square)

        return Group(vector_label, entries, left_bracket, right_bracket, k_label, grid).set_z_index(8)

    def _weight_cloud(self, axes: Axes, samples: np.ndarray, color: str, opacity: float, radius: float) -> VGroup:
        cloud = VGroup()
        for w0, w1 in samples:
            if self.W0_MIN <= w0 <= self.W0_MAX and self.W1_MIN <= w1 <= self.W1_MAX:
                dot = Dot(axes.c2p(float(w0), float(w1)), radius=radius, color=color)
                dot.set_opacity(opacity)
                dot.set_z_index(5)
                cloud.add(dot)
        return cloud

    def _weight_dots(self, axes: Axes, samples: np.ndarray, colors: list[str]) -> VGroup:
        dots = VGroup()
        for (w0, w1), color in zip(samples, colors):
            dot = Dot(axes.c2p(float(w0), float(w1)), radius=0.052, color=color)
            dot.set_stroke(self.WHITE, width=0.9, opacity=0.90)
            dot.set_z_index(12)
            dots.add(dot)
        return dots

    def _line_family(
        self,
        axes: Axes,
        xs: np.ndarray,
        weights: np.ndarray,
        colors: list[str],
        opacity: float,
    ) -> VGroup:
        lines = VGroup()
        for (w0, w1), color in zip(weights, colors):
            ys = w0 + w1 * xs
            line = self._curve(axes, xs, ys, color=color, width=2.75, opacity=opacity)
            line.set_z_index(6)
            lines.add(line)
        return lines

    def _flexible_curves(
        self,
        axes: Axes,
        xs: np.ndarray,
        weights: np.ndarray,
        colors: list[str],
        obs_x: float,
    ) -> VGroup:
        curves = VGroup()
        phases = np.linspace(0.15, 2.7, len(weights))
        amplitudes = np.linspace(0.16, 0.38, len(weights))
        for (w0, w1), color, phase, amplitude in zip(weights, colors, phases, amplitudes):
            base = w0 + w1 * xs
            bend = np.sin(2.05 * xs + phase) - np.sin(2.05 * obs_x + phase)
            bend += 0.45 * (np.cos(3.1 * xs - phase) - np.cos(3.1 * obs_x - phase))
            envelope = 0.55 + 0.45 * np.exp(-0.22 * (xs - obs_x) ** 2)
            ys = base + amplitude * envelope * bend
            curve = self._curve(axes, xs, ys, color=color, width=2.9, opacity=0.78)
            curve.set_z_index(6)
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
        return curve

    def _gaussian_contours(
        self,
        axes: Axes,
        mean: np.ndarray,
        covariance: np.ndarray,
        color: str,
        levels: tuple[float, ...] = (0.85, 1.45, 2.15),
    ) -> VGroup:
        eigvals, eigvecs = np.linalg.eigh(covariance)
        transform = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-8)))
        theta = np.linspace(0, TAU, 150)
        contours = VGroup()
        for index, level in enumerate(levels):
            values = mean[:, None] + level * transform @ np.vstack([np.cos(theta), np.sin(theta)])
            points = [axes.c2p(float(x), float(y)) for x, y in values.T]
            contour = VMobject(color=color, stroke_width=2.0 - 0.22 * index)
            contour.set_points_smoothly(points)
            contour.set_fill(opacity=0)
            contour.set_stroke(opacity=0.68 - 0.13 * index)
            contour.set_z_index(4)
            contours.add(contour)
        return contours

    def _vertical_marker(self, axes: Axes, x_value: float, color: str, label: str) -> VGroup:
        vertical = DashedLine(
            axes.c2p(x_value, self.Y_MIN + 0.18),
            axes.c2p(x_value, self.Y_MAX - 0.18),
            dash_length=0.10,
            color=color,
            stroke_width=1.8,
        )
        vertical.set_opacity(0.62).set_z_index(4)
        base = Dot(axes.c2p(x_value, 0), radius=0.045, color=color)
        base.set_z_index(9)
        label_mob = self._latex(label, height=0.23, color=color)
        label_mob.next_to(base, DOWN, buff=0.10).set_z_index(9)
        return Group(vertical, base, label_mob)

    def _value_dots(self, axes: Axes, weights: np.ndarray, x_value: float, colors: list[str], radius: float) -> VGroup:
        dots = VGroup()
        for (w0, w1), color in zip(weights, colors):
            y = float(w0 + w1 * x_value)
            dot = Dot(axes.c2p(x_value, y), radius=radius, color=color)
            dot.set_stroke(self.WHITE, width=0.75, opacity=0.75)
            dot.set_z_index(12)
            dots.add(dot)
        return dots

    def _finite_value_dots(self, axes: Axes, weights: np.ndarray, x_values: np.ndarray, colors: list[str]) -> VGroup:
        groups = VGroup()
        point_colors = [self.BLUE, self.GREEN, self.YELLOW]
        for weight, curve_color in zip(weights, colors):
            group = VGroup()
            for x_value, point_color in zip(x_values, point_colors):
                y = float(weight[0] + weight[1] * x_value)
                dot = Dot(axes.c2p(float(x_value), y), radius=0.036, color=point_color)
                dot.set_stroke(curve_color, width=1.0, opacity=0.85)
                dot.set_z_index(13)
                group.add(dot)
            groups.add(group)
        return groups

    def _histogram_bars(self, axes: Axes, samples: np.ndarray, color: str) -> VGroup:
        heights, edges = np.histogram(samples, bins=16, range=(-3.1, 3.1), density=True)
        bars = VGroup()
        for left, right, height in zip(edges[:-1], edges[1:], heights):
            bottom_left = axes.c2p(float(left), 0.0)
            bottom_right = axes.c2p(float(right), 0.0)
            top = axes.c2p(float((left + right) / 2), float(height))
            bar = Rectangle(
                width=max(bottom_right[0] - bottom_left[0] - 0.012, 0.01),
                height=max(top[1] - bottom_left[1], 0.01),
                stroke_width=0,
                fill_color=color,
                fill_opacity=0.35,
            )
            bar.move_to((bottom_left + bottom_right) / 2 + UP * (bar.height / 2))
            bar.set_z_index(3)
            bars.add(bar)
        return bars

    def _density_curve(self, axes: Axes, samples: np.ndarray, color: str) -> VMobject:
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        values = np.linspace(-3.2, 3.2, 180)
        density = np.exp(-0.5 * ((values - mean) / std) ** 2) / (std * np.sqrt(TAU))
        points = [axes.c2p(float(x), float(y)) for x, y in zip(values, density)]
        curve = VMobject(color=color, stroke_width=3.0)
        curve.set_points_smoothly(points)
        curve.set_fill(opacity=0)
        curve.set_z_index(5)
        return curve

    def _joint_cloud(self, axes: Axes, values: np.ndarray, color: str) -> VGroup:
        cloud = VGroup()
        for x_value, y_value in values:
            if -3.0 <= x_value <= 3.0 and -3.0 <= y_value <= 3.0:
                dot = Dot(axes.c2p(float(x_value), float(y_value)), radius=0.017, color=color)
                dot.set_opacity(0.32)
                dot.set_z_index(4)
                cloud.add(dot)
        return cloud

    def _likelihood_band(self, axes: Axes, x_value: float, y_value: float, half_width: float) -> VGroup:
        low = self._clipped_likelihood_segment(x_value, y_value - half_width)
        high = self._clipped_likelihood_segment(x_value, y_value + half_width)
        center = self._clipped_likelihood_segment(x_value, y_value)
        polygon_points = [axes.c2p(float(w0), float(w1)) for w0, w1 in high]
        polygon_points += [axes.c2p(float(w0), float(w1)) for w0, w1 in low[::-1]]
        fill = Polygon(*polygon_points, stroke_width=0)
        fill.set_fill(self.YELLOW, opacity=0.18)
        fill.set_z_index(3)
        line = Line(
            axes.c2p(float(center[0, 0]), float(center[0, 1])),
            axes.c2p(float(center[1, 0]), float(center[1, 1])),
            color=self.YELLOW,
            stroke_width=3.0,
        )
        line.set_opacity(0.86)
        line.set_z_index(5)
        return VGroup(fill, line)

    def _clipped_likelihood_segment(self, x_value: float, y_value: float) -> np.ndarray:
        candidates: list[np.ndarray] = []
        for w1 in [self.W1_MIN, self.W1_MAX]:
            w0 = y_value - x_value * w1
            if self.W0_MIN <= w0 <= self.W0_MAX:
                candidates.append(np.array([w0, w1]))
        for w0 in [self.W0_MIN, self.W0_MAX]:
            w1 = (y_value - w0) / x_value
            if self.W1_MIN <= w1 <= self.W1_MAX:
                candidates.append(np.array([w0, w1]))

        unique: list[np.ndarray] = []
        for point in candidates:
            if not any(np.linalg.norm(point - other) < 1e-6 for other in unique):
                unique.append(point)
        if len(unique) < 2:
            return np.array([[self.W0_MIN, y_value - self.W0_MIN], [self.W0_MAX, y_value - self.W0_MAX]])

        best_pair = (unique[0], unique[1])
        best_distance = -1.0
        for i, first in enumerate(unique):
            for second in unique[i + 1 :]:
                distance = float(np.linalg.norm(first - second))
                if distance > best_distance:
                    best_pair = (first, second)
                    best_distance = distance
        return np.array(sorted(best_pair, key=lambda point: point[1]))

    def _line_emphasis(self, weights: np.ndarray, x_value: float, y_value: float, noise: float) -> list[tuple[str, float, float]]:
        emphasis = []
        for w0, w1 in weights:
            residual = abs(float(w0 + w1 * x_value - y_value))
            score = float(np.exp(-0.5 * (residual / (noise * 2.0)) ** 2))
            if score > 0.58:
                emphasis.append((self.YELLOW, 0.94, 3.4))
            elif score > 0.20:
                emphasis.append((self.GREEN, 0.66, 2.8))
            else:
                emphasis.append((self.MUTED, 0.16, 2.0))
        return emphasis

    def _linear_posterior_band(self, axes: Axes, xs: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> tuple[VMobject, Polygon]:
        design = np.column_stack([np.ones_like(xs), xs])
        values = design @ mean
        variance = np.maximum(np.sum((design @ covariance) * design, axis=1), 1e-8)
        std = np.sqrt(variance)
        upper = np.clip(values + 1.96 * std, self.Y_MIN + 0.08, self.Y_MAX - 0.08)
        lower = np.clip(values - 1.96 * std, self.Y_MIN + 0.08, self.Y_MAX - 0.08)
        upper_points = [axes.c2p(float(x), float(y)) for x, y in zip(xs, upper)]
        lower_points = [axes.c2p(float(x), float(y)) for x, y in zip(xs[::-1], lower[::-1])]
        band = Polygon(*upper_points, *lower_points, stroke_width=0)
        band.set_fill(self.BLUE, opacity=0.15)
        band.set_z_index(3)
        curve = self._curve(axes, xs, values, color=self.YELLOW, width=3.4, opacity=1.0)
        curve.set_z_index(8)
        return curve, band

    @staticmethod
    def _posterior_weights(
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        x_design: np.ndarray,
        y_values: np.ndarray,
        noise: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        prior_precision = np.linalg.inv(prior_cov)
        posterior_cov = np.linalg.inv(prior_precision + x_design.T @ x_design / noise**2)
        posterior_mean = posterior_cov @ (prior_precision @ prior_mean + x_design.T @ y_values / noise**2)
        return posterior_mean, posterior_cov

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

        fig = plt.figure(figsize=(6.8, 1.1), dpi=320)
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
