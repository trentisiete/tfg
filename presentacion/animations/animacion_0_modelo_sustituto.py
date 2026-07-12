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


class Animacion0ModeloSustituto(Scene):
    """Visual explanation of an expensive function and its surrogate model."""

    X_MIN = -3.2
    X_MAX = 3.2
    Y_MIN = -1.9
    Y_MAX = 1.9
    N_POINTS = 260

    BG = "#E8E8E8"
    TEXT = "#1F2933"
    BLUE = "#174C5B"
    AMBER = "#C9822B"
    GREEN = "#2F7D6D"
    GREY = "#5F6368"
    LIGHT_GREY = "#C9CDD2"
    WHITE = "#F8FAFC"

    def construct(self) -> None:
        self.camera.background_color = self.BG

        xs = np.linspace(self.X_MIN, self.X_MAX, self.N_POINTS)
        x_obs = np.array([-2.75, -1.65, -0.75, 0.35, 1.35, 2.45])
        y_obs = self._real_function(x_obs)
        y_hat = self._surrogate_curve(x_obs, y_obs, xs)
        y_hat_obs = self._surrogate_curve(x_obs, y_obs, x_obs)

        axes = self._axes()
        axis_labels = self._axis_labels(axes)
        real_box = self._real_box()
        clock, clock_hand = self._clock(real_box.get_right() + RIGHT * 0.72)
        cost_label = self._text("evaluación costosa", 27, self.AMBER)
        cost_label.next_to(clock, RIGHT, buff=0.22)
        cost_group = VGroup(clock, cost_label).set_z_index(20)

        observed_dots = VGroup()
        observed_stems = VGroup()
        for x, y in zip(x_obs, y_obs):
            observed_dots.add(self._dot(axes.c2p(float(x), float(y)), self.AMBER, 0.075))
            observed_stems.add(self._stem(axes, float(x), float(y), self.AMBER, opacity=0.24))

        data_label = self._latex(
            r"$\mathcal{D}_n=\{(x_i,y_i)\}_{i=1}^{n}$",
            height=0.29,
            color=self.TEXT,
        )
        data_label.move_to(axes.c2p(1.55, 1.45)).set_z_index(15)

        surrogate_curve = self._curve(axes, xs, y_hat, self.GREEN, width=4.2)
        fit_gaps = self._fit_gaps(axes, x_obs, y_obs, y_hat_obs)
        surrogate_label = self._latex(r"$\hat f(x)$", height=0.36, color=self.GREEN)
        label_x = 1.55
        label_y = float(np.interp(label_x, xs, y_hat)) + 0.42
        surrogate_label.move_to(axes.c2p(label_x, label_y)).set_z_index(18)

        candidate_x = np.array([-2.85, -2.35, -1.95, -1.55, -1.05, -0.65, -0.25, 0.25, 0.65, 1.05, 1.55, 1.95, 2.45, 2.85])
        candidate_y = self._surrogate_curve(x_obs, y_obs, candidate_x)
        candidate_inputs = VGroup(
            *(self._dot(axes.c2p(float(x), 0), self.GREEN, 0.047, stroke=False) for x in candidate_x)
        )
        query_lines = VGroup(
            *(self._stem(axes, float(x), float(y), self.GREEN, opacity=0.36) for x, y in zip(candidate_x, candidate_y))
        )
        query_dots = VGroup(
            *(self._dot(axes.c2p(float(x), float(y)), self.GREEN, 0.048, stroke=False) for x, y in zip(candidate_x, candidate_y))
        )

        fast_label = self._text("consulta rápida", 28, self.GREEN)
        fast_label.next_to(axes.c2p(-2.85, self.Y_MAX - 0.25), RIGHT, buff=0.1).set_z_index(20)
        speed_marks = self._speed_marks(fast_label)

        best_idx = int(np.argmin(candidate_y))
        highlight = self._highlight(query_dots[best_idx].get_center())
        decision_box = self._decision_box()
        decision_arrow = Arrow(
            start=query_dots[best_idx].get_center() + UP * 0.12 + RIGHT * 0.12,
            end=decision_box.get_left() + LEFT * 0.08,
            color=self.GREEN,
            stroke_width=4,
            buff=0.05,
            max_tip_length_to_length_ratio=0.08,
        ).set_z_index(16)
        final_sentence = self._text(
            "No sustituye el experimento: ayuda a decidir cuál hacer después.",
            30,
            self.TEXT,
        )
        final_sentence.scale_to_fit_width(12.4)
        final_sentence.to_edge(DOWN, buff=0.36).set_z_index(25)

        self.play(Create(axes), FadeIn(axis_labels), FadeIn(real_box, shift=DOWN * 0.12), run_time=1.8)
        self.play(FadeIn(cost_group, shift=LEFT * 0.08), run_time=0.7)
        self.wait(0.5)

        first_dot, first_stem = observed_dots[0], observed_stems[0]
        self._expensive_evaluation(axes, real_box, clock_hand, float(x_obs[0]), first_dot, first_stem, slow=True)
        self.wait(0.25)

        for i in range(1, len(x_obs)):
            self._expensive_evaluation(
                axes,
                real_box,
                clock_hand,
                float(x_obs[i]),
                observed_dots[i],
                observed_stems[i],
                slow=False,
            )

        self.play(FadeIn(data_label, shift=DOWN * 0.08), run_time=0.75)
        self.wait(0.8)

        self.play(
            cost_group.animate.set_opacity(0.28),
            real_box.animate.set_opacity(0.55),
            observed_stems.animate.set_opacity(0.14),
            observed_dots.animate.set_color(self.BLUE),
            run_time=0.9,
        )
        self.play(Create(surrogate_curve), run_time=3.8, rate_func=smooth)
        self.play(FadeIn(fit_gaps), run_time=0.45)
        self.play(FadeIn(surrogate_label, shift=UP * 0.08), run_time=0.65)
        self.wait(0.75)

        self.play(
            FadeOut(data_label, shift=UP * 0.05),
            FadeIn(fast_label, shift=RIGHT * 0.1),
            LaggedStart(*(FadeIn(mark, shift=RIGHT * 0.08) for mark in speed_marks), lag_ratio=0.12),
            run_time=0.85,
        )
        self.play(LaggedStart(*(GrowFromCenter(dot) for dot in candidate_inputs), lag_ratio=0.035), run_time=0.95)
        self.play(
            LaggedStart(
                *(
                    AnimationGroup(Create(line), GrowFromCenter(dot), lag_ratio=0.0)
                    for line, dot in zip(query_lines, query_dots)
                ),
                lag_ratio=0.035,
            ),
            run_time=2.4,
        )
        self.play(candidate_inputs.animate.set_opacity(0.35), run_time=0.45)
        self.wait(0.35)

        self.play(GrowFromCenter(highlight), query_dots[best_idx].animate.scale(1.55), run_time=0.7)
        self.play(Create(decision_arrow), FadeIn(decision_box, shift=LEFT * 0.08), run_time=1.05)
        self.play(FadeIn(final_sentence, shift=UP * 0.08), run_time=1.0)
        self.wait(4.2)

    def _expensive_evaluation(
        self,
        axes: Axes,
        real_box: VGroup,
        clock_hand: Line,
        x_value: float,
        observed_dot: Dot,
        observed_stem: DashedLine,
        slow: bool,
    ) -> None:
        token = self._dot(axes.c2p(x_value, 0), self.AMBER, 0.07)
        token.set_z_index(30)
        target = real_box[0].get_left() + RIGHT * 0.34
        self.play(GrowFromCenter(token), run_time=0.32 if slow else 0.30)
        self.play(token.animate.move_to(target), run_time=0.95 if slow else 0.65, rate_func=smooth)
        self.play(
            Rotate(clock_hand, angle=TAU, about_point=clock_hand.get_start(), rate_func=linear),
            real_box[0].animate.set_stroke(self.AMBER, width=3.0, opacity=0.72),
            real_box[1].animate.set_opacity(0.92),
            run_time=1.35 if slow else 0.85,
        )
        self.play(
            FadeOut(token, scale=0.75),
            real_box[0].animate.set_stroke(self.TEXT, width=1.4, opacity=0.18),
            real_box[1].animate.set_opacity(0.48),
            Create(observed_stem),
            GrowFromCenter(observed_dot),
            run_time=0.70 if slow else 0.50,
        )

    def _axes(self) -> Axes:
        axes = Axes(
            x_range=[self.X_MIN, self.X_MAX, 1],
            y_range=[self.Y_MIN, self.Y_MAX, 1],
            x_length=11.75,
            y_length=4.85,
            axis_config={
                "color": self.GREY,
                "stroke_width": 2.1,
                "include_tip": False,
                "include_ticks": True,
                "tick_size": 0.075,
            },
            tips=False,
        )
        axes.move_to(DOWN * 0.78)
        axes.set_z_index(1)
        return axes

    def _axis_labels(self, axes: Axes) -> VGroup:
        x_label = self._latex(r"$x$", height=0.27, color=self.GREY)
        y_label = self._latex(r"$y$", height=0.27, color=self.GREY)
        x_label.next_to(axes.x_axis, RIGHT, buff=0.1)
        y_label.next_to(axes.y_axis, UP, buff=0.1)
        return Group(x_label, y_label).set_z_index(8)

    def _real_box(self) -> VGroup:
        body = RoundedRectangle(
            width=1.72,
            height=0.92,
            corner_radius=0.12,
            color=self.TEXT,
            stroke_width=1.4,
            stroke_opacity=0.18,
            fill_color=self.TEXT,
            fill_opacity=1.0,
        )
        label = self._latex(r"$f(x)$", height=0.39, color=self.WHITE)
        activity = self._box_activity()
        label.move_to(body.get_center())
        activity.move_to(body.get_center())
        box = Group(body, activity, label)
        box.move_to(LEFT * 4.82 + UP * 3.08)
        box.set_z_index(20)
        return box

    def _box_activity(self) -> VGroup:
        wave_left = VMobject(color=self.AMBER, stroke_width=1.45)
        wave_left.set_points_smoothly(
            [
                LEFT * 0.66 + UP * 0.22,
                LEFT * 0.48 + UP * 0.33,
                LEFT * 0.30 + UP * 0.13,
                LEFT * 0.12 + UP * 0.24,
            ]
        )
        wave_right = VMobject(color=self.AMBER, stroke_width=1.45)
        wave_right.set_points_smoothly(
            [
                RIGHT * 0.13 + DOWN * 0.22,
                RIGHT * 0.32 + DOWN * 0.08,
                RIGHT * 0.50 + DOWN * 0.30,
                RIGHT * 0.67 + DOWN * 0.16,
            ]
        )
        node_a = Dot(LEFT * 0.60 + DOWN * 0.20, radius=0.025, color=self.AMBER)
        node_b = Dot(RIGHT * 0.60 + UP * 0.22, radius=0.025, color=self.AMBER)
        node_c = Circle(radius=0.052, color=self.AMBER, stroke_width=1.35)
        node_c.move_to(RIGHT * 0.55 + DOWN * 0.18)
        activity = VGroup(wave_left, wave_right, node_a, node_b, node_c)
        activity.set_opacity(0.48)
        activity.set_z_index(21)
        return activity

    def _clock(self, center: np.ndarray) -> tuple[VGroup, Line]:
        ring = Circle(radius=0.19, color=self.AMBER, stroke_width=3.2)
        hand = Line(ORIGIN, UP * 0.13, color=self.AMBER, stroke_width=3.0)
        pin = Dot(ORIGIN, radius=0.025, color=self.AMBER)
        clock = VGroup(ring, hand, pin)
        clock.move_to(center)
        clock.set_z_index(20)
        return clock, hand

    def _decision_box(self) -> VGroup:
        body = RoundedRectangle(
            width=2.95,
            height=0.78,
            corner_radius=0.12,
            color=self.GREEN,
            stroke_width=2.2,
            fill_color=self.BG,
            fill_opacity=0.92,
        )
        label = self._text("decidir siguiente evaluación", 23, self.GREEN)
        label.scale_to_fit_width(2.55)
        label.move_to(body.get_center())
        box = VGroup(body, label)
        box.move_to(RIGHT * 4.35 + UP * 1.95)
        box.set_z_index(22)
        return box

    def _curve(self, axes: Axes, xs: np.ndarray, ys: np.ndarray, color: str, width: float) -> VMobject:
        points = [axes.c2p(float(x), float(y)) for x, y in zip(xs, ys)]
        curve = VMobject(color=color, stroke_width=width)
        curve.set_points_smoothly(points)
        curve.set_fill(opacity=0)
        curve.set_z_index(10)
        return curve

    def _stem(self, axes: Axes, x: float, y: float, color: str, opacity: float) -> DashedLine:
        line = DashedLine(
            axes.c2p(x, 0),
            axes.c2p(x, y),
            dash_length=0.085,
            color=color,
            stroke_width=1.8,
        )
        line.set_opacity(opacity)
        line.set_z_index(4)
        return line

    def _fit_gaps(
        self,
        axes: Axes,
        x_values: np.ndarray,
        y_real: np.ndarray,
        y_fit: np.ndarray,
    ) -> VGroup:
        gaps = VGroup()
        for x, y_a, y_b in zip(x_values, y_real, y_fit):
            gap = DashedLine(
                axes.c2p(float(x), float(y_a)),
                axes.c2p(float(x), float(y_b)),
                dash_length=0.05,
                color=self.GREY,
                stroke_width=1.45,
            )
            gap.set_opacity(0.24)
            gap.set_z_index(7)
            gaps.add(gap)
        return gaps

    def _dot(self, point: np.ndarray, color: str, radius: float, stroke: bool = True) -> Dot:
        dot = Dot(point, radius=radius, color=color)
        if stroke:
            dot.set_stroke(self.WHITE, width=1.35, opacity=0.96)
        dot.set_z_index(14)
        return dot

    def _highlight(self, center: np.ndarray) -> VGroup:
        outer = Circle(radius=0.18, color=self.AMBER, stroke_width=3.0)
        inner = Circle(radius=0.29, color=self.AMBER, stroke_width=1.4)
        inner.set_opacity(0.45)
        highlight = VGroup(outer, inner)
        highlight.move_to(center)
        highlight.set_z_index(18)
        return highlight

    def _speed_marks(self, label: Text) -> VGroup:
        marks = VGroup()
        origin = label.get_left() + LEFT * 0.18 + DOWN * 0.02
        for i, length in enumerate([0.32, 0.24, 0.16]):
            mark = Line(
                origin + LEFT * (0.12 * i) + UP * (0.08 * (i - 1)),
                origin + LEFT * (0.12 * i + length) + UP * (0.08 * (i - 1)),
                color=self.GREEN,
                stroke_width=3.0,
            )
            mark.set_z_index(20)
            marks.add(mark)
        return marks

    def _text(self, text: str, size: int, color: str) -> Text:
        return Text(text, font="Segoe UI", font_size=size, color=color)

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

        fig = plt.figure(figsize=(6.3, 1.05), dpi=320)
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
    def _real_function(x: np.ndarray) -> np.ndarray:
        return (
            0.36 * np.sin(1.65 * x + 0.35)
            + 0.26 * np.sin(3.70 * x - 0.55)
            + 0.18 * np.cos(6.10 * x + 0.30) * np.exp(-0.10 * (x - 0.25) ** 2)
            - 0.50 * np.exp(-((x + 1.28) / 0.35) ** 2)
            + 0.43 * np.exp(-((x - 1.45) / 0.42) ** 2)
            - 0.07 * x
        )

    @staticmethod
    def _surrogate_curve(x_train: np.ndarray, y_train: np.ndarray, xs: np.ndarray) -> np.ndarray:
        xs = np.atleast_1d(xs)
        centers = np.array([-2.90, -1.65, -0.45, 0.75, 1.95, 3.00])
        width = 1.00

        def design_matrix(x_values: np.ndarray) -> np.ndarray:
            x_values = np.atleast_1d(x_values)
            radial = np.exp(-0.5 * ((x_values[:, None] - centers[None, :]) / width) ** 2)
            trend = np.column_stack([np.ones_like(x_values), 0.32 * x_values])
            return np.column_stack([trend, radial])

        phi_train = design_matrix(x_train)
        regularization = np.diag([0.0, 0.04, *([0.20] * len(centers))])
        weights = np.linalg.solve(
            phi_train.T @ phi_train + regularization,
            phi_train.T @ y_train,
        )
        return design_matrix(xs) @ weights
