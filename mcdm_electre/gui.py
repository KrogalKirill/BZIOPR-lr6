"""gui.py

Модуль графического интерфейса пользователя.
Реализован на базе CustomTkinter (с fallback на tkinter в случае отсутствия библиотеки).
Позволяет интерактивно менять пороги согласия и несогласия, запускать расчет
и детально просматривать результаты на всех этапах.
"""

import tkinter as tk
from tkinter import ttk

try:
    import customtkinter as ctk

    GUI_BASE = ctk.CTk
    FRAME_BASE = ctk.CTkFrame
    BUTTON_BASE = ctk.CTkButton
    LABEL_BASE = ctk.CTkLabel
    SLIDER_BASE = ctk.CTkSlider
    TEXTBOX_BASE = ctk.CTkTextbox
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
except ImportError:
    # Запасной вариант на чистом Tkinter, если CustomTkinter не установлен
    GUI_BASE = tk.Tk
    FRAME_BASE = ttk.Frame
    BUTTON_BASE = ttk.Button
    LABEL_BASE = ttk.Label
    SLIDER_BASE = ttk.Scale
    TEXTBOX_BASE = tk.Text

import numpy as np
from electre_core import calc_weights_by_rank, find_pareto_set, run_electre


class ElectreApp(GUI_BASE):

    def __init__(self, alt_names, crit_names, scores, is_max, expert_ranks):
        super().__init__()

        self.title("МПР: Метод ЭЛЕКТРА I — Вариант 6 (Поставщики руды)")
        self.geometry("980x740")
        self.minsize(800, 600)

        self.alt_names = alt_names
        self.crit_names = crit_names
        self.scores = scores
        self.is_max = is_max
        self.expert_ranks = expert_ranks

        self._init_ui()
        self.solve()

    def _init_ui(self):
        # Панель управления (слайдеры порогов c* и d*)
        control_frame = FRAME_BASE(self)
        control_frame.pack(fill="x", padx=15, pady=10)

        # Слайдер c*
        self.lbl_c = LABEL_BASE(control_frame, text="Порог согласия c*: 0.50")
        self.lbl_c.pack(anchor="w", padx=10, pady=(5, 0))

        self.slider_c = SLIDER_BASE(
            control_frame, from_=0.0, to=1.0, command=self._on_slider_change
        )
        self.slider_c.set(0.50)
        self.slider_c.pack(fill="x", padx=10, pady=(0, 10))

        # Слайдер d*
        self.lbl_d = LABEL_BASE(control_frame, text="Порог несогласия d*: 0.50")
        self.lbl_d.pack(anchor="w", padx=10, pady=(5, 0))

        self.slider_d = SLIDER_BASE(
            control_frame, from_=0.0, to=1.0, command=self._on_slider_change
        )
        self.slider_d.set(0.50)
        self.slider_d.pack(fill="x", padx=10, pady=(0, 10))

        # Кнопки быстрых пресетов
        btn_frame = FRAME_BASE(control_frame)
        btn_frame.pack(fill="x", padx=10, pady=5)

        presets = [
            ("По умолчанию (0.5; 0.5)", 0.50, 0.50),
            ("Строгий (0.7; 0.3)", 0.70, 0.30),
            ("Селективный (0.75; 0.25)", 0.75, 0.25),
        ]
        for title, cv, dv in presets:
            btn = BUTTON_BASE(
                btn_frame,
                text=title,
                command=lambda c=cv, d=dv: self.set_preset(c, d),
            )
            btn.pack(side="left", padx=5)

        # Текстовое поле вывода протокола решения
        self.txt_output = TEXTBOX_BASE(self, font=("Consolas", 11))
        self.txt_output.pack(fill="both", expand=True, padx=15, pady=(5, 15))

    def set_preset(self, c_val, d_val):
        self.slider_c.set(c_val)
        self.slider_d.set(d_val)
        self._on_slider_change()

    def _on_slider_change(self, *args):
        c_val = float(self.slider_c.get())
        d_val = float(self.slider_d.get())
        self.lbl_c.configure(text=f"Порог согласия c*: {c_val:.2f}")
        self.lbl_d.configure(text=f"Порог несогласия d*: {d_val:.2f}")
        self.solve()

    def solve(self):
        c_star = round(float(self.slider_c.get()), 2)
        d_star = round(float(self.slider_d.get()), 2)

        # 1. Парето-отбор
        pareto_mask = find_pareto_set(self.scores, self.is_max)
        pareto_indices = np.where(pareto_mask)[0]
        pareto_names = [self.alt_names[i] for i in pareto_indices]
        pareto_scores = self.scores[pareto_mask]

        dominated = [
            self.alt_names[i]
            for i in range(len(self.alt_names))
            if not pareto_mask[i]
        ]

        # 2. Веса методом ранга
        weights = calc_weights_by_rank(self.expert_ranks)

        # 3. Метод ЭЛЕКТРА I
        c_mat, d_mat, kernel = run_electre(
            pareto_scores,
            pareto_names,
            weights,
            self.is_max,
            c_star,
            d_star,
        )

        # Формирование отчета
        lines = []
        lines.append("=" * 75)
        lines.append(
            "ЛАБОРАТОРНАЯ РАБОТА №2: ВЫБОР РЕШЕНИЯ МЕТОДОМ ЭЛЕКТРА (ВАРИАНТ 6)"
        )
        lines.append("=" * 75)

        lines.append("\n[ЭТАП 1] ИСХОДНЫЕ ДАННЫЕ И МНОЖЕСТВО ПАРЕТО")
        lines.append("-" * 75)
        hdr = f"{'Поставщик':<12}" + "".join(
            f"{cn:>20}" for cn in self.crit_names
        )
        lines.append(hdr)
        for i, name in enumerate(self.alt_names):
            row = (
                f"{name:<12}{self.scores[i, 0]:>20.1f}{self.scores[i, 1]:>20.1f}{self.scores[i, 2]:>20.2f}"
            )
            lines.append(row)

        lines.append(
            f"\nДоминируемые альтернативы: {', '.join(dominated) if dominated else 'нет'}"
        )
        lines.append(f"Множество Парето: {', '.join(pareto_names)}")

        lines.append("\n\n[ЭТАП 2] ВЕСА КРИТЕРИЕВ (МЕТОД РАНГА)")
        lines.append("-" * 75)
        for e_idx in range(self.expert_ranks.shape[0]):
            r_str = ", ".join(
                f"{self.crit_names[k]}={self.expert_ranks[e_idx, k]}"
                for k in range(len(self.crit_names))
            )
            lines.append(f"Эксперт {e_idx + 1}: {r_str}")
        lines.append("Нормализованные веса:")
        for k, cn in enumerate(self.crit_names):
            lines.append(f"  * {cn}: {weights[k]:.4f}")

        lines.append(f"\n\n[ЭТАП 3] МЕТОД ЭЛЕКТРА I (c* = {c_star:.2f}, d* = {d_star:.2f})")
        lines.append("-" * 75)

        lines.append("Матрица индексов согласия C:")
        c_hdr = f"{'':>8}" + "".join(f"{pn:>10}" for pn in pareto_names)
        lines.append(c_hdr)
        for i, pn in enumerate(pareto_names):
            row = f"{pn:>8}" + "".join(
                f"{'---':>10}" if i == j else f"{c_mat[i, j]:>10.3f}"
                for j in range(len(pareto_names))
            )
            lines.append(row)

        lines.append("\nМатрица индексов несогласия D:")
        d_hdr = f"{'':>8}" + "".join(f"{pn:>10}" for pn in pareto_names)
        lines.append(d_hdr)
        for i, pn in enumerate(pareto_names):
            row = f"{pn:>8}" + "".join(
                f"{'---':>10}" if i == j else f"{d_mat[i, j]:>10.3f}"
                for j in range(len(pareto_names))
            )
            lines.append(row)

        lines.append("\n" + "=" * 75)
        lines.append(
            f"ИТОГОВОЕ ЯДРО НЕДОМИНИРУЕМЫХ АЛЬТЕРНАТИВ: {', '.join(kernel)}"
        )
        lines.append("=" * 75)

        # Вывод в текстовый виджет
        if hasattr(self.txt_output, "delete"):
            self.txt_output.delete("1.0", tk.END)
            self.txt_output.insert(tk.END, "\n".join(lines))