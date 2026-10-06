"""main.py

Главный файл запуска приложения для Лабораторной работы №2 (Вариант 6).
"""

from data_variant import ALT_NAMES, CRIT_NAMES, EXPERT_RANKS, MAXIMIZE, SCORES
from gui import ElectreApp


def main():
    app = ElectreApp(
        alt_names=ALT_NAMES,
        crit_names=CRIT_NAMES,
        scores=SCORES,
        is_max=MAXIMIZE,
        expert_ranks=EXPERT_RANKS,
    )
    app.mainloop()


if __name__ == "__main__":
    main()