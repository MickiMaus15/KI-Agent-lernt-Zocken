"""
Todesursachen aus dem Snake-Trainingslog als Diagramm darstellen.

Aufruf:
    python todesursachen_diagramm.py training_log.csv
    python todesursachen_diagramm.py lauf_01/training_log.csv --out lauf_01.png --block 250

Erwartete Spalten: episode, todesursache (weitere Spalten werden ignoriert).
Enthält die Datei mehrere Seeds, werden alle Läufe zusammen ausgewertet.
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

# Feste Farben pro Ursache, damit mehrere Diagramme vergleichbar bleiben
FARBEN = {
    "Wand": "#4C78A8",
    "Körper": "#F58518",
    "Timeout": "#54A24B",
}


def lade_daten(pfad: str) -> pd.DataFrame:
    df = pd.read_csv(pfad, encoding="utf-8")
    fehlend = {"episode", "todesursache"} - set(df.columns)
    if fehlend:
        raise SystemExit(f"Spalten fehlen in der CSV: {', '.join(sorted(fehlend))}")
    return df


def zeichne(df: pd.DataFrame, block: int, ausgabe: str) -> None:
    reihenfolge = [u for u in FARBEN if u in set(df["todesursache"])]
    reihenfolge += [u for u in df["todesursache"].unique() if u not in reihenfolge]
    farben = [FARBEN.get(u, "#999999") for u in reihenfolge]

    gesamt = df["todesursache"].value_counts().reindex(reihenfolge)
    anteil = gesamt / gesamt.sum() * 100

    # Anteile pro Block von Episoden (zeigt die Entwicklung über das Training)
    df = df.copy()
    df["block"] = (df["episode"] - 1) // block
    tabelle = (
        df.groupby(["block", "todesursache"]).size().unstack(fill_value=0)
        .reindex(columns=reihenfolge, fill_value=0)
    )
    verlauf = tabelle.div(tabelle.sum(axis=1), axis=0) * 100
    x = (verlauf.index * block) + block / 2

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.6]}
    )

    # Links: Gesamtverteilung als Balkendiagramm
    balken = ax1.bar(reihenfolge, gesamt.values, color=farben)
    for b, n, p in zip(balken, gesamt.values, anteil.values):
        ax1.text(
            b.get_x() + b.get_width() / 2, b.get_height(),
            f"{n}\n({p:.1f} %)", ha="center", va="bottom", fontsize=10,
        )
    ax1.set_ylim(0, gesamt.max() * 1.18)
    ax1.set_ylabel("Anzahl Episoden")
    ax1.set_title("Todesursachen insgesamt")

    # Rechts: Verlauf der Anteile über das Training (gestapelt, 100 %)
    ax2.stackplot(
        x, [verlauf[u].values for u in reihenfolge],
        labels=reihenfolge, colors=farben, alpha=0.9,
    )
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(0, 100)
    ax2.set_xlabel(f"Episode (Blöcke à {block})")
    ax2.set_ylabel("Anteil der Todesursachen (%)")
    ax2.set_title("Verlauf während des Trainings")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(reihenfolge), frameon=False)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Todesursachen des Agenten ({len(df)} Episoden)", fontsize=13)
    fig.tight_layout()
    fig.savefig(ausgabe, dpi=200, bbox_inches="tight")
    print(f"Gespeichert: {ausgabe}")
    print(gesamt.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Todesursachen als Diagramm")
    parser.add_argument("csv", nargs="?", default="training_log.csv", help="Pfad zur CSV-Datei")
    parser.add_argument("--out", default="todesursachen.png", help="Ausgabedatei (PNG)")
    parser.add_argument("--block", type=int, default=500, help="Episoden pro Block im Verlaufsdiagramm")
    args = parser.parse_args()

    zeichne(lade_daten(args.csv), args.block, args.out)


if __name__ == "__main__":
    main()
