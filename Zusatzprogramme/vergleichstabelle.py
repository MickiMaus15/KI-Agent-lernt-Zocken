"""
Vergleichstabelle der Trainingsvarianten (Kapitel 4.2).

Erwartete Ordnerstruktur (Seed und Variante stehen in den Ordnernamen):

    wurzel/
        Seed=0/
            base/training_log.csv
            alpha/training_log.csv
            ...
        Seed=50/
            ...
        Seed=209/
            ...

Pro Variante berechnet das Skript:
  - Ø Score der letzten N Episoden je Lauf (Seed), danach Mittel über die Seeds
  - Streuung zwischen den Seeds (Minimum bis Maximum)
  - Abweichung zur Basisvariante

Aufruf:
    python vergleichstabelle.py "C:/pfad/zu/wurzel"
    python vergleichstabelle.py wurzel --letzte 500 --base base --datei training_log.csv

Liegen die CSV-Dateien tiefer (z. B. Seed=0/alpha/lauf_01/training_log.csv), wird trotzdem
der Ordner direkt unter "Seed=..." als Variante verwendet.
Ohne "Seed="-Ordner wird ersatzweise der Ordner der CSV als Variante und die Spalte "seed" genutzt.
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def lade_laeufe(wurzel: Path, dateiname: str) -> pd.DataFrame:
    teile = []
    for pfad in sorted(wurzel.rglob(dateiname)):
        df = pd.read_csv(pfad, encoding="utf-8")
        teile_pfad = pfad.relative_to(wurzel).parts[:-1]

        seed_idx = next(
            (i for i, t in enumerate(teile_pfad) if re.fullmatch(r"seed\s*=\s*\d+", t, re.I)), None
        )
        if seed_idx is not None and seed_idx + 1 < len(teile_pfad):
            seed = int(re.search(r"\d+", teile_pfad[seed_idx]).group())
            variante = teile_pfad[seed_idx + 1]
        else:
            seed = df["seed"].iloc[0] if "seed" in df.columns else 0
            variante = pfad.parent.name

        df = df.assign(variante=variante, seed_lauf=seed)
        teile.append(df[["variante", "seed_lauf", "episode", "score"]])

    if not teile:
        raise SystemExit(f"Keine Datei '{dateiname}' unter {wurzel} gefunden.")
    return pd.concat(teile, ignore_index=True)


def berechne(df: pd.DataFrame, letzte: int, base: str) -> pd.DataFrame:
    # Pro Lauf: Ø Score der letzten N Episoden
    je_lauf = []
    for (variante, seed), g in df.groupby(["variante", "seed_lauf"]):
        g = g.sort_values("episode").tail(letzte)
        je_lauf.append({"variante": variante, "seed": seed, "oe_score": g["score"].mean()})
    je_lauf = pd.DataFrame(je_lauf)

    # Pro Variante: Mittel und Streuung über die Seeds
    t = je_lauf.groupby("variante")["oe_score"].agg(
        laeufe="count", mittel="mean", minimum="min", maximum="max", std="std"
    )

    # Basisvariante unabhängig von Groß-/Kleinschreibung finden
    base_name = next((n for n in t.index if n.lower() == base.lower()), None)
    if base_name is not None:
        t["diff_base"] = t["mittel"] - t.loc[base_name, "mittel"]
    else:
        print(f"Hinweis: Variante '{base}' nicht gefunden, Abweichung zur Basis entfällt.")
        t["diff_base"] = float("nan")

    # Basis zuerst, Rest nach Ø Score absteigend
    rest = t.drop(index=base_name, errors="ignore").sort_values("mittel", ascending=False).index
    reihenfolge = ([base_name] if base_name is not None else []) + list(rest)
    return t.loc[reihenfolge].round(2)


def als_markdown(t: pd.DataFrame, letzte: int) -> str:
    zeilen = [
        f"| Variante | Läufe | Ø Score (letzte {letzte} Ep.) | Min bis Max | Abw. zu Base |",
        "|---|---|---|---|---|",
    ]
    for name, r in t.iterrows():
        diff = "" if pd.isna(r["diff_base"]) else f"{r['diff_base']:+.1f}"
        zeilen.append(
            f"| {name} | {int(r['laeufe'])} | {r['mittel']:.1f} | {r['minimum']:.1f} bis {r['maximum']:.1f} | {diff} |"
        )
    return "\n".join(zeilen)


def main() -> None:
    p = argparse.ArgumentParser(description="Vergleichstabelle der Trainingsvarianten")
    p.add_argument("wurzel", help="Ordner, der die Ordner Seed=0, Seed=50, ... enthält")
    p.add_argument("--letzte", type=int, default=500, help="Anzahl letzter Episoden pro Lauf (Standard 500)")
    p.add_argument("--base", default="base", help="Ordnername der Basisvariante (Standard 'base')")
    p.add_argument("--datei", default="training_log.csv", help="Name der CSV-Dateien (Standard training_log.csv)")
    p.add_argument("--out", default="vergleichstabelle.csv", help="Ausgabedatei (CSV)")
    a = p.parse_args()

    df = lade_laeufe(Path(a.wurzel), a.datei)
    t = berechne(df, a.letzte, a.base)
    t.to_csv(a.out, encoding="utf-8-sig")
    print(als_markdown(t, a.letzte))
    print(f"\nGespeichert: {a.out}")


if __name__ == "__main__":
    main()
