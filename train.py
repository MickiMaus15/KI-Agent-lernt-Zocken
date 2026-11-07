import csv
import json
import os
import random
import time

import numpy as np

from snake_env import SnakeEnv

#Einstellungen
NUM_EPISODES = 10000                 
RESULT_DIR = "ergebnisse"
RUN_PREFIX = "lauf"
SHOW_EVERY = 0                    #0=nix zeigen, 100=jede 100. Episode zeigen  
SHOW_FPS = 15                       
SEED = 0                           

#Spielfeld & Rewards
from snake_env import (
    SnakeEnv,
    GRID_SIZE,
    REWARD_FOOD,
    REWARD_DEATH,
    STEP_PENALTY,
    MAX_STEPS_WITHOUT_FOOD,
)

#Q-Learning Parameter
ALPHA = 0.3             # Lernrate: wie stark neue Erfahrung die Tabelle verändert
GAMMA = 0.9             # Discount-Faktor: wie wichtig zukünftige Belohnung ist
EPSILON_START = 1     # Anfangs nur zufällige Aktionen (Erkunden)
EPSILON_END = 0.01      # am Ende fast nur noch beste bekannte Aktion (Ausnutzen)
EPSILON_DECAY = 0.99   # Epsilon wird nach jeder Episode mit diesem Faktor multipliziert


#hilfsfunktion (wandelt 11 vektoren zu zahl um)
def state_to_index(obs):
    index = 0
    for bit in obs:
        index = index * 2 + int(bit)
    return index



#file system anlegen
def create_run_dir():
    os.makedirs(RESULT_DIR, exist_ok= True)
    numbers = []
    for name in os.listdir(RESULT_DIR):
        prefix, _, number = name.rpartition("_")
        if prefix == RUN_PREFIX and number.isdigit():
            numbers.append(int(number))
    next_number = max(numbers, default = 0) + 1 
    while True:
        run_dir = os.path.join(RESULT_DIR, f"{RUN_PREFIX}_{next_number:02d}")
        try:
            os.makedirs(run_dir)
            return(run_dir)
        except FileExistsError:
            next_number += 1
    



#trainieren
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    run_dir = create_run_dir()
    print(f"als {os.path.abspath(run_dir)} gespeichert")
    env_kwargs = dict(
        grid_size=GRID_SIZE,
        reward_food=REWARD_FOOD,
        reward_death=REWARD_DEATH,
        step_penalty=STEP_PENALTY,
        max_steps_without_food=MAX_STEPS_WITHOUT_FOOD,
    )
    env = SnakeEnv(**env_kwargs)                    # schnelles Training ohne Fenster
    show_env = None                                 # Fenster-Environment, nur bei Bedarf
    if SHOW_EVERY > 0:
        show_env = SnakeEnv(render_mode="human", fps=SHOW_FPS, **env_kwargs)

    # Q-Tabelle: eine Zeile pro Zustand (2^11 = 2048), eine Spalte pro Aktion (3)
    q_table = np.zeros((2 ** 11, env.action_space.n))
    epsilon = EPSILON_START
    log_rows = []                                   # eine Zeile pro Episode für die CSV
    training_step = 0
    start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        # In dieser Episode Fenster zeigen? Nur jede SHOW_EVERY-te.
        active_env = show_env if (show_env and episode % SHOW_EVERY == 0) else env
        obs, info = active_env.reset(seed=SEED if episode == 1 else None)
        state = state_to_index(obs)
        total_reward, steps,  done = 0.0, 0, False

        while not done:
            # Epsilon-greedy: mit Wahrscheinlichkeit epsilon zufällig, sonst beste Aktion
            if random.random() < epsilon:
                action = active_env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            obs, reward, terminated, truncated, info = active_env.step(action)
            next_state = state_to_index(obs)
            done = terminated or truncated

            # Q-Learning-Update:  Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a))
            # Bei Tod (terminated) gibt es keinen Folgezustand, daher kein gamma-Teil.
            # Bei Timeout (truncated) schon: das Spiel ging ja weiter.
            future = 0.0 if terminated else GAMMA * np.max(q_table[next_state])
            q_table[state, action] += ALPHA * (reward + future - q_table[state, action])

            state = next_state
            total_reward += reward
            training_step += 1
            steps += 1

        # Nach der Episode: Epsilon senken und Ergebnis ins Log schreiben
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        log_rows.append({
            "episode": episode,
            "score": info["score"],                       # gefressene Äpfel
            "seed" : SEED,
            "length": info["length"],
            "trainingsschritt": training_step,
            "total_reward": round(total_reward, 3),
            "todesursache": info["death_cause"],
            "steps": steps,
            "epsilon": round(epsilon, 4),
        })

        if episode % 100 == 0 or episode == NUM_EPISODES:
            recent = [r["score"] for r in log_rows[-100:]]
            print(f"Episode {episode}/{NUM_EPISODES} | "
                  f"Ø Score (letzte 100): {np.mean(recent):.2f} | epsilon {epsilon:.3f}")

    env.close()
    if show_env:
        show_env.close()

    save_results(run_dir, q_table, log_rows, time.time() - start_time)


#Speichern
def save_results(run_dir, q_table, log_rows, seconds):
    #CSV-Log
    csv_path = os.path.join(run_dir, "training_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    # Q-TAbelle
    np.save(os.path.join(run_dir, "q_table.npy"), q_table)

    # Einstellungen
    settings = {
        "NUM_EPISODES": NUM_EPISODES, "SEED": SEED, "GRID_SIZE": GRID_SIZE,
        "REWARD_FOOD": REWARD_FOOD, "REWARD_DEATH": REWARD_DEATH,
        "STEP_PENALTY": STEP_PENALTY, "MAX_STEPS_WITHOUT_FOOD": MAX_STEPS_WITHOUT_FOOD,
        "ALPHA": ALPHA, "GAMMA": GAMMA, "EPSILON_START": EPSILON_START,
        "EPSILON_END": EPSILON_END, "EPSILON_DECAY": EPSILON_DECAY,
    }
    with open(os.path.join(run_dir, "einstellungen.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

    # --- Zusammenfassung
    scores = np.array([r["score"] for r in log_rows])
    last = scores[-100:]
    summary = {
        "episoden": len(log_rows),
        "dauer_sekunden": round(seconds, 1),
        "bester_score": int(scores.max()),
        "mittlerer_score_erste_100": round(float(scores[:100].mean()), 2),
        "mittlerer_score_letzte_100": round(float(last.mean()), 2),
        "laengeste_schlange": max(r["length"] for r in log_rows),
        "trainingsschritte_gesamt" : log_rows[-1]["trainingsschritt"],
        "todesursache_letzte_100":{
            cause: sum(r["todesursache"] == cause for r in log_rows[-100:])
            for cause in sorted({r["todesursache"] for r in log_rows[-100::]})
    }
    }
    with open(os.path.join(run_dir, "zusammenfassung.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    #Lernkurve
    try:
        import matplotlib
        matplotlib.use("Agg")                  # kein Fenster nötig, nur Datei
        import matplotlib.pyplot as plt

        window = min(100, len(scores))
        moving = np.convolve(scores, np.ones(window) / window, mode="valid")
        plt.figure(figsize=(8, 4.5))
        plt.plot(scores, alpha=0.25, label="Score pro Episode")
        plt.plot(range(window - 1, len(scores)), moving, label=f"Gleitender Mittelwert ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Score (Äpfel)")
        plt.title("Lernkurve Q-Learning")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "lernkurve.png"), dpi=150)
        plt.close()
    except ImportError:
        print("Hinweis: matplotlib fehlt, Diagramm wurde übersprungen (pip install matplotlib).")

    print(f"\nFertig. Alle Daten gespeichert in: {os.path.abspath(run_dir)}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    train()
