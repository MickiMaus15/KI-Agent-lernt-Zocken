import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

GRID_SIZE = 10
REWARD_FOOD= 5.0
REWARD_DEATH = -10.0
STEP_PENALTY = -0.5
MAX_STEPS_WITHOUT_FOOD = 50




class SnakeEnv(gym.Env):
#initialisierung
    def __init__(
        self,
        grid_size=GRID_SIZE,                #spielfeld größe
        reward_food=REWARD_FOOD,            # belohnung für apfel
        reward_death=REWARD_DEATH,          # bestafung bei tod
        step_penalty=STEP_PENALTY,          # bestafung für jeden schritt
        max_steps_without_food=MAX_STEPS_WITHOUT_FOOD,  # max schritte bis tod
        render_mode = None,
        cell_size = 30,
        fps = 10,
    ):
        super().__init__()
        self.n = grid_size
        self.obs_type = "features"

        # belohnung und strafen
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.step_penalty = step_penalty
        self.max_steps_without_food = max_steps_without_food
        self.steps_since_food = 0    # schritte zähler seit letzem apfel

        #renderer
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.fps = fps
        self._window= None
        self._clock = None
        
        # Zustand umwelt 
        self.snake = []              # Liste von (x, y); Index 0 ist der Kopf
        self.direction = (1, 0)      # aktuelle laufrichtung
        self.food = (0, 0)           # Position des apfels
        self.score = 0               # Anzahl gegessener äpfel

        #aktionsräume 
        self.action_space = spaces.Discrete(3)

    #hilfsfunktionen   
    @staticmethod
    #links drehen relativ zur schlange
    def _turn_left(d):
        dx, dy = d
        return (dy, -dx)

    @staticmethod
    #rechts drehen relativ zur schlange
    def _turn_right(d):
        dx, dy = d
        return (-dy, dx)

    #gefahren(wand/schlange selbst)
    def _is_danger(self, pos):
        x, y = pos
        # Außerhalb des spielfelds => wand
        if x < 0 or x >= self.n or y < 0 or y >= self.n:
            return True
    #eigener körper; außer letztes glied, da sich mit jedem schritt weg bewegt
        return pos in self.snake[:-1]

    #apfel setzen
    def _place_food(self):
        free = [
            (x, y)
            for x in range(self.n)
            for y in range(self.n)
            if (x, y) not in self.snake
        ]
        self.food = random.choice(free)

    #was agent sieht   
    def _get_obs(self):
        return self._get_features()

    #11 mermalsvektoren (0=nein, 1=ja)
    # 0-2 : gefahr links,vorne,rechts
    #3-6: laufrichtung : links,rechts,oben,unten
    #7-10: apfel relativ zum kopf: links,rechts,oben,unten
    def _get_features(self):
        hx, hy = self.snake[0]   # Kopfposition
        d = self.direction

        def ahead(direction):
            #feld, welches snake kommt, wenn in direction geht
            return (hx + direction[0], hy + direction[1])

        fx, fy = self.food
        features = [
            #gefahr
            self._is_danger(ahead(d)),
            self._is_danger(ahead(self._turn_left(d))),
            self._is_danger(ahead(self._turn_right(d))),
            #laufrichtung
            d == (0, -1),
            d == (0, 1),
            d == (-1, 0),
            d == (1, 0),
            #apfelposition vom kopf aus
            fx < hx,
            fx > hx,
            fy < hy,
            fy > hy,
        ]
        return np.array(features, dtype=np.int8)

    #reset: neue episode starten
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        mid = self.n // 2
        # Schlange mit 3 Gliedern in der Feldmitte, Kopf zuerst, Richtung rechts
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self.direction = (1, 0)
        self.score = 0
        self.steps_since_food = 0
        self._place_food()

        if self.render_mode == "human":
            self.render()
        # return von Beobachtung und score
        return self._get_obs(), {"score": self.score, "length": len(self.snake), "death_cause": None}

    #spielzug/step
    def step(self, action):
        """Führt eine Aktion aus und gibt zurück:
        (Beobachtung, Belohnung, terminated, truncated, info)
        terminated = Spiel zu Ende (Tod oder gewonnen)
        truncated  = Abbruch von außen (Timeout)"""

        #relative aktion in absolute richtung um rechnen
        if action == 1:
            self.direction = self._turn_left(self.direction)
        elif action == 2:
            self.direction = self._turn_right(self.direction)
        # action == 0 -> Richtung bleibt gleich

        # neue kopfposition = alte kopfposition + richtung
        head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1],
        )
        self.steps_since_food += 1

        terminated = False      #spiel zu ende durch tod oder win
        truncated = False   #spiel durch timeout zu ende
        death_cause = None
        reward = self.step_penalty   # schrittstafe

        if self._is_danger(head):
            # wand oder körper getroffen
            terminated = True
            reward = self.reward_death
            out_of_field = not (0 <= head[0] < self.n and 0 <= head[1] < self.n)
            death_cause = "Wand" if out_of_field else "Körper"
        else:
            # schlange bewegt sich
            self.snake.insert(0, head)

            if head == self.food:
                #apfel gegessen -> schlange bewegt sich, schwanz nicht
                self.score += 1
                self.steps_since_food = 0   # schrittezähler zurücksetzen
                reward = self.reward_food
                if len(self.snake) == self.n * self.n:
                    terminated = True       # feld voll -> gewonnen
                    death_cause = "Gewonnen"
                else:
                    self._place_food()      # neuen apfel setzen
            else:
                self.snake.pop()

            # timeout bei zuvielen schritten und keinen apfel gegessen
            if not terminated and self.steps_since_food >= self.max_steps_without_food:
                truncated = True
                death_cause = "Timeout"

        info = {
            "score": self.score,
            "length": len(self.snake),
            "death_cause": death_cause,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

# renderer
    def render(self):
        if self.render_mode == "human":
            return self._render_pygame()
        return self._render_text()
    
    def _render_text(self):
        grid = [["." for _ in range(self.n)] for _ in range(self.n)]
        grid[self.food[1]][self.food[0]] = "F"
        for i, (x,y) in enumerate(self.snake):
            if 0 <= x < self.n and 0 <= y < self.n:
                if i == 0:
                    grid[y][x] = "H" 
                else:
                    grid[y][x] = "o"
        return "\n".join(" ".join(row) for row in grid)

    def _render_pygame(self):
        size = self.n * self.cell_size
        if self._window is None:
            pygame.init()
            pygame.display.set_caption("Snake")
            self._window = pygame.display.set_mode((size,size))
            self._clock= pygame.time.Clock()

        self._window.fill((20,20,20))
        c = self.cell_size
        fx, fy = self.food
        pygame.draw.rect(self._window, (220,50,50), (fx * c + 3, fy * c + 3, c - 6, c - 6))
        for i, (x,y) in enumerate(self.snake): 
            if i == 0: 
                color = (120,255,120)
            else :
                color = (40,170,40)
            pygame.draw.rect(self._window, color, (x * c +1, y*c+1, c-2,c-2))

        pygame.event.pump()
        pygame.display.flip()
        self._clock.tick(self.fps)

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None
                                                  
