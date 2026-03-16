"""
large_taxi.py
--------------
Environnement Taxi custom 15×10 avec multi-passagers.
Optimisé pour minimiser les pas et maximiser les livraisons.

Grille : 15 colonnes (largeur) × 10 lignes (hauteur)
Passagers : 2 passagers à ramasser et livrer
Actions : 0=Sud, 1=Nord, 2=Est, 3=Ouest, 4=Prendre, 5=Déposer
État : Box contenant [taxi_x, taxi_y, passenger1_x, passenger1_y, passenger1_in_taxi, 
                       passenger2_x, passenger2_y, passenger2_in_taxi, ...]
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LargeTaxiEnv(gym.Env):
    """Environnement Taxi 15×10 avec 2 passagers."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_width=10, grid_height=8, n_passengers=2):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.n_passengers = n_passengers
        self.render_mode = render_mode

        # Positions fixes des zones de pickup/dropoff
        self.pickup_zones = [
            (1, 1),    # Passager 1 - zone rouge
            (8, 6),    # Passager 2 - zone bleue
        ]
        self.dropoff_zones = [
            (8, 1),    # Destination passager 1 - zone verte
            (1, 6),    # Destination passager 2 - zone jaune
        ]

        # Observation space NORMALISÉE : [taxi_x_norm, taxi_y_norm, p1_x_norm, p1_y_norm, p1_in_taxi, 
        #                                  p1_dest_x_norm, p1_dest_y_norm, p2_x_norm, p2_y_norm, p2_in_taxi, 
        #                                  p2_dest_x_norm, p2_dest_y_norm]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7 + 5 * (n_passengers - 1),),
            dtype=np.float32,
        )

        # Action space : 6 actions
        self.action_space = spaces.Discrete(6)
        self.action_labels = {
            0: "Sud",
            1: "Nord",
            2: "Est",
            3: "Ouest",
            4: "Prendre",
            5: "Déposer",
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Taxi commence au centre
        self.taxi_x = self.grid_width // 2
        self.taxi_y = self.grid_height // 2

        # Passagers aux zones de pickup
        self.passengers = []
        for i in range(self.n_passengers):
            self.passengers.append({
                "x": self.pickup_zones[i][0],
                "y": self.pickup_zones[i][1],
                "in_taxi": False,
                "dest_x": self.dropoff_zones[i][0],
                "dest_y": self.dropoff_zones[i][1],
            })

        self.steps_taken = 0
        self.max_steps = 300
        self.delivered = 0

        return self._get_obs(), {}

    def _get_obs(self):
        obs = [self.taxi_x / self.grid_width, self.taxi_y / self.grid_height]
        for p in self.passengers:
            obs.extend([p["x"] / self.grid_width, p["y"] / self.grid_height, float(p["in_taxi"]), 
                       p["dest_x"] / self.grid_width, p["dest_y"] / self.grid_height])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps_taken += 1
        reward = -0.1  # Pénalité légère par pas

        if action == 0:  # Sud
            self.taxi_y = min(self.taxi_y + 1, self.grid_height - 1)
        elif action == 1:  # Nord
            self.taxi_y = max(self.taxi_y - 1, 0)
        elif action == 2:  # Est
            self.taxi_x = min(self.taxi_x + 1, self.grid_width - 1)
        elif action == 3:  # Ouest
            self.taxi_x = max(self.taxi_x - 1, 0)
        elif action == 4:  # Prendre
            for p in self.passengers:
                if not p["in_taxi"] and p["x"] == self.taxi_x and p["y"] == self.taxi_y:
                    p["in_taxi"] = True
                    reward += 1.0  # Bonus pour pickup
                    break
        elif action == 5:  # Déposer
            for p in self.passengers:
                if p["in_taxi"] and p["dest_x"] == self.taxi_x and p["dest_y"] == self.taxi_y:
                    p["in_taxi"] = False
                    p["x"] = p["dest_x"]
                    p["y"] = p["dest_y"]
                    self.delivered += 1
                    reward += 10.0  # Bonus pour dropoff réussi
                    break

        terminated = self.delivered == self.n_passengers
        truncated = self.steps_taken >= self.max_steps
        done = terminated or truncated

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        grid = [["." for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        # Zones de pickup
        for i, (px, py) in enumerate(self.pickup_zones):
            grid[py][px] = f"P{i+1}"

        # Zones de dropoff
        for i, (dx, dy) in enumerate(self.dropoff_zones):
            grid[dy][dx] = f"D{i+1}"

        # Passagers
        for i, p in enumerate(self.passengers):
            if not p["in_taxi"]:
                grid[p["y"]][p["x"]] = f"p{i+1}"

        # Taxi
        grid[self.taxi_y][self.taxi_x] = "T"

        print("\n" + "=" * (self.grid_width * 2 + 1))
        for row in grid:
            print(" ".join(row))
        print(f"Étape: {self.steps_taken} | Livrés: {self.delivered}/{self.n_passengers}")
        print("=" * (self.grid_width * 2 + 1) + "\n")

    def _render_rgb_array(self):
        # Rendu RGB simple (optionnel, pour visualisation)
        pass

    def close(self):
        pass
