import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame


class SimpleTaxiEnv(gym.Env):
    """Environnement Taxi 10×8 avec remparts et positions aléatoires."""

    def __init__(self, render_mode=None, num_passengers=2):
        self.grid_width = 10
        self.grid_height = 8
        self.render_mode = render_mode
        # Nombre max de passagers ; à chaque reset on tirera 1 ou 2
        self.max_passengers = num_passengers

        # Remparts fixes
        self.walls = {
            (2, 2), (2, 3), (2, 4),  # Mur vertical 1
            (7, 1), (7, 2), (7, 3),  # Mur vertical 2
            (4, 5), (5, 5), (6, 5),  # Mur horizontal 1
            (1, 7), (2, 7), (3, 7),  # Mur horizontal 2
        }

        # Observation space normalisée [0, 1]
        # taxi_x, taxi_y + passagers (6 chacun: x, y, in_taxi, dest_x, dest_y, active) + masque des remparts
        obs_size = 2 + (self.max_passengers * 6) + (self.grid_width * self.grid_height)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Action space : 6 actions
        self.action_space = spaces.Discrete(6)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.taxi_x = int(self.grid_width // 2)
        self.taxi_y = int(self.grid_height // 2)

        # Tirer aléatoirement 1 ou 2 passagers (puis padding jusqu'à max_passengers)
        self.num_passengers = int(self.np_random.integers(1, self.max_passengers + 1))
        self.passengers = []

        # Actifs
        for _ in range(self.num_passengers):
            pickup_x = self.np_random.integers(0, self.grid_width)
            pickup_y = self.np_random.integers(0, self.grid_height)
            while (pickup_x, pickup_y) in self.walls:
                pickup_x = self.np_random.integers(0, self.grid_width)
                pickup_y = self.np_random.integers(0, self.grid_height)

            dest_x = self.np_random.integers(0, self.grid_width)
            dest_y = self.np_random.integers(0, self.grid_height)
            while (dest_x, dest_y) in self.walls or (dest_x == pickup_x and dest_y == pickup_y):
                dest_x = self.np_random.integers(0, self.grid_width)
                dest_y = self.np_random.integers(0, self.grid_height)

            self.passengers.append({
                "x": pickup_x,
                "y": pickup_y,
                "in_taxi": False,
                "dest_x": dest_x,
                "dest_y": dest_y,
                "active": True,
            })

        # Padding passagers inactifs pour conserver la taille d'observation
        while len(self.passengers) < self.max_passengers:
            self.passengers.append({
                "x": 0,
                "y": 0,
                "in_taxi": False,
                "dest_x": 0,
                "dest_y": 0,
                "active": False,
            })

        self.steps_taken = 0
        self.max_steps = 500
        self.delivered = 0
        self._last_action = None
        self._repeat_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        obs = [self.taxi_x / self.grid_width, self.taxi_y / self.grid_height]
        for p in self.passengers:
            obs.extend([
                p["x"] / self.grid_width,
                p["y"] / self.grid_height, 
                float(p["in_taxi"]),
                p["dest_x"] / self.grid_width,
                p["dest_y"] / self.grid_height,
                float(p.get("active", False))
            ])
        # Masque des remparts (aplati ligne par ligne)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                obs.append(1.0 if (x, y) in self.walls else 0.0)
        return np.array(obs, dtype=np.float32)

    def _current_target_distance(self):
        """Distance de Manhattan vers la cible courante (passager à prendre ou destination du passager embarqué)."""
        # Si un passager actif est dans le taxi, la cible est sa destination
        for p in self.passengers:
            if p.get("active") and p["in_taxi"]:
                return abs(self.taxi_x - p["dest_x"]) + abs(self.taxi_y - p["dest_y"])
        # Sinon cible = passager actif non embarqué le plus proche
        best = None
        for p in self.passengers:
            if p.get("active") and not p["in_taxi"]:
                d = abs(self.taxi_x - p["x"]) + abs(self.taxi_y - p["y"])
                if best is None or d < best:
                    best = d
        return best if best is not None else 0
    
    def _can_move_to(self, x, y):
        """Vérifie si on peut se déplacer à (x, y)."""
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
        if (x, y) in self.walls:
            return False
        return True

    def step(self, action):
        self.steps_taken += 1
        reward = -0.001  # pénalité très légère par pas

        # Calcul du déplacement proposé
        new_x, new_y = self.taxi_x, self.taxi_y
        if action == 0:  # Sud
            new_y += 1
        elif action == 1:  # Nord
            new_y -= 1
        elif action == 2:  # Est
            new_x += 1
        elif action == 3:  # Ouest
            new_x -= 1

        # Appliquer le déplacement ou pénaliser la collision
        if action in [0, 1, 2, 3]:
            if self._can_move_to(new_x, new_y):
                self.taxi_x, self.taxi_y = new_x, new_y
            else:
                reward -= 0.05  # collision rempart/bord

        elif action == 4:  # Prendre
            for p in self.passengers:
                if not p.get("active"): continue
                if not p["in_taxi"] and p["x"] == self.taxi_x and p["y"] == self.taxi_y:
                    p["in_taxi"] = True
                    reward += 50.0
                    break
            else:
                # Aucun passager à prendre au bon endroit
                reward -= 0.5
        elif action == 5:  # Déposer
            for p in self.passengers:
                if p["in_taxi"] and p["dest_x"] == self.taxi_x and p["dest_y"] == self.taxi_y:
                    p["in_taxi"] = False
                    p["active"] = False
                    self.delivered += 1
                    reward += 200.0
                    break
            else:
                # Aucun passager à déposer au bon endroit
                reward -= 0.5

        # Shaping simple : récompense pour se rapprocher de la cible
        dist_new = self._current_target_distance()
        if dist_new > 0:
            # Seulement récompenser si on se rapproche (pas de pénalité si on s'éloigne)
            # Cela laisse l'agent explorer sans être trop pénalisé
            pass
        
        terminated = self.delivered == self.num_passengers
        truncated = self.steps_taken >= self.max_steps

        self._last_action = action
        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        pass
