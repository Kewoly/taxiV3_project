import pygame
import numpy as np
from env import SimpleTaxiEnv
from stable_baselines3 import DQN

# Configuration Pygame
CELL_SIZE = 60
GRID_WIDTH = 10
GRID_HEIGHT = 8
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)


def draw_grid(screen):
    """Dessine la grille."""
    for x in range(0, SCREEN_WIDTH + 1, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT + 1, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (SCREEN_WIDTH, y))


def draw_walls(screen, env):
    """Dessine les remparts fixes."""
    for (x, y) in env.walls:
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, GRAY, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)


def draw_taxi(screen, env):
    """Dessine le taxi."""
    x = env.taxi_x * CELL_SIZE + CELL_SIZE // 2
    y = env.taxi_y * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, YELLOW, (x, y), CELL_SIZE // 3)
    pygame.draw.circle(screen, BLACK, (x, y), CELL_SIZE // 3, 2)


def draw_destinations(screen, env):
    """Dessine les destinations des passagers."""
    for i, p in enumerate(env.passengers):
        if not p.get("active", True): continue
        dx, dy = p["dest_x"], p["dest_y"]
        rect = pygame.Rect(dx * CELL_SIZE, dy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, GREEN, rect, 2)
        font = pygame.font.Font(None, 20)
        text = font.render(f"D{i+1}", True, GREEN)
        screen.blit(text, (dx * CELL_SIZE + 5, dy * CELL_SIZE + 5))


def draw_passengers(screen, env):
    """Dessine les passagers."""
    for i, p in enumerate(env.passengers):
        if not p.get("active", True): continue
        if not p["in_taxi"]:
            x = p["x"] * CELL_SIZE + CELL_SIZE // 2
            y = p["y"] * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, RED, (x, y), CELL_SIZE // 4)
            font = pygame.font.Font(None, 20)
            text = font.render(str(i+1), True, WHITE)
            screen.blit(text, (x - 5, y - 5))


def draw_info(screen, env, episode, total_reward, action_names):
    """Affiche les informations."""
    font = pygame.font.Font(None, 28)
    info_text = [
        f"Episode: {episode}",
        f"Steps: {env.steps_taken}/{env.max_steps}",
        f"Delivered: {env.delivered}/{env.num_passengers}",
        f"Reward: {total_reward:.1f}",
    ]
    
    for i, text in enumerate(info_text):
        surface = font.render(text, True, BLACK)
        screen.blit(surface, (10, SCREEN_HEIGHT + 10 + i * 30))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 150))
    pygame.display.set_caption("Taxi Environment - DQN")
    clock = pygame.time.Clock()
    
    env = SimpleTaxiEnv()
    
    # Charger le modèle entraîné
    try:
        agent = DQN.load("models/dqn_taxi", env=env, render_mode="human")
        print("✓ Modèle chargé")
    except:
        print("⚠ Modèle non trouvé, utilisation d'actions aléatoires")
        agent = None
    
    action_names = ["↓ Sud", "↑ Nord", "→ Est", "← Ouest", "📦 Prendre", "📍 Déposer"]
    
    episode = 0
    running = True
    paused = False
    obs, _ = env.reset()
    total_reward = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    episode = 0
                    obs, _ = env.reset()
                    total_reward = 0
        
        if not paused:
            # Prédiction
            if agent:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # Fin d'épisode
            if terminated or truncated:
                episode += 1
                obs, _ = env.reset()
                total_reward = 0
        
        # Rendu
        screen.fill(WHITE)
        draw_grid(screen)
        draw_walls(screen, env)
        draw_destinations(screen, env)
        draw_taxi(screen, env)
        draw_passengers(screen, env)
        draw_info(screen, env, episode, total_reward, action_names)
        
        # Instructions
        font = pygame.font.Font(None, 20)
        instructions = [
            "SPACE: Pause/Resume | R: Reset | Q: Quit",
        ]
        for i, text in enumerate(instructions):
            surface = font.render(text, True, GRAY)
            screen.blit(surface, (10, SCREEN_HEIGHT + 130))
        
        pygame.display.flip()
        clock.tick(4)
    
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
