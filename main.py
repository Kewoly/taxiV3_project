"""
main.py
-------
Point d'entrée unique du projet Taxi-v3 RL.

Modes disponibles :
  train   — Lance l'entraînement complet
  eval    — Évalue un modèle sauvegardé
  demo    — Joue un épisode en mode visuel (terminal)

Exemples :
  python main.py train
  python main.py train --episodes 3000 --alpha 0.2
  python main.py eval
  python main.py eval --model training/checkpoints/qtable_ep1000.pkl
  python main.py demo
  python main.py demo --delay 0.5
"""

import argparse
import sys
import os


# ---------------------------------------------------------------------------
# Mode TRAIN
# ---------------------------------------------------------------------------

def run_train(args):
    from training.train import train, DEFAULTS

    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "mode"})
    train(cfg)


# ---------------------------------------------------------------------------
# Mode EVAL
# ---------------------------------------------------------------------------

def run_eval(args):
    from src.environment import make_env
    from src.agent import QLearningAgent
    from evaluation.evaluate import evaluate_silent, print_metrics, plot_eval_distribution

    model_path = args.model or "training/checkpoints/qtable_final.pkl"
    if not os.path.exists(model_path):
        print(f"[Erreur] Modèle introuvable : {model_path}")
        print("  Lancez d'abord : python main.py train")
        sys.exit(1)

    env = make_env()
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    env.close()

    agent.load(model_path)
    agent.epsilon = 0.0

    n_ep = args.episodes or 100
    print(f"\n[Eval] Évaluation sur {n_ep} épisodes...")
    metrics = evaluate_silent(agent, n_episodes=n_ep)
    print_metrics(metrics)

    os.makedirs("logs", exist_ok=True)
    rewards_list = []
    env2 = make_env()
    for _ in range(n_ep):
        s, _ = env2.reset()
        tot = 0
        for _ in range(200):
            a = agent.select_action(s)
            s, r, te, tr, _ = env2.step(a)
            tot += r
            if te or tr:
                break
        rewards_list.append(tot)
    env2.close()
    plot_eval_distribution(rewards_list, save_path="logs/eval_distribution.png")


# ---------------------------------------------------------------------------
# Mode DEMO
# ---------------------------------------------------------------------------

def run_demo(args):
    from src.environment import make_env
    from src.agent import QLearningAgent
    from src.interface import render_episode

    model_path = args.model or "training/checkpoints/qtable_final.pkl"
    if not os.path.exists(model_path):
        print(f"[Erreur] Modèle introuvable : {model_path}")
        print("  Lancez d'abord : python main.py train")
        sys.exit(1)

    env = make_env(render_mode="human")
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    agent.load(model_path)
    agent.epsilon = 0.0  # greedy pur pour la démo

    delay = args.delay if hasattr(args, "delay") and args.delay else 0.3
    render_episode(env, agent, max_steps=200, delay=delay)
    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Taxi-v3 RL — Q-Learning | Mode : train / eval / demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Lancer l'entraînement")
    p_train.add_argument("--episodes",       type=int,   help="Nombre d'épisodes (défaut: 2000)")
    p_train.add_argument("--alpha",          type=float, help="Taux d'apprentissage (défaut: 0.1)")
    p_train.add_argument("--gamma",          type=float, help="Facteur d'actualisation (défaut: 0.99)")
    p_train.add_argument("--epsilon",        type=float, help="Epsilon initial (défaut: 1.0)")
    p_train.add_argument("--epsilon_min",    type=float, help="Epsilon minimum (défaut: 0.01)")
    p_train.add_argument("--epsilon_decay",  type=float, help="Décroissance epsilon (défaut: 0.995)")
    p_train.add_argument("--checkpoint_every", type=int, help="Sauvegarder tous les N épisodes")
    p_train.add_argument("--log_dir",        type=str,   help="Dossier logs TensorBoard")

    # --- eval ---
    p_eval = subparsers.add_parser("eval", help="Évaluer un modèle sauvegardé")
    p_eval.add_argument("--model",    type=str, help="Chemin vers qtable_.pkl")
    p_eval.add_argument("--episodes", type=int, help="Épisodes d'évaluation (défaut: 100)")

    # --- demo ---
    p_demo = subparsers.add_parser("demo", help="Démo visuelle dans le terminal")
    p_demo.add_argument("--model", type=str,   help="Chemin vers qtable_.pkl")
    p_demo.add_argument("--delay", type=float, help="Délai entre pas en secondes (défaut: 0.3)")

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "demo":
        run_demo(args)


if __name__ == "__main__":
    main()
