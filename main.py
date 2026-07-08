"""Point d'entrée : sessions visuelles, benchmark et entraînements.

Exemples :
    uv run python main.py --mode play --ai search --seed 3
    uv run python main.py --mode train --generations 100 --curriculum
    uv run python main.py --mode train-dqn --episodes 800
    uv run python main.py --mode train-clone --samples 60000
    uv run python main.py --mode bench --episodes 25
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import TYPE_CHECKING

from config import (
    AI_TIME_BUDGET_MS,
    DEFAULT_GENERATIONS,
    DQN_EPISODES,
    DQN_MODEL_PATH,
    EPISODES_PER_EVAL,
    IMITATION_SAMPLES,
    MODEL_PATH,
    POLICY_MODEL_PATH,
    POPULATION_SIZE,
    TARGET_SCORE,
)
from engine import Engine
from ai_base import BaseAI, HeuristicAI
from ai_search import AdaptiveSearchAI, SearchAI
from ai_mcts import MCTSAI
from ai_genetic import GeneticTrainer, NeuralAI, save_genome
from ai_qlearning import DQNAI, DQNTrainer
from ai_hybrid import CloneAI, GuidedSearchAI, collect_dataset, train_policy

if TYPE_CHECKING:
    from ui import Renderer

# Agents jouables : robots (algorithmes), IA (apprentissage), hybride.
AI_CHOICES = ("heuristic", "search", "search+", "mcts", "nn", "dqn", "clone", "guided")

# Chemin de modèle par défaut des agents qui en chargent un.
DEFAULT_MODELS = {
    "nn": MODEL_PATH,
    "dqn": DQN_MODEL_PATH,
    "clone": POLICY_MODEL_PATH,
    "guided": POLICY_MODEL_PATH,
}

TRAIN_HINTS = {
    "nn": "--mode train",
    "dqn": "--mode train-dqn",
    "clone": "--mode train-clone",
    "guided": "--mode train-clone",
}


def _model_path(name: str, override: str | None) -> str:
    path = override or DEFAULT_MODELS[name]
    if not os.path.exists(path):
        raise SystemExit(
            f"modèle introuvable : {path} (lancer {TRAIN_HINTS[name]} d'abord)"
        )
    return path


def build_ai(name: str, model_override: str | None = None) -> BaseAI:
    if name == "heuristic":
        return HeuristicAI()
    if name == "search":
        return SearchAI()
    if name == "search+":
        return AdaptiveSearchAI()
    if name == "mcts":
        return MCTSAI()
    if name == "nn":
        return NeuralAI.from_file(_model_path(name, model_override))
    if name == "dqn":
        return DQNAI.from_file(_model_path(name, model_override))
    if name == "clone":
        return CloneAI.from_file(_model_path(name, model_override))
    if name == "guided":
        return GuidedSearchAI.from_file(_model_path(name, model_override))
    raise SystemExit(f"agent inconnu : {name}")


def run_episode(
    engine: Engine, ai: BaseAI, renderer: "Renderer | None" = None
) -> dict[str, float]:
    """Joue un épisode complet ; retourne les métriques (score, temps de décision)."""
    times_ms: list[float] = []
    interrupted = False
    while not engine.game_over:
        if renderer is not None and not renderer.handle_events():
            interrupted = True
            break
        t0 = time.perf_counter()
        move = ai.get_move(engine)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        engine.step(move)
        if renderer is not None:
            renderer.draw(engine, ai, times_ms[-1])
    return {
        "score": engine.score,
        "ticks": engine.tick,
        "won": float(engine.won),
        "interrupted": float(interrupted),
        "avg_ms": statistics.mean(times_ms) if times_ms else 0.0,
        "max_ms": max(times_ms) if times_ms else 0.0,
    }


# --------------------------------------------------------------------- modes

def cmd_play(args: argparse.Namespace) -> None:
    from ui import Renderer  # import différé : l'entraînement reste sans pygame

    ai = build_ai(args.ai, args.model)
    engine = Engine(seed=args.seed)
    renderer = Renderer(title=f"Crossy IA — {ai.name}")
    try:
        result = run_episode(engine, ai, renderer)
        if not result["interrupted"]:
            # écran de fin avec verdict, jusqu'à une touche ou la fermeture
            renderer.draw_end(engine, ai, result["avg_ms"])
            renderer.wait_until_dismissed()
    finally:
        renderer.close()
    if engine.won:
        verdict = "GAGNÉ"
    elif result["interrupted"]:
        verdict = "interrompu"
    else:
        verdict = f"MORT ({engine.death_cause})" if engine.death_cause else "MORT"
    print(
        f"[{ai.name}] seed={args.seed} : {verdict} | score {engine.score}/{TARGET_SCORE} "
        f"| {engine.tick} ticks | décision moy {result['avg_ms']:.2f} ms "
        f"/ max {result['max_ms']:.2f} ms (budget {AI_TIME_BUDGET_MS:.0f} ms)"
    )


def cmd_train(args: argparse.Namespace) -> None:
    print(
        f"neuroévolution : population={args.population}, générations={args.generations}, "
        f"épisodes/évaluation={args.episodes}, graine={args.seed}, "
        f"curriculum={'oui' if args.curriculum else 'non'}"
    )
    trainer = GeneticTrainer(
        population_size=args.population, episodes=args.episodes, seed=args.seed
    )
    t0 = time.perf_counter()
    best, fitness = trainer.evolve(args.generations, curriculum=args.curriculum)
    elapsed = time.perf_counter() - t0
    path = args.model or MODEL_PATH
    save_genome(best, path)
    print(
        f"terminé en {elapsed:.1f} s — meilleur score validé {fitness:.1f}, "
        f"modèle sauvegardé dans {path}"
    )


def cmd_train_dqn(args: argparse.Namespace) -> None:
    print(f"DQN : épisodes={args.episodes}, graine={args.seed}")
    trainer = DQNTrainer(seed=args.seed)
    t0 = time.perf_counter()
    net, score = trainer.train(episodes=args.episodes)
    elapsed = time.perf_counter() - t0
    path = args.model or DQN_MODEL_PATH
    net.save(path)
    print(
        f"terminé en {elapsed:.1f} s — meilleur score validé {score:.1f}, "
        f"modèle sauvegardé dans {path}"
    )


def cmd_train_clone(args: argparse.Namespace) -> None:
    print(f"imitation du planificateur : {args.samples} exemples")
    t0 = time.perf_counter()
    x, y = collect_dataset(samples=args.samples)
    net, accuracy = train_policy(x, y, seed=args.seed)
    elapsed = time.perf_counter() - t0
    path = args.model or POLICY_MODEL_PATH
    net.save(path)
    print(
        f"terminé en {elapsed:.1f} s — précision validation {accuracy * 100:.1f} %, "
        f"politique sauvegardée dans {path} (agents clone et guided)"
    )


def cmd_bench(args: argparse.Namespace) -> None:
    """Protocole de comparaison : mêmes graines pour tous les agents, sans affichage."""
    ais: list[BaseAI] = [HeuristicAI(), SearchAI(), AdaptiveSearchAI(), MCTSAI()]
    for name, ctor in (("nn", NeuralAI), ("dqn", DQNAI), ("clone", CloneAI), ("guided", GuidedSearchAI)):
        path = DEFAULT_MODELS[name]
        if os.path.exists(path):
            ais.append(ctor.from_file(path))
        else:
            print(f"({name} ignoré : modèle {path} absent — {TRAIN_HINTS[name]})")

    seeds = [10_000 + i for i in range(args.episodes)]
    print(f"{'agent':<10} {'famille':<8} {'score moy':>10} {'médiane':>8} {'max':>5} "
          f"{'victoires':>10} {'ms moy':>8} {'ms max':>8}")
    for ai in ais:
        scores: list[int] = []
        wins = 0
        avg_ms: list[float] = []
        max_ms = 0.0
        for seed in seeds:
            ai.reset()
            result = run_episode(Engine(seed=seed), ai)
            scores.append(int(result["score"]))
            wins += int(result["won"])
            avg_ms.append(result["avg_ms"])
            max_ms = max(max_ms, result["max_ms"])
        print(
            f"{ai.name:<10} {ai.family:<8} {statistics.mean(scores):>10.1f} "
            f"{statistics.median(scores):>8.0f} {max(scores):>5} "
            f"{wins:>5}/{len(seeds):<4} {statistics.mean(avg_ms):>8.3f} {max_ms:>8.2f}"
        )


# ----------------------------------------------------------------------- CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crossy Road — simulateur et comparateur de robots et d'IA"
    )
    parser.add_argument(
        "--mode",
        choices=("play", "train", "train-dqn", "train-clone", "bench"),
        default="play",
        help="play : session visuelle ; train : neuroévolution ; train-dqn : "
             "Q-learning ; train-clone : imitation du planificateur ; "
             "bench : comparatif headless multi-graines",
    )
    parser.add_argument("--ai", choices=AI_CHOICES, default="search",
                        help="agent évalué en mode play")
    parser.add_argument("--seed", type=int, default=0,
                        help="graine de la partie / de l'entraînement")
    parser.add_argument("--model", default=None,
                        help="chemin de modèle .npz (défaut : propre à chaque agent)")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS,
                        help="générations du GA (mode train)")
    parser.add_argument("--curriculum", action="store_true",
                        help="GA : commencer sur des mondes sans rivière")
    parser.add_argument("--population", type=int, default=POPULATION_SIZE,
                        help="taille de population du GA (mode train)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="épisodes : par évaluation GA (défaut 3), d'entraînement "
                             "DQN (défaut 800) ou par agent au bench (défaut 20)")
    parser.add_argument("--samples", type=int, default=IMITATION_SAMPLES,
                        help="exemples collectés pour l'imitation (mode train-clone)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes is None:
        args.episodes = {
            "train": EPISODES_PER_EVAL,
            "train-dqn": DQN_EPISODES,
        }.get(args.mode, 20)
    if args.mode == "play":
        cmd_play(args)
    elif args.mode == "train":
        cmd_train(args)
    elif args.mode == "train-dqn":
        cmd_train_dqn(args)
    elif args.mode == "train-clone":
        cmd_train_clone(args)
    else:
        cmd_bench(args)


if __name__ == "__main__":
    main()
