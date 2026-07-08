"""Point d'entrée : sessions d'évaluation visuelles, benchmark et entraînement.

Exemples :
    uv run python main.py --mode play --ai search --seed 3
    uv run python main.py --mode train --generations 60
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
    EPISODES_PER_EVAL,
    MODEL_PATH,
    POPULATION_SIZE,
    TARGET_SCORE,
)
from engine import Engine
from ai_base import BaseAI, HeuristicAI
from ai_search import AdaptiveSearchAI, SearchAI
from ai_genetic import GeneticTrainer, NeuralAI, save_genome

if TYPE_CHECKING:
    from ui import Renderer

AI_CHOICES = ("heuristic", "search", "search+", "nn")


def build_ai(name: str, model_path: str) -> BaseAI:
    if name == "heuristic":
        return HeuristicAI()
    if name == "search":
        return SearchAI()
    if name == "search+":
        return AdaptiveSearchAI()
    if name == "nn":
        if not os.path.exists(model_path):
            raise SystemExit(
                f"modèle introuvable : {model_path} (lancer --mode train d'abord)"
            )
        return NeuralAI.from_file(model_path)
    raise SystemExit(f"IA inconnue : {name}")


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
        f"entraînement : population={args.population}, générations={args.generations}, "
        f"épisodes/évaluation={args.episodes}, graine={args.seed}"
    )
    trainer = GeneticTrainer(
        population_size=args.population, episodes=args.episodes, seed=args.seed
    )
    t0 = time.perf_counter()
    best, fitness = trainer.evolve(args.generations)
    elapsed = time.perf_counter() - t0
    save_genome(best, args.model)
    print(
        f"terminé en {elapsed:.1f} s — meilleure fitness validée {fitness:.1f}, "
        f"modèle sauvegardé dans {args.model}"
    )


def cmd_bench(args: argparse.Namespace) -> None:
    """Protocole de comparaison : mêmes graines pour toutes les IA, sans affichage."""
    ais: list[BaseAI] = [HeuristicAI(), SearchAI(), AdaptiveSearchAI()]
    if os.path.exists(args.model):
        ais.append(NeuralAI.from_file(args.model))
    else:
        print(f"(nn ignorée : modèle {args.model} absent)")

    seeds = [10_000 + i for i in range(args.episodes)]
    print(f"{'IA':<10} {'score moy':>10} {'médiane':>8} {'max':>5} {'victoires':>10} "
          f"{'ms moy':>8} {'ms max':>8}")
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
            f"{ai.name:<10} {statistics.mean(scores):>10.1f} "
            f"{statistics.median(scores):>8.0f} {max(scores):>5} "
            f"{wins:>5}/{len(seeds):<4} {statistics.mean(avg_ms):>8.3f} {max_ms:>8.2f}"
        )


# ----------------------------------------------------------------------- CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crossy Road — simulateur et comparateur d'IA"
    )
    parser.add_argument(
        "--mode", choices=("play", "train", "bench"), default="play",
        help="play : session visuelle ; train : neuroévolution headless ; "
             "bench : comparatif headless multi-graines",
    )
    parser.add_argument("--ai", choices=AI_CHOICES, default="search",
                        help="IA évaluée en mode play")
    parser.add_argument("--seed", type=int, default=0, help="graine de la partie / du GA")
    parser.add_argument("--model", default=MODEL_PATH, help="chemin du modèle .npz")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS,
                        help="générations du GA (mode train)")
    parser.add_argument("--population", type=int, default=POPULATION_SIZE,
                        help="taille de population du GA (mode train)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="épisodes par évaluation (train, défaut 3) "
                             "ou par IA (bench, défaut 20)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes is None:
        args.episodes = EPISODES_PER_EVAL if args.mode == "train" else 20
    if args.mode == "play":
        cmd_play(args)
    elif args.mode == "train":
        cmd_train(args)
    else:
        cmd_bench(args)


if __name__ == "__main__":
    main()
