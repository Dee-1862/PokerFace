"""Poker hand equity engine.

Ported out of the legacy ``unified_ar_system.py`` so the FastAPI server can
compute win-probability + opponent-hand distributions for the saved hand
and visible board.

Public entry point: :func:`calculate_equity` -> ``(equity_pct, outs, breakdown)``
where ``breakdown = {'my_hands': [...], 'opp_hands': [...]}`` and each entry is
``{'name': str, 'prob': float, 'cards': [treys-string, ...]}``.

Heavyweight: a full flop enumeration is ~150k evaluations. The server is
expected to cache the result keyed by ``(hand, board)`` so this only runs when
state actually changes.
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from itertools import combinations

try:
    from treys import Card, Deck, Evaluator
    EVALUATOR = Evaluator()
except Exception as e:                          # pragma: no cover - optional dep
    print(f"[poker_engine] treys unavailable: {e}")
    Card = Deck = Evaluator = None              # type: ignore[assignment]
    EVALUATOR = None


_PREFLOP: dict[str, float] = {}
_PREFLOP_PATH = os.path.join(os.path.dirname(__file__), 'preflop_equity.csv')
try:
    with open(_PREFLOP_PATH, 'r') as f:
        for row in csv.DictReader(f):
            _PREFLOP[row['hand']] = float(row['equity'])
    print(f"[poker_engine] preflop table loaded ({len(_PREFLOP)} hands)")
except Exception as e:                          # pragma: no cover
    print(f"[poker_engine] preflop table missing: {e}")


_RANK_ORDER = 'AKQJT98765432'


def _normalize_hand(hand: list[str]) -> str | None:
    """Map ['Ah', 'Kc'] -> 'AKo' / 'AKs' / 'AA' for preflop lookup."""
    if len(hand) != 2:
        return None
    ranks = [c[0] for c in hand]
    suits = [c[1] for c in hand]
    sorted_ranks = sorted(ranks, key=lambda r: _RANK_ORDER.index(r))
    suited = 's' if suits[0] == suits[1] else 'o'
    if sorted_ranks[0] == sorted_ranks[1]:
        return sorted_ranks[0] + sorted_ranks[1]
    return sorted_ranks[0] + sorted_ranks[1] + suited


def _best_5(cards_int: list) -> list[str]:
    """Best 5-card subset of >=5 ints, returned as treys strings."""
    if EVALUATOR is None or not cards_int:
        return [Card.int_to_str(c) for c in (cards_int or [])[:5]] if Card else []
    best_score = float('inf')
    best = None
    for five in combinations(cards_int, 5):
        score = EVALUATOR.evaluate(list(five), [])
        if score < best_score:
            best_score = score
            best = list(five)
    return [Card.int_to_str(c) for c in (best or cards_int[:5])]


def _evaluate(my_hand: list[str], board: list[str], cards_needed: int):
    """Shared core for flop / turn / river enumeration. Returns
    ``(equity_pct, breakdown)``."""
    hero = [Card.new(c) for c in my_hand]
    board_int = [Card.new(c) for c in board]
    deck = Deck()
    for c in hero + board_int:
        if c in deck.cards:
            deck.cards.remove(c)

    wins = 0
    total = 0
    hero_counts: dict[str, int]  = defaultdict(int)
    opp_counts:  dict[str, int]  = defaultdict(int)
    hero_examples: dict[str, list[str]] = {}
    opp_examples:  dict[str, list[str]] = {}

    if cards_needed == 0:
        # River - just iterate all opponent holdings against the fixed board.
        runouts = [()]
    else:
        runouts = list(combinations(deck.cards, cards_needed))

    for runout in runouts:
        remaining = [c for c in deck.cards if c not in runout]
        opp_combos = list(combinations(remaining, 2))
        if len(opp_combos) > 200:
            opp_combos = random.sample(opp_combos, 200)

        full_board = board_int + list(runout)
        for opp in opp_combos:
            total += 1
            try:
                hero_score = EVALUATOR.evaluate(full_board, hero)
                opp_score  = EVALUATOR.evaluate(full_board, list(opp))
                hero_class = EVALUATOR.class_to_string(EVALUATOR.get_rank_class(hero_score))
                opp_class  = EVALUATOR.class_to_string(EVALUATOR.get_rank_class(opp_score))
                if hero_score < opp_score:
                    wins += 1
                    hero_counts[hero_class] += 1
                    if hero_class not in hero_examples:
                        hero_examples[hero_class] = _best_5(hero + full_board)
                else:
                    opp_counts[opp_class] += 1
                    if opp_class not in opp_examples:
                        opp_examples[opp_class] = _best_5(list(opp) + full_board)
            except Exception:
                pass

    eq = (wins / total * 100.0) if total > 0 else 0.0

    def _build(counts, examples):
        out = []
        for name, cnt in counts.items():
            prob = cnt / total if total > 0 else 0.0
            if prob >= 0.01:
                out.append({
                    'name':  name,
                    'prob':  prob,
                    'cards': examples.get(name, [])[:7],
                })
        out.sort(key=lambda x: x['prob'], reverse=True)
        return out

    return eq, {
        'my_hands':  _build(hero_counts, hero_examples),
        'opp_hands': _build(opp_counts,  opp_examples),
    }


def calculate_equity(my_hand: list[str], board: list[str]):
    """Top-level API. Returns ``(equity_pct, outs, {'my_hands':..,'opp_hands':..})``.

    Outs aren't enumerated yet - kept as 0 to match the legacy contract.
    """
    empty = (0.0, 0, {'my_hands': [], 'opp_hands': []})
    if EVALUATOR is None or not my_hand or len(my_hand) != 2:
        return empty
    n = len(board)
    try:
        if n == 0:
            key = _normalize_hand(my_hand)
            if key and key in _PREFLOP:
                eq = _PREFLOP[key]
                is_pair = my_hand[0][0] == my_hand[1][0]
                breakdown = {
                    'my_hands': [{
                        'name':  'Pair' if is_pair else 'High Card',
                        'prob':  1.0,
                        'cards': list(my_hand),
                    }],
                    # Reasonable preflop opponent priors (rough public stats):
                    # any pair ~6%, otherwise high-card.
                    'opp_hands': [
                        {'name': 'Pair',      'prob': 0.06, 'cards': ['As', 'Ac']},
                        {'name': 'High Card', 'prob': 0.94, 'cards': ['Ah', 'Kh']},
                    ],
                }
                return eq, 0, breakdown
            return 50.0, 0, {'my_hands': [], 'opp_hands': []}
        if n in (3, 4):
            eq, breakdown = _evaluate(my_hand, board, cards_needed=5 - n)
            return eq, 0, breakdown
        if n == 5:
            eq, breakdown = _evaluate(my_hand, board, cards_needed=0)
            return eq, 0, breakdown
        return empty
    except Exception as e:                       # pragma: no cover
        print(f"[poker_engine] calculate_equity error: {e}")
        return empty
