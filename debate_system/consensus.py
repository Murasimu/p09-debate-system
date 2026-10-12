"""Consensus: weighted voting over the final-round proposals.

Vote weight = critique_score^w * confidence^(1-w) with w =
critique_weight (default 0.7): argument quality dominates, self-reported
confidence refines. A quorum rule guards against a split house: if no
proposal crosses the quorum threshold, consensus returns status
"no_majority" and the aggregator must synthesize rather than crown.

Voting itself is done by an LLM voter (each voter ranks proposals), or
deterministically by critique score when no voter is supplied -- the
tally mechanics are identical either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .agents import Proposal

LLM = Callable[[list[dict[str, str]]], str]


@dataclass
class Consensus:
    winner: str | None
    status: str            # "decided" | "no_majority"
    weights: dict[str, float]
    votes: dict[str, int]
    quorum: float
    top_score: float


def tally(proposals: list[Proposal], votes: dict[str, int],
          critique_weight: float = 0.7, quorum: float = 0.5) -> Consensus:
    """Combine votes (counts per proposer) with quality weights."""
    if not proposals:
        return Consensus(None, "no_majority", {}, {}, quorum, 0.0)

    weights = {}
    for p in proposals:
        c = p.critique_score if p.critique_score is not None else 0.5
        conf = p.confidence if p.confidence is not None else 0.5
        weights[p.proposer] = (c ** critique_weight) * (conf ** (1 - critique_weight))

    total_votes = sum(votes.values()) or 1
    total_weight = sum(weights.values()) or 1.0

    # weighted vote share per proposer
    share = {}
    for p in proposals:
        v = votes.get(p.proposer, 0) / total_votes
        w = weights[p.proposer] / total_weight
        share[p.proposer] = v * 0.5 + w * 0.5

    winner = max(share, key=share.get)  # type: ignore[arg-type]
    top = share[winner]
    status = "decided" if top >= quorum else "no_majority"
    return Consensus(
        winner=winner if status == "decided" else None,
        status=status, weights={k: round(v, 4) for k, v in weights.items()},
        votes=votes, quorum=quorum, top_score=round(top, 4),
    )


def llm_votes(voter: LLM, question: str, proposals: list[Proposal]) -> dict[str, int]:
    """A fresh LLM voter picks the best proposal (one vote each)."""
    listing = "\n".join(
        f"{i+1}. {p.proposer}: {p.text[:250]}" for i, p in enumerate(proposals))
    raw = voter([
        {"role": "system", "content":
            "You are an impartial judge. Pick the single best answer to the "
            "question. Reply with ONLY the proposer's name."},
        {"role": "user", "content":
            f"Question: {question}\n\nProposals:\n{listing}"},
    ])
    votes: dict[str, int] = {p.proposer: 0 for p in proposals}
    m = re.search(r"[A-Za-z-]+", raw.strip())
    if m and m.group(0) in votes:
        votes[m.group(0)] += 1
    else:
        # fallback: fuzzy match on names contained in the reply
        for name in votes:
            if name.lower() in raw.lower():
                votes[name] += 1
                break
    return votes
