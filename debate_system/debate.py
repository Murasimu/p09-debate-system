"""Debate orchestration: propose -> critique -> rebut -> re-critique.

Rounds keep the full transcript so the aggregator sees how positions
evolved (concessions matter for the final synthesis).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .agents import Proposal, Proposer

LLM = Callable[[list[dict[str, str]]], str]


@dataclass
class Round:
    number: int
    proposals: list[Proposal] = field(default_factory=list)
    rebuttals: dict[str, str] = field(default_factory=dict)

    def transcript(self) -> str:
        parts = [f"Round {self.number}"]
        for p in self.proposals:
            parts.append(f"{p.proposer} (score {p.critique_score}): {p.text}")
        for name, text in self.rebuttals.items():
            parts.append(f"{name} rebuttal: {text}")
        return "\n".join(parts)


@dataclass
class DebateConfig:
    rounds: int = 2
    critique_weight: float = 0.7   # vote weight = critique^w * confidence^(1-w)


@dataclass
class Debate:
    question: str
    proposers: list[Proposer]
    critic_llm: LLM
    config: DebateConfig = field(default_factory=DebateConfig)
    rounds: list[Round] = field(default_factory=list)

    def run(self) -> list[Round]:
        from .agents import Critic
        critic = Critic(self.critic_llm)

        for r in range(1, self.config.rounds + 1):
            round_ = Round(number=r)

            # 1) proposals (round 1: fresh; later: may incorporate rebuttals)
            for p in self.proposers:
                context = self._context_so_far() if r > 1 else ""
                round_.proposals.append(p.propose(self.question, context))

            # 2) critique each proposal (quality score + justification)
            for prop in round_.proposals:
                score, text = critic.critique(self.question, prop)
                prop.critique_score, prop.critique = score, text
                prop.confidence = critic.self_confidence(self.question, prop)

            # 3) rebuttals (only in rounds before the last)
            if r < self.config.rounds:
                for proposer, prop in zip(self.proposers, round_.proposals):
                    others = [o for o in round_.proposals if o is not prop]
                    round_.rebuttals[prop.proposer] = proposer.rebut(
                        self.question, prop, others)

            self.rounds.append(round_)
        return self.rounds

    def _context_so_far(self) -> str:
        return "\n\n".join(r.transcript() for r in self.rounds)

    def final_proposals(self) -> list[Proposal]:
        return self.rounds[-1].proposals if self.rounds else []
