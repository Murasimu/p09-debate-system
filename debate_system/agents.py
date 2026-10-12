"""Debate participants: proposers with distinct stances + a critic.

A Proposer is an LLM + a persona (name + stance instruction). Stances are
what make debate real: "cost-minimizer", "risk-averse", "speed-first"
produce genuinely different proposals instead of three echoes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

LLM = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class Persona:
    name: str
    stance: str  # instruction fragment shaping this proposer's bias


DEFAULT_PERSONAS = (
    Persona("Cost-Minimizer", "You prioritize minimizing cost and resource use."),
    Persona("Risk-Averse", "You prioritize safety, robustness, and avoiding failure."),
    Persona("Speed-First", "You prioritize speed of delivery and iteration."),
    Persona("Quality-Maximizer", "You prioritize long-term quality and maintainability."),
)


@dataclass
class Proposal:
    proposer: str
    text: str
    critique_score: float | None = None  # set by the critic (0..1)
    critique: str = ""
    confidence: float | None = None      # self-reported confidence (0..1)


@dataclass
class Proposer:
    persona: Persona
    llm: LLM

    def propose(self, question: str, context: str = "") -> Proposal:
        messages = [
            {"role": "system", "content":
                f"You are {self.persona.name}, an expert advisor in a debate. "
                f"{self.persona.stance} Propose ONE concrete solution with 2-3 "
                "supporting reasons. Be specific and brief (max 120 words)."},
            {"role": "user", "content": f"Question: {question}\n{context}".strip()},
        ]
        return Proposal(proposer=self.persona.name, text=self.llm(messages).strip())

    def rebut(self, question: str, own: Proposal, others: list[Proposal]) -> str:
        other_text = "\n".join(f"- {o.proposer}: {o.text[:200]}" for o in others)
        messages = [
            {"role": "system", "content":
                f"You are {self.persona.name}. {self.persona.stance} "
                "Respond to the other proposals: defend your solution where "
                "it is stronger, and concede specific points where they are "
                "right. Max 100 words."},
            {"role": "user", "content":
                f"Question: {question}\n\nYour proposal:\n{own.text}\n\n"
                f"Other proposals:\n{other_text}"},
        ]
        return self.llm(messages).strip()


@dataclass
class Critic:
    """Evaluates proposals on argument quality -- not on which answer the
    critic personally prefers. The score weights the vote."""

    llm: LLM
    rubric: str = "reasoning, evidence, feasibility, and honesty about trade-offs"

    def critique(self, question: str, proposal: Proposal) -> tuple[float, str]:
        messages = [
            {"role": "system", "content":
                "You are an impartial debate critic. Score the proposal's "
                f"ARGUMENT QUALITY ({self.rubric}) on 0-100. Respond with the "
                "number on the first line, then one sentence of justification."},
            {"role": "user", "content":
                f"Question: {question}\nProposal by {proposal.proposer}:\n{proposal.text}"},
        ]
        raw = self.llm(messages)
        score = _parse_score(raw)
        return score, raw.strip()

    def self_confidence(self, question: str, proposal: Proposal) -> float:
        messages = [
            {"role": "system", "content":
                "Rate 0-100 how confident you are that your proposal is the "
                "best available answer. Reply with ONLY the number."},
            {"role": "user", "content":
                f"Question: {question}\nYour proposal:\n{proposal.text}"},
        ]
        return _parse_score(self.llm(messages))


def _parse_score(raw: str) -> float:
    import re
    m = re.search(r"\b(\d{1,3})\b", raw)
    return min(1.0, int(m.group(1)) / 100.0) if m else 0.5
