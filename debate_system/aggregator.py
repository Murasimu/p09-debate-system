"""Aggregator: synthesize the final answer from the debate.

With a decided winner: present it, noting the strongest counterpoints it
survived. With no majority: fuse the proposals' complementary strengths
(the critique text tells the aggregator which parts were conceded).
The final answer carries a confidence derived from consensus top score
and critic agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .agents import Proposal
from .consensus import Consensus

LLM = Callable[[list[dict[str, str]]], str]


@dataclass
class FinalAnswer:
    text: str
    winner: str | None
    confidence: float
    status: str


@dataclass
class Aggregator:
    llm: LLM

    def synthesize(self, question: str, proposals: list[Proposal],
                   consensus: Consensus, transcript: str) -> FinalAnswer:
        prop_map = {p.proposer: p for p in proposals}
        winner_prop = prop_map.get(consensus.winner or "", None)

        if consensus.status == "decided" and winner_prop is not None:
            prompt = (
                f"Question: {question}\n\n"
                f"The debate selected {winner_prop.proposer}'s proposal "
                f"(score {winner_prop.critique_score}).\n\n"
                f"Winning proposal:\n{winner_prop.text}\n\n"
                "Write the final answer: the winning solution, plus any "
                "valid concessions made during the debate that improve it. "
                "Max 120 words."
            )
            conf = consensus.top_score
        else:
            strengths = "\n".join(
                f"- {p.proposer} (score {p.critique_score}): {p.text[:180]}"
                for p in proposals)
            prompt = (
                f"Question: {question}\n\nNo proposal won a majority. "
                "Synthesize the best final answer by combining the "
                "complementary strengths of these proposals:\n"
                f"{strengths}\n\nDebate transcript (for concessions):\n"
                f"{transcript[-1500:]}\nMax 120 words."
            )
            conf = consensus.top_score * 0.8  # split house -> less confident

        text = self.llm([
            {"role": "system", "content":
                "You are the debate's closing synthesizer. Be concrete and "
                "balanced; credit where ideas came from."},
            {"role": "user", "content": prompt},
        ]).strip()

        return FinalAnswer(
            text=text,
            winner=consensus.winner,
            confidence=round(conf, 3),
            status=consensus.status,
        )
