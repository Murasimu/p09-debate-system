"""Demo: a product decision debated by stance-diverse agents.

Question: "Should our 4-person startup migrate from a managed Postgres
to self-hosted on a VPS to save money?"

Three proposers (Cost-Minimizer, Risk-Averse, Speed-First) argue, a
critic scores the arguments, weighted voting decides, and the aggregator
synthesizes the final recommendation with confidence.

Usage:
  python demo.py            # scripted, no network
  python demo.py --live     # full live debate with a real LLM
"""

from __future__ import annotations

import sys

from debate_system import (
    Aggregator, Debate, DebateConfig, Persona, Proposer, tally,
)
from debate_system.consensus import llm_votes

QUESTION = ("Should our 4-person startup migrate from managed Postgres to "
            "self-hosting on a VPS to save $80/month?")


def scripted_llm(replies):
    stack = list(replies)
    def llm(messages):
        return stack.pop(0)
    return llm


def main():
    print("=" * 62)
    print(f"QUESTION: {QUESTION}")
    print("=" * 62)

    personas = [
        Persona("Cost-Minimizer", "You prioritize minimizing cost."),
        Persona("Risk-Averse", "You prioritize reliability and data safety."),
        Persona("Speed-First", "You prioritize team velocity."),
    ]

    if "--live" in sys.argv:
        from debate_system.llm import make_llm
        try:
            base = make_llm(temperature=0.6)
        except RuntimeError as e:
            print(f"skipping live demo: {e}")
            return
        import functools
        llms = [functools.partial(lambda s, m: s(m), make_llm(temperature=0.7))
                for _ in personas]
        proposers = [Proposer(p, l) for p, l in zip(personas, llms)]
        critic_llm = make_llm(temperature=0.2)
        voter = make_llm(temperature=0.2)
        aggregator_llm = make_llm(temperature=0.3)
    else:
        proposers = [
            Proposer(personas[0], scripted_llm([
                "Migrate now: $80/month is ~$960/year, nearly a free engineer-week. "
                "Use the standard Postgres docker image, daily pg_dump to object storage."])),
            Proposer(personas[1], scripted_llm([
                "Keep managed. Backups, failover, monitoring and patching are included; "
                "one data-loss incident costs more than years of savings. Revisit at Series A."])),
            Proposer(personas[2], scripted_llm([
                "Hybrid: keep managed for prod, spin a cheap VPS for staging/dev only. "
                "Saves ~$30/month with zero prod risk and takes an afternoon."])),
        ]
        critic_llm = scripted_llm([
            "35\nIgnores operational burden and recovery time objectives.",
            "45\nRight on safety but dismisses the cost pressure entirely.",
            "85\nConcrete, low-risk, actionable today with honest trade-offs.",
            "60\nModest scope limits the impact.",
            "50\nReasonable but static.",
            "75\nAligns with team velocity."])
        voter = scripted_llm(["Speed-First"])
        aggregator_llm = scripted_llm([
            "Keep managed Postgres for production (Risk-Averse's core point), move "
            "staging/dev to a cheap VPS (Speed-First's hybrid), and revisit a full "
            "migration only when the bill exceeds $300/month (Cost-Minimizer's trigger)."])

    debate = Debate(QUESTION, proposers, critic_llm, DebateConfig(rounds=1))
    rounds = debate.run()

    for r in rounds:
        print(f"\n--- Round {r.number} ---")
        for p in r.proposals:
            print(f"\n[{p.proposer}] (critique {p.critique_score}, confidence {p.confidence})")
            print(f"  {p.text[:200]}")
            print(f"  critic: {p.critique[:100]}")
        for name, reb in r.rebuttals.items():
            print(f"\n[{name} rebuts]: {reb[:150]}")

    proposals = debate.final_proposals()
    votes = llm_votes(voter, QUESTION, proposals)
    consensus = tally(proposals, votes)
    print(f"\n--- CONSENSUS ---")
    print(f"votes={votes} status={consensus.status} winner={consensus.winner} "
          f"top_score={consensus.top_score}")

    fa = Aggregator(aggregator_llm).synthesize(
        QUESTION, proposals, consensus, debate._context_so_far())
    print(f"\n--- FINAL ANSWER (confidence {fa.confidence}) ---")
    print(fa.text)


if __name__ == "__main__":
    main()
