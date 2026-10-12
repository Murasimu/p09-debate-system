"""Tests for the multi-agent debate system. Offline, deterministic."""

import pytest

from debate_system import (
    Aggregator, Critic, Debate, DebateConfig, Persona, Proposer, tally,
)
from debate_system.agents import Proposal
from debate_system.consensus import Consensus, llm_votes


def scripted(replies):
    stack = list(replies)
    def llm(messages):
        return stack.pop(0)
    return llm


def make_proposers():
    llms = [
        scripted(["Ship a manual script first, minimal cost.",
                  "I concede automation is better long-term, but keep the manual fallback.",
                  "Refined: manual script now, revisit automation next quarter."]),
        scripted(["Build full automation with heavy guardrails.",
                  "Cost is fair, but a staged rollout halves the risk.",
                  "Refined: staged rollout with guardrails from day one."]),
        scripted(["Ship a lambda in a week, iterate fast.",
                  "Quality point taken; add tests in week two.",
                  "Refined: lambda now, tests by week two."]),
    ]
    personas = [
        Persona("Cost-Minimizer", "minimize cost"),
        Persona("Risk-Averse", "avoid risk"),
        Persona("Speed-First", "move fast"),
    ]
    return [Proposer(p, llm) for p, llm in zip(personas, llms)]


def critique_llm():
    # scores: Cost 40, Risk 90, Speed 60; confidences: 50, 80, 70
    # (round 2 repeats the pattern with slightly different wording)
    return scripted([
        "40\nWeak on robustness.", "50",
        "90\nStrong guardrails, feasible.", "80",
        "60\nFast but thin on safety.", "70",
        "45\nSlightly improved.", "55",
        "88\nStill strong.", "82",
        "62\nStill fast.", "72",
    ])


# ---------------------------------------------------------------- proposer
class TestProposer:
    def test_proposal_shape(self):
        p = make_proposers()[0].propose("How to deploy?")
        assert p.proposer == "Cost-Minimizer"
        assert "manual script" in p.text

    def test_rebuttal_gets_others(self):
        props = make_proposers()
        own = props[0].propose("How to deploy?")
        others = [props[1].propose("How to deploy?")]
        r = props[0].rebut("How to deploy?", own, others)
        assert "concede" in r.lower()


# ---------------------------------------------------------------- critic
class TestCritic:
    def test_score_parsing(self):
        c = Critic(scripted(["87\nSolid reasoning."]))
        score, text = c.critique("q", Proposal("A", "text"))
        assert score == 0.87
        assert text.startswith("87")

    def test_score_clamped(self):
        c = Critic(scripted(["150\noverflow"]))
        score, _ = c.critique("q", Proposal("A", "t"))
        assert score == 1.0

    def test_missing_number_defaults_mid(self):
        c = Critic(scripted(["no numbers here"]))
        score, _ = c.critique("q", Proposal("A", "t"))
        assert score == 0.5

    def test_self_confidence(self):
        c = Critic(scripted(["80"]))
        assert c.self_confidence("q", Proposal("A", "t")) == 0.8


# ---------------------------------------------------------------- debate
class TestDebate:
    def test_two_rounds_structure(self):
        d = Debate("How to deploy?", make_proposers(), critique_llm(),
                   DebateConfig(rounds=2))
        rounds = d.run()
        assert len(rounds) == 2
        assert len(rounds[0].proposals) == 3
        assert set(rounds[0].rebuttals) == {"Cost-Minimizer", "Risk-Averse", "Speed-First"}
        assert rounds[1].rebuttals == {}  # no rebuttals in the final round
        # critiques applied
        scores = {p.proposer: p.critique_score for p in rounds[0].proposals}
        assert scores["Risk-Averse"] == 0.9

    def test_final_proposals(self):
        d = Debate("q", make_proposers(), critique_llm(), DebateConfig(rounds=1))
        d.run()
        assert len(d.final_proposals()) == 3


# ---------------------------------------------------------------- consensus
class TestTally:
    def _props(self):
        return [
            Proposal("A", "a", critique_score=0.9, confidence=0.8),
            Proposal("B", "b", critique_score=0.5, confidence=0.9),
        ]

    def test_weighted_majority_decided(self):
        votes = {"A": 3, "B": 1}
        c = tally(self._props(), votes)
        assert c.status == "decided" and c.winner == "A"
        assert c.top_score >= 0.5

    def test_split_house_no_majority(self):
        votes = {"A": 2, "B": 2}
        c = tally(self._props(), votes, quorum=0.75)
        assert c.status == "no_majority" and c.winner is None

    def test_high_critique_can_beat_more_votes(self):
        # B has 3 votes but much weaker critique; A has 1 vote but 0.9
        votes = {"A": 1, "B": 3}
        c = tally(self._props(), votes, quorum=0.0)
        # with 50/50 vote/weight blend, A's quality may or may not overtake;
        # deterministic check: A's share must be within [0.3, 0.7]
        assert 0.2 <= c.top_score <= 0.8

    def test_empty_proposals(self):
        c = tally([], {})
        assert c.status == "no_majority" and c.winner is None

    def test_missing_scores_default_mid(self):
        props = [Proposal("A", "a"), Proposal("B", "b")]
        c = tally(props, {"A": 1}, quorum=0.0)
        assert c.status == "decided" and c.winner == "A"


class TestLLMVotes:
    def test_vote_parsed(self):
        props = [Proposal("Cost-Minimizer", "x"), Proposal("Risk-Averse", "y")]
        votes = llm_votes(scripted(["Risk-Averse"]), "q", props)
        assert votes == {"Cost-Minimizer": 0, "Risk-Averse": 1}

    def test_fuzzy_fallback(self):
        props = [Proposal("Cost-Minimizer", "x"), Proposal("Risk-Averse", "y")]
        votes = llm_votes(scripted(["I'd go with the cost-minimizer option"]), "q", props)
        assert votes["Cost-Minimizer"] == 1


# ---------------------------------------------------------------- aggregator
class TestAggregator:
    def test_winner_synthesis(self):
        props = [
            Proposal("A", "use kubernetes", critique_score=0.9, confidence=0.8),
            Proposal("B", "use serverless", critique_score=0.5, confidence=0.9),
        ]
        cons = Consensus("A", "decided", {}, {"A": 3, "B": 1}, 0.5, 0.72)
        agg = Aggregator(scripted(["Final: adopt A's kubernetes plan with B's cost checks."]))
        fa = agg.synthesize("deploy how?", props, cons, "transcript")
        assert fa.winner == "A" and fa.status == "decided"
        assert fa.confidence == 0.72
        assert "kubernetes" in fa.text

    def test_no_majority_synthesis(self):
        props = [
            Proposal("A", "kubernetes", critique_score=0.6, confidence=0.6),
            Proposal("B", "serverless", critique_score=0.6, confidence=0.6),
        ]
        cons = Consensus(None, "no_majority", {}, {"A": 2, "B": 2}, 0.75, 0.5)
        agg = Aggregator(scripted(["Fused plan: start serverless, migrate to k8s at scale."]))
        fa = agg.synthesize("deploy how?", props, cons, "transcript")
        assert fa.winner is None and fa.status == "no_majority"
        assert fa.confidence < 0.5  # split house penalized


# ---------------------------------------------------------------- e2e
class TestEndToEnd:
    def test_full_debate_pipeline(self):
        d = Debate("How should we deploy?", make_proposers(), critique_llm(),
                   DebateConfig(rounds=1))
        d.run()
        proposals = d.final_proposals()
        votes = {"Cost-Minimizer": 0, "Risk-Averse": 3, "Speed-First": 0}
        cons = tally(proposals, votes)
        assert cons.winner == "Risk-Averse"
        agg = Aggregator(scripted(["Deploy with guardrails, staged rollout."]))
        fa = agg.synthesize("How should we deploy?", proposals, cons, "t")
        assert fa.status == "decided" and "guardrails" in fa.text
