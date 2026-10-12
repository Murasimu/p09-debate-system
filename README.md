# Multi-Agent Debate System

**Project 9 / 12 — Agentic AI portfolio.**
*Shows: you can orchestrate swarms, not single agents.*

Stance-diverse proposer agents argue competing solutions, an impartial
critic scores argument quality, weighted voting decides (or honestly
reports no majority), and an aggregator synthesizes the final answer
with a confidence score.

## Why this matters

Single LLM calls average toward bland consensus; asking one model
"what's the best approach?" gets you its default bias. Debate extracts
more signal: forced stances surface trade-offs, a critic separates
*well-argued* from *confidently stated*, and voting/consensus rules make
disagreement measurable instead of averaged away.

1. **Stance diversity** — Cost-Minimizer, Risk-Averse, Speed-First,
   Quality-Maximizer personas produce genuinely different proposals
   (not three echoes of one prior).
2. **Critic scoring** — an impartial judge scores argument quality
   (reasoning, evidence, feasibility, honesty about trade-offs); scores
   weight the votes.
3. **Weighted consensus with quorum** — vote share blends LLM votes
   (50%) with quality weights (50%, critique^0.7 * confidence^0.3). A
   split house reports `no_majority` instead of crowning a weak winner.
4. **Aggregation with confidence** — a decided winner is presented with
   the concessions it survived; no-majority debates fuse complementary
   strengths. Confidence derives from consensus score (penalized when
   split).

## Architecture

```
question ──> Proposer(Cost-Minimizer) ─┐
            Proposer(Risk-Averse)    ──┼─> proposals
            Proposer(Speed-First)    ─┘        │
                                      Critic.score() per proposal
                                               │
            round 1: proposals + rebuttals ────┤  (later rounds refine)
                                               ▼
                              llm_votes(voter) + tally()
                                               │
                              decided ─────────┼──────── no_majority
                                │                                 │
                     Aggregator: winner                Aggregator: fuse strengths
                     + survived concessions            (transcript concessions)
                                │                                 │
                        FinalAnswer(text, winner, confidence, status)
```

- `debate_system/agents.py` — Persona, Proposer (propose/rebut), Critic
  (score + self-confidence, clamped parsing)
- `debate_system/debate.py` — rounds: propose → critique → rebut; full
  transcript retained
- `debate_system/consensus.py` — weighted tally, quorum, LLM voter with
  fuzzy fallback
- `debate_system/aggregator.py` — winner synthesis vs fusion synthesis

## Quickstart

```bash
uv pip install -e ".[demo]"
python demo.py            # scripted debate, no network
python demo.py --live     # full live debate on a real LLM
```

Demo question: *"Should our 4-person startup migrate from managed
Postgres to self-hosting on a VPS to save $80/month?"* — three stances
argue, the critic scores, the voter picks, and the final answer
integrates the winning argument plus the conceded points.

## Tests

```bash
pytest -q
```

18 tests cover: proposal shape and stance behavior, rebuttals receiving
other proposals, critic score parsing (normal, clamped overflow,
missing-number default), self-confidence, two-round structure (final
round without rebuttals, critique application), weighted majority
decided, split-house no-majority, high-critique-vs-votes balance bounds,
empty proposals, default scores, LLM voter parsing and fuzzy fallback,
winner synthesis, no-majority synthesis with confidence penalty, and a
full pipeline end-to-end.

## Design decisions

- **Stances over samplers** — diversity from persona instructions is
  cheaper and more controllable than diversity from temperature alone.
- **Critic scores argument quality, not the answer** — the question
  "which answer do you like?" reuses the same biases the debate exists
  to expose; "which argument is well-made?" is orthogonal and scoreable.
- **Quality-weighted votes** — pure vote counting rewards popularity;
  pure scoring ignores peer judgment; the 50/50 blend with a quorum
  keeps both honest and reports splits instead of hiding them.
- **Concessions are first-class** — the transcript of rebuttals is what
  lets the aggregator produce "winner + conceded improvements" instead
  of just restating the winner.
- **No-majority is a feature** — forcing a winner from a split house
  fabricates certainty; reporting the split and fusing strengths is the
  honest output.

## Portfolio context

Project 9 of a 12-project agentic-AI portfolio (structured output, RAG
grounding, ReAct planning, tool orchestration, memory, HITL approval,
cost routing, event automation, debate, self-eval, observability, OSS
contribution). See the profile README for the full map.
