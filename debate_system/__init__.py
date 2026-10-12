"""Multi-agent debate system.

Multiple proposer agents argue different solutions, a critic evaluates
the arguments, weighted voting / consensus decides, and an aggregator
synthesizes the final answer with a confidence score.

Modules:
- agents.py    proposer + critic personas (LLM callables with stances)
- debate.py    rounds of argument -> rebuttal -> critique
- consensus.py voting, weight by critique score, quorum rules
- aggregator.py final synthesis with confidence
"""

from .agents import Critic, Persona, Proposer
from .debate import Debate, Round, DebateConfig
from .consensus import Consensus, tally
from .aggregator import Aggregator, FinalAnswer

__all__ = [
    "Critic", "Persona", "Proposer",
    "Debate", "Round", "DebateConfig",
    "Consensus", "tally",
    "Aggregator", "FinalAnswer",
]
__version__ = "1.0.0"
