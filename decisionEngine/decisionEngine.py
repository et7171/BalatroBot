"""
Strategy pattern interface for selecting the best move.
Routes between rule-based, LLM-based, or learned AI policies.
"""


class DecisionEngine:
    def __init__(self, strategy):
        self.strategy = strategy

    def chooseAction(self, state):
        return self.strategy.chooseAction(state)
