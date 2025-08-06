"""
Singleton and observer that holds the current game state.
Allows other systems to listen for updates.
"""


class StateManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.state = {}

    def updateState(self, newState):
        self.state = newState
        # Notify observers here (optional)
