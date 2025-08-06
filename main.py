"""
Main controller for running the Balatro AI bot.
Coordinates game launch, state capture, decision-making, and action execution.
"""

from gameLauncher.gameLauncher import GameLauncher

if __name__ == "__main__":
    launcher = GameLauncher(delay=5)
    launcher.launchGame()
