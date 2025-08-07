"""
Main controller for running the Balatro AI bot.
Coordinates game launch, state capture, decision-making, and action execution.
"""

# flake8: noqa

from gameLauncher.gameLauncher import GameLauncher

# from windowManager.windowManager import WindowManager

from screenCapture.screenCapturer import ScreenCapturer
import cv2


if __name__ == "__main__":

    # Game launcher
    # launcher = GameLauncher(delay=5)
    # launcher.launchGame()

    # Balatro is autimatically put into focus
    """
    Balatro autimatically is launched and is put in focus

    windowManager = WindowManager(windowTitle="Balatro")
    windowManager.focusWindow()
    """

    # Screenshot test
    """
    capturer = (
        ScreenCapturer()
    )  # or pass a region if you want just a piece of the screen
    frame = capturer.capture()
    cv2.imshow("Game Screenshot", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
