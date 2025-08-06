"""
Launches the Balatro game process (via Steam or standalone).
Uses a factory approach to support multiple launch methods.
"""

import subprocess
import time
from pathlib import Path


class GameLauncher:
    """
    Launches Balatro directly using its executable.
    """

    def __init__(self, delay=10):
        self.delay = delay
        self.exePath = (
            r"D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe"  # noqa: E501
        )

    def launchGame(self):
        """
        Launches Balatro from its executable and waits for it to load.
        """
        print("[LAUNCH] Launching Balatro...")

        if not Path(self.exePath).exists():
            raise FileNotFoundError(
                "Balatro.exe not found at: " + self.exePath
            )  # noqa: E501

        subprocess.Popen(self.exePath)

        print(
            f"[LAUNCH] Waiting {self.delay} seconds for the game to start..."
        )  # noqa: E501
        time.sleep(self.delay)
        print("[LAUNCH] Launch complete.")
