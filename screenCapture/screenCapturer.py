"""
Captures frames from the Balatro game window.
Acts as an adapter for MSS, OpenCV, or RAM reading tools.
"""


class ScreenCapturer:
    def __init__(self, windowManager):
        self.window = windowManager

    def getFrame(self):
        pass  # Capture screen region and return image
