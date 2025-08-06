"""
Manages window detection, focus, and dimensions for Balatro.
Implements Singleton pattern to avoid multiple conflicting window handles.
"""


class WindowManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WindowManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.hwnd = None

    def findWindow(self):
        pass

    def bringToFront(self):
        pass
