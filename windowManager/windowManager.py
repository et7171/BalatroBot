"""
Manages window detection, focus, and dimensions for Balatro.
"""


from pywinauto import Desktop
import psutil
import time

# flake8: noqa
class WindowManager:
    def __init__(self, windowTitle="Balatro", processName="Balatro.exe"):
        self.windowTitle = windowTitle
        self.processName = processName

    def focusWindow(self):
        print(
            f"[WINDOW] Scanning for '{self.windowTitle}' (process: {self.processName})..."
        )

        desktop = Desktop(backend="uia")
        candidates = []

        # Prefer exact process match
        for w in desktop.windows():
            try:
                title = w.window_text() or ""
                pid = w.process_id()
                pname = psutil.Process(pid).name() if pid else ""
            except Exception:
                continue

            if self.windowTitle in title and pname.lower() == self.processName.lower():
                candidates.append(w)

        # Fallback: title-only match, excluding common false positives
        if not candidates:
            for w in desktop.windows():
                try:
                    title = w.window_text() or ""
                except Exception:
                    continue
                if self.windowTitle in title and not any(
                    x in title for x in ["File Explorer", "Visual Studio", "Code"]
                ):
                    candidates.append(w)

        if not candidates:
            raise Exception("No matching Balatro window found.")

        target = candidates[0]  # already a UIAWrapper; no wrapper_object() needed

        try:
            if target.is_minimized():
                target.restore()
                time.sleep(0.2)
            target.set_focus()
            time.sleep(0.2)
            target.set_focus()  # second nudge improves reliability
        except Exception as e:
            raise Exception(f"Unable to focus target window: {e}")

        print(f"[WINDOW] Focused '{target.window_text()}' (pid={target.process_id()})")

