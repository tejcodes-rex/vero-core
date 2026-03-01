import tkinter as tk
import time

class AlertOverlay:
    """
    Transparent, topmost, fullscreen alert with a flashing red border.
    Triggered on deepfake identification.
    """
    def __init__(self):
        self.root = None
        self.flashing = False

    def show(self, duration=3.0, insight=None):
        if self.root: return # Already showing
        
        self.root = tk.Toplevel()
        self.root.overrideredirect(True) # No title bar
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0) # Start transparent
        
        # Fullscreen
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_w}x{screen_h}+0+0")
        
        # Transparent background (Windows specific)
        self.root.attributes("-transparentcolor", "white")
        self.canvas = tk.Canvas(self.root, width=screen_w, height=screen_h, bg="white", highlightthickness=0)
        self.canvas.pack()
        
        # Red Border
        bw = 40
        self.border = self.canvas.create_rectangle(
            bw/2, bw/2, screen_w-bw/2, screen_h-bw/2,
            outline="red", width=bw
        )
        
        # Warning Text
        header_text = "⚠ VERO-CORE THREAT DETECTED ⚠\nAI CLONE IDENTIFIED"
        body_text = ""
        offset_y = 0
        if insight:
            body_text = f"\n\n🎓 Teach-Back Insight:\n{insight}"
            offset_y = 60
            
        self.canvas.create_text(
            screen_w/2, screen_h/2 - offset_y,
            text=header_text,
            fill="red", font=("Outfit", 48, "bold"), justify="center"
        )
        
        if insight:
            self.canvas.create_text(
                screen_w/2, screen_h/2 + 80,
                text=body_text,
                fill="#F1C40F", font=("Outfit", 20, "bold"), justify="center"
            )

        # Make window click-through (Windows)
        try:
            import win32gui, win32con, win32api
            hwnd = win32gui.GetParent(self.root.winfo_id())
            # GWL_EXSTYLE = -20, WS_EX_LAYERED = 0x80000, WS_EX_TRANSPARENT = 0x20
            style = win32gui.GetWindowLong(hwnd, -20)
            win32gui.SetWindowLong(hwnd, -20, style | 0x80000 | 0x20)
        except Exception as e:
            logger.warning(f"Windows click-through optimization failed: {e}")

        self.flashing = True
        self._flash(0.1)
        self.root.after(int(duration * 1000), self.dismiss)

    def _flash(self, alpha):
        if not self.flashing: return
        new_alpha = 0.8 if alpha < 0.2 else 0.1
        self.root.attributes("-alpha", alpha)
        self.root.after(400, lambda: self._flash(new_alpha))

    def dismiss(self):
        self.flashing = False
        if self.root:
            self.root.destroy()
            self.root = None

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    alert = AlertOverlay()
    alert.show()
    root.mainloop()
