import tkinter as tk
import logging

logger = logging.getLogger("RegionSelector")

class RegionSelector(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.callback = callback
        
        # Make the window fullscreen and completely transparent
        self.attributes("-fullscreen", True)
        self.attributes("-alpha", 0.3)
        self.attributes("-topmost", True)
        self.configure(bg='black')
        self.overrideredirect(True)
        
        # Instructions Label
        self.label = tk.Label(self, text="Click and drag to select the target face area. Press ESC to cancel.", 
                              fg="white", bg="black", font=("Segoe UI", 24, "bold"))
        self.label.pack(pady=50)

        # Canvas for drawing the selection rectangle
        self.canvas = tk.Canvas(self, cursor="cross", bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # State variables
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.bind("<Escape>", self.cancel)
        
    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        # Create an initial 0x0 rectangle where the mouse clicked
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, 
            outline='red', width=5, dash=(4, 4)
        )

    def on_mouse_drag(self, event):
        if self.rect_id:
            # Update the rectangle's coordinates as the mouse moves
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        if self.rect_id:
            end_x = event.x
            end_y = event.y
            
            # Ensure top-left and bottom-right ordering
            x1 = min(self.start_x, end_x)
            y1 = min(self.start_y, end_y)
            x2 = max(self.start_x, end_x)
            y2 = max(self.start_y, end_y)
            
            width = x2 - x1
            height = y2 - y1
            
            # Enforce a minimum size so accidental clicks don't crash the scanner
            if width > 50 and height > 50:
                logger.info(f"User selected ROI: x={x1}, y={y1}, w={width}, h={height}")
                self.destroy()
                self.callback({"left": x1, "top": y1, "width": width, "height": height})
            else:
                logger.warning("Selected ROI was too small. Resetting to full screen.")
                self.destroy()
                self.callback(None) # None means use full screen
        else:
            self.destroy()
            self.callback(None)

    def cancel(self, event=None):
        logger.info("User cancelled ROI selection.")
        self.destroy()
        self.callback(None)
