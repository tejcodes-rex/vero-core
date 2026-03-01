import customtkinter as ctk
import tkinter as tk
import math

class TrustRing(ctk.CTkFrame):
    """
    Animated circular gauge showing trust score 0-100%.
    Smooth color transitions (Green -> Yellow -> Red).
    """
    def __init__(self, master, size=300, **kwargs):
        super().__init__(master, **kwargs)
        self.size = size
        self.score = 1.0
        self.target_score = 1.0
        
        self.canvas = tk.Canvas(
            self, width=size, height=size, 
            bg="#0F0F1A", highlightthickness=0
        )
        self.canvas.pack(pady=20)
        
        self.score_label = ctk.CTkLabel(
            self, text="100%", font=("Outfit", 42, "bold")
        )
        self.score_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self._draw_background()
        self.update_animation()

    def _draw_background(self):
        # Outer faint ring
        margin_outer = 10
        self.canvas.create_oval(
            margin_outer, margin_outer, self.size-margin_outer, self.size-margin_outer,
            outline="#1A1A24", width=2
        )
        
        # Inner dashed radar ring
        margin_inner = 50
        self.canvas.create_oval(
            margin_inner, margin_inner, self.size-margin_inner, self.size-margin_inner,
            outline="#2A2A35", width=3, dash=(4, 8)
        )

        # Main track background
        margin = 25
        self.canvas.create_oval(
            margin, margin, self.size-margin, self.size-margin,
            outline="#181825", width=18
        )

    def set_score(self, score: float):
        self.target_score = float(score)
        # print(f"DEBUG: TrustRing.set_score received {score}")

    def update_animation(self):
        # Easing/Lerp
        if math.isnan(self.target_score):
            self.target_score = self.score
            
        # Rapid Recovery, Smooth Decline
        if self.target_score > self.score: 
            self.score += (self.target_score - self.score) * 0.40 # Snap back to Green
        else:
            self.score += (self.target_score - self.score) * 0.15 # Smooth drop
        
        self.canvas.delete("gauge")
        
        margin = 25
        extent = -(self.score * 359.9)
        
        # Determine color
        if self.score > 0.7:
            color = "#00D2FF" # Neon Cyan (Safe)
        elif self.score > 0.45:
            color = "#F1C40F" # Yellow (Warning)
        else:
            color = "#FF2A2A" # Neon Red (Danger)
            
        self.canvas.create_arc(
            margin, margin, self.size-margin, self.size-margin,
            start=90, extent=extent, outline=color, width=18, 
            style="arc", tags="gauge"
        )
        
        self.score_label.configure(
            text=f"{int(self.score * 100)}%",
            text_color=color
        )
        self.canvas.update_idletasks() # Force UI refresh
        self.after(50, self.update_animation)
