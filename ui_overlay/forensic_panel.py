import customtkinter as ctk

class ForensicPanel(ctk.CTkScrollableFrame):
    """
    Live metrics and evidence strings for the dashboard.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.entries = []
        self.last_update = 0
        self.update_interval = 1.0 / 30.0 # 30Hz throttle

    def clear_evidence(self):
        """Removes all evidence frames from the panel."""
        for entry in self.entries:
            entry.destroy()
        self.entries = []

    def update_evidence(self, evidence_list: list):
        # UI Message Throttler (30Hz)
        import time
        now = time.time()
        if now - self.last_update < self.update_interval:
            return
        self.last_update = now

        # Clear old
        for entry in self.entries:
            entry.destroy()
        self.entries = []

        for i, item in enumerate(evidence_list):
            color = "#00D2FF" # Safe Neon Cyan
            icon = "🛡️"
            
            if item["severity"] == "WARNING":
                color = "#F1C40F" # Yellow
                icon = "⚠️"
            elif item["severity"] == "CRITICAL":
                color = "#FF2A2A" # Neon Red
                icon = "🚨"
            elif item["severity"] == "INFO":
                color = "#3498DB"
                icon = "ℹ️"

            bg_color = "#12121D" if i % 2 == 0 else "transparent"
            frame = ctk.CTkFrame(self, fg_color=bg_color, corner_radius=8)
            frame.pack(fill="x", pady=4, padx=5)
            
            dot = ctk.CTkLabel(frame, text=icon, text_color=color, font=("Arial", 16))
            dot.pack(side="left", padx=10, pady=8)
            
            source_lbl = ctk.CTkLabel(
                frame, text=f"{item['source']}",
                font=("Outfit", 12, "bold"), text_color=color, width=90, anchor="w"
            )
            source_lbl.pack(side="left", padx=(0, 10))
            
            lbl = ctk.CTkLabel(
                frame, text=f"{item['message']}",
                font=("Outfit", 12), text_color="#E0E0E0", wraplength=400, justify="left", anchor="w"
            )
            lbl.pack(side="left", padx=5, fill="x", expand=True, pady=8)
            
            self.entries.append(frame)
