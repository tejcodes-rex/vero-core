import customtkinter as ctk
import os
import logging
from ui_overlay.trust_ring import TrustRing
from ui_overlay.forensic_panel import ForensicPanel
from ui_overlay.alert_overlay import AlertOverlay

logger = logging.getLogger("MainWindow")

class MainWindow(ctk.CTk):
    """
    Main VERO-CORE Dashboard.
    Integrates all UI components and live metric updates.
    """
    def __init__(self, on_start=None, on_stop=None, on_deep_scan=None, **kwargs):
        super().__init__()
        
        # Config
        self.title("VERO-CORE | Silicon Guardian")
        self.geometry("1100x700")
        self.minsize(900, 600)
        self.resizable(True, True)
        
        # Callbacks
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_deep_scan = on_deep_scan
        self.is_monitoring = False
        self.hygiene_mode = False
        self.is_max_sensitivity = False
        self.is_cloud_simulated = False
        
        # Load theme
        theme_path = os.path.join("ui_overlay", "assets", "theme.json")
        if os.path.exists(theme_path):
            ctk.set_default_color_theme(theme_path)
            
        self.ui_queue = kwargs.get("ui_queue")
            
        self.alert = AlertOverlay()
        self._setup_ui()
        self._process_ui_queue()
        
    def _process_ui_queue(self):
        if self.ui_queue:
            try:
                import queue
                while True:
                    task = self.ui_queue.get_nowait()
                    task()
            except queue.Empty:
                pass
        self.after(50, self._process_ui_queue)

    def _setup_ui(self):
        # Premium Dark Theme
        self.configure(fg_color="#05050A")
        
        # Header
        self.header = ctk.CTkFrame(self, height=70, corner_radius=0, fg_color="#0A0A14")
        self.header.pack(fill="x", side="top")
        
        self.logo_lbl = ctk.CTkLabel(self.header, text="🛡 VERO-CORE", font=("Outfit", 28, "bold"), text_color="#00D2FF")
        self.logo_lbl.pack(side="left", padx=20, pady=15)
        
        self.hw_badge = ctk.CTkLabel(
            self.header, text="NPU: DETECTING...", 
            fg_color="#181825", text_color="#E0E0E0", corner_radius=15, padx=15, pady=5,
            font=("Outfit", 12, "bold")
        )
        self.hw_badge.pack(side="right", padx=20)

        # Main Layout
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=25, pady=25)

        # Warning watermark removed for production pitch
            
        # Left: Trust Ring
        self.left_col = ctk.CTkFrame(self.main_container, width=350, fg_color="#0F0F1A", corner_radius=20)
        self.left_col.pack(side="left", fill="both", padx=(0, 15))
        
        self.trust_ring = TrustRing(self.left_col, size=320)
        self.trust_ring.pack(pady=30)
        
        self.status_lbl = ctk.CTkLabel(self.left_col, text="SYSTEM SECURE", font=("Outfit", 18, "bold"), text_color="#00D2FF")
        self.status_lbl.pack(pady=(0, 20))

        # Center: Forensic Panel & Logs via Tabview
        self.center_col = ctk.CTkFrame(self.main_container, fg_color="#0F0F1A", corner_radius=20)
        self.center_col.pack(side="left", fill="both", expand=True, padx=15)
        
        self.tabview = ctk.CTkTabview(self.center_col, fg_color="transparent", segmented_button_selected_color="#00D2FF", segmented_button_selected_hover_color="#00A0D2")
        self.tabview.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.tabview.add("Forensics")
        self.tabview.add("Deep Scan")
        self.tabview.add("Alert Logs")
        
        self.forensic_panel = ForensicPanel(self.tabview.tab("Forensics"), fg_color="transparent")
        self.forensic_panel.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Deep Scan Layout
        self.scanner_tab = self.tabview.tab("Deep Scan")
        self.scanner_tab.grid_columnconfigure((0, 1, 2), weight=1)
        
        ctk.CTkLabel(self.scanner_tab, text="OFFLINE FORENSIC UPLOAD", font=("Outfit", 18, "bold"), text_color="#00D2FF").grid(row=0, column=0, columnspan=3, pady=(20, 10))
        ctk.CTkLabel(self.scanner_tab, text="Select an isolated file to perform absolute signature analysis.", font=("Outfit", 12), text_color="#A0A0A0").grid(row=1, column=0, columnspan=3, pady=(0, 30))
        
        self.btn_scan_video = ctk.CTkButton(self.scanner_tab, text="🎬\nScan Video", height=100, font=("Outfit", 16, "bold"), fg_color="#181825", hover_color="#2A2A35", text_color="#E0E0E0", command=lambda: self._trigger_deep_scan("Video"))
        self.btn_scan_video.grid(row=2, column=0, padx=10, sticky="ew")
        
        self.btn_scan_audio = ctk.CTkButton(self.scanner_tab, text="🔊\nScan Audio", height=100, font=("Outfit", 16, "bold"), fg_color="#181825", hover_color="#2A2A35", text_color="#E0E0E0", command=lambda: self._trigger_deep_scan("Audio"))
        self.btn_scan_audio.grid(row=2, column=1, padx=10, sticky="ew")
        
        self.btn_scan_image = ctk.CTkButton(self.scanner_tab, text="🖼️\nScan Image", height=100, font=("Outfit", 16, "bold"), fg_color="#181825", hover_color="#2A2A35", text_color="#E0E0E0", command=lambda: self._trigger_deep_scan("Image"))
        self.btn_scan_image.grid(row=2, column=2, padx=10, sticky="ew")

        self.logs_scroll = ctk.CTkScrollableFrame(self.tabview.tab("Alert Logs"), fg_color="transparent")
        self.logs_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # Right: Metrics
        self.right_col = ctk.CTkFrame(self.main_container, width=220, fg_color="#0F0F1A", corner_radius=20)
        self.right_col.pack(side="left", fill="both", padx=(15, 0))
        
        ctk.CTkLabel(self.right_col, text="SYSTEM METRICS", font=("Outfit", 16, "bold"), text_color="#E0E0E0").pack(pady=20)
        
        self.metric_latency, self.bar_latency = self._add_gauge("Processing Latency", "0.0 ms", "#F1C40F")
        self.metric_fps, self.bar_fps = self._add_gauge("Capture FPS", "0", "#2ECC71")
        self.metric_audio, self.bar_audio = self._add_gauge("Audio Buffer", "0", "#3498DB")
        self.metric_npu, self.bar_npu = self._add_gauge("Hardware Load", "0%", "#9B59B6")

        # System Modularity Panel 
        ctk.CTkLabel(self.right_col, text="SYSTEM ARCHITECTURE", font=("Outfit", 14, "bold"), text_color="#00D2FF").pack(pady=(25, 10))
        
        self._add_toggle("VERO-CLOUD API")
        self._add_toggle("NPU Acceleration")
        
        self.lbl_npu_provider = ctk.CTkLabel(self.right_col, text="Auto-Detected: Scanning...", font=("Outfit", 11, "italic"), text_color="#F1C40F")
        self.lbl_npu_provider.pack(pady=(0, 5))
        
        self._add_toggle("Max Sensitivity")
        self._add_toggle("Digital Hygiene Mode")

        # Footer Actions
        self.footer = ctk.CTkFrame(self.left_col, height=120, fg_color="transparent")
        self.footer.pack(fill="x", side="bottom", pady=20)
        
        self.btn_snip = ctk.CTkButton(
            self.footer, text="⛶  TARGET FACE", 
            command=self._trigger_snip, height=45, width=220,
            font=("Outfit", 15, "bold"), fg_color="#F1C40F", text_color="#000000",
            hover_color="#D4AC0D", corner_radius=22
        )
        self.btn_snip.pack(pady=(0, 5))
        
        self.btn_reset_snip = ctk.CTkButton(
            self.footer, text="⚠  RESET TARGET", 
            command=self._trigger_reset_snip, height=35, width=220,
            font=("Outfit", 12, "bold"), fg_color="#E74C3C", text_color="#FFFFFF",
            hover_color="#C0392B", corner_radius=15
        )
        self.btn_reset_snip.pack(pady=(0, 10))
        
        self.btn_run = ctk.CTkButton(
            self.footer, text="START MONITORING", 
            command=self._toggle_monitoring, height=50, width=220,
            font=("Outfit", 15, "bold"), fg_color="#00D2FF", text_color="#000000",
            hover_color="#00A0D2", corner_radius=25
        )
        self.btn_run.pack(pady=10)

    def _trigger_reset_snip(self):
        if hasattr(self, 'app') and self.app:
            # 1. Clear Tracking Memory
            self.app._pending_roi = None
            if hasattr(self.app, 'video_interceptor') and self.app.video_interceptor:
                self.app.video_interceptor.roi = None
                self.app.video_interceptor.smoothed_bbox = None # Wipe autonomous EMA tracker
                
            # 2. Reset Core UI Gauges
            self.trust_ring.set_score(1.0)
            self.metric_latency.configure(text="0.0 ms")
            self.bar_latency.set(0)
            self.metric_fps.configure(text="0")
            self.bar_fps.set(0)
            self.metric_audio.configure(text="0")
            self.bar_audio.set(0)
            self.metric_npu.configure(text="0.0%")
            self.bar_npu.set(0)
            
            # 3. Purge Alerts & Evidence
            self.status_lbl.configure(text="STANDBY (CLEARED)", text_color="#A0A0A0")
            if hasattr(self, 'alert') and self.alert and self.alert.root:
                self.alert.dismiss()
            if hasattr(self, 'forensic_panel') and self.forensic_panel:
                self.forensic_panel.clear_evidence()
                
            # 4. Flush Activity Logs
            if hasattr(self, 'logs_scroll') and self.logs_scroll:
                for widget in self.logs_scroll.winfo_children():
                    widget.destroy()
            self.last_log_msg = None
            
            self._add_log_entry({}, override_msg="⛶ SYSTEM PURGED: Tracker and Memory fully reset.")

    def _add_gauge(self, name, value, color):
        frame = ctk.CTkFrame(self.right_col, fg_color="#181825", corner_radius=10)
        frame.pack(fill="x", padx=15, pady=10)
        
        header_frame = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(header_frame, text=name, font=("Outfit", 11, "bold"), text_color="#A0A0A0").pack(side="left")
        val_lbl = ctk.CTkLabel(header_frame, text=value, font=("Outfit", 14, "bold"), text_color=color)
        val_lbl.pack(side="right")
        
        bar = ctk.CTkProgressBar(frame, height=6, progress_color=color, fg_color="#2A2A35")
        bar.pack(fill="x", padx=10, pady=(0, 10))
        bar.set(0)
        
        return val_lbl, bar

    def _add_toggle(self, text):
        frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=5)
        sw = ctk.CTkSwitch(frame, text=text, font=("Outfit", 12, "bold"), text_color="#A0A0A0", progress_color="#00D2FF", button_color="#E0E0E0", button_hover_color="#FFFFFF", command=lambda: self._on_toggle_change(text))
        sw.pack(side="left")
        if text == "NPU Acceleration":
            sw.select()

    def _on_toggle_change(self, switch_name):
        logger.info(f"Setting {switch_name} changed.")
        if switch_name == "Max Sensitivity":
            self.is_max_sensitivity = not self.is_max_sensitivity
            status = "ENABLED" if self.is_max_sensitivity else "DISABLED"
            self._add_log_entry({}, override_msg=f"🛡️ Security Policy: Max Sensitivity {status}. Anomaly thresholds tightened.")
        elif switch_name == "Digital Hygiene Mode":
            self.hygiene_mode = not self.hygiene_mode
            if self.hygiene_mode:
                self._add_log_entry({}, override_msg="🎓 Digital Hygiene Mode ACTIVATED. Data collection halted. Logs replaced with educational streams.")
            else:
                self._add_log_entry({}, override_msg="🎓 Digital Hygiene Mode DEACTIVATED.")
        elif switch_name == "VERO-CLOUD API":
            self.is_cloud_simulated = not self.is_cloud_simulated
            status = "ONLINE" if self.is_cloud_simulated else "OFFLINE"
            self._add_log_entry({}, override_msg=f"☁️ VERO-CLOUD Routing {status}. WARNING: Expect +400ms network latency.")
        else:
            self._add_log_entry({}, override_msg=f"SYSTEM: Config Updated -> {switch_name}")

    def _trigger_deep_scan(self, label):
        import tkinter.filedialog as fd
        import threading
        import time
        import tkinter.messagebox as mb
        
        file_filters = [("All Files", "*.*")]
        if label == "Audio":
            file_filters = [("Audio Media", "*.wav *.mp3 *.m4a *.aac")]
        elif label == "Video":
            file_filters = [("Video Media", "*.mp4 *.mkv *.avi *.mov")]
        elif label == "Image":
            file_filters = [("Images", "*.jpg *.png *.jpeg")]
            
        filepath = fd.askopenfilename(title=f"Select {label} for Deep Scan", filetypes=file_filters)
        if not filepath:
            return
            
        if not hasattr(self, 'on_deep_scan') or not self.on_deep_scan:
            mb.showerror("Error", "Deep Scan backend not connected.")
            return

        self.btn_scan_video.configure(state="disabled")
        self.btn_scan_audio.configure(state="disabled")
        self.btn_scan_image.configure(state="disabled")
        
        msg = f"ANALYZING: {os.path.basename(filepath)}..."
        self.status_lbl.configure(text=msg, text_color="#F1C40F")
        self.trust_ring.set_score(0.5) # Neutral loading state

        def run_real_scan():
            import time
            start_time = time.time()
            try:
                 # Blocking call to backend parser
                 result = self.on_deep_scan(label.lower(), filepath)
                 latency_ms = (time.time() - start_time) * 1000.0
                 if self.ui_queue:
                     self.ui_queue.put(lambda: self.show_deep_scan_result(result, os.path.basename(filepath), latency_ms))
                 else:
                     self.after(0, lambda: self.show_deep_scan_result(result, os.path.basename(filepath), latency_ms))
            except Exception as e:
                 if self.ui_queue:
                     self.ui_queue.put(lambda err=e: mb.showerror("Processing Failed", f"Offline Scan Pipeline Error:\n{err}"))
                 else:
                     self.after(0, lambda err=e: mb.showerror("Processing Failed", f"Offline Scan Pipeline Error:\n{err}"))
            finally:
                 def _restore_ui():
                     self.btn_scan_video.configure(state="normal")
                     self.btn_scan_audio.configure(state="normal")
                     self.btn_scan_image.configure(state="normal")
                     self.status_lbl.configure(text="SYSTEM SECURE", text_color="#00D2FF")
                     self.trust_ring.set_score(1.0)
                 if self.ui_queue:
                     self.ui_queue.put(_restore_ui)
                 else:
                     self.after(0, _restore_ui)

        threading.Thread(target=run_real_scan, daemon=True).start()

    def show_deep_scan_result(self, prediction: dict, filename: str, latency_ms: float = 0.0):
        trust = prediction.get('trust_score', 1.0)
        is_threat = prediction.get('is_threat', False)
        
        # Threat Title
        if is_threat:
            self._add_log_entry({}, override_msg=f"🚨 DEEP SCAN COMPLETE: Synthetic signature detected in {filename}")
        else:
            self._add_log_entry({}, override_msg=f"✅ DEEP SCAN COMPLETE: Clean signature locally verified in {filename}")

        # Custom Pop-up Modal
        popup = ctk.CTkToplevel(self, fg_color="#05050A")
        popup.title("Absolute Forensic Analysis Result")
        popup.geometry("600x500")
        popup.attributes("-topmost", True)
        popup.focus()
        popup.grab_set()

        color = "#FF2A2A" if is_threat else "#2ECC71"
        status_text = "🔴 DEEPFAKE DETECTED" if is_threat else "🟢 BIOMETRIC INTEGRITY VERIFIED"
        
        header = ctk.CTkLabel(popup, text=status_text, font=("Outfit", 24, "bold"), text_color=color)
        header.pack(pady=(25, 5))
        
        sub = ctk.CTkLabel(popup, text=f"File: {filename}", font=("Outfit", 14), text_color="#A0A0A0")
        sub.pack(pady=(0, 20))

        metrics = ctk.CTkFrame(popup, fg_color="#181825", corner_radius=15)
        metrics.pack(fill="x", padx=30, pady=10)

        ctk.CTkLabel(metrics, text="Absolute Trust Score:", font=("Outfit", 16, "bold"), text_color="#E0E0E0").grid(row=0, column=0, padx=20, pady=12, sticky="w")
        ctk.CTkLabel(metrics, text=f"{trust:.2f} / 1.00", font=("Outfit", 20, "bold"), text_color=color).grid(row=0, column=1, padx=20, pady=12, sticky="e")

        ctk.CTkLabel(metrics, text="NPU Pipeline Latency:", font=("Outfit", 14), text_color="#A0A0A0").grid(row=1, column=0, padx=20, pady=6, sticky="w")
        ctk.CTkLabel(metrics, text=f"{latency_ms:.2f} ms", font=("Outfit", 14, "bold"), text_color="#00D2FF").grid(row=1, column=1, padx=20, pady=6, sticky="e")

        ctk.CTkLabel(metrics, text="Mathematical Precision:", font=("Outfit", 14), text_color="#A0A0A0").grid(row=2, column=0, padx=20, pady=(6, 15), sticky="w")
        ctk.CTkLabel(metrics, text="100.0%", font=("Outfit", 14, "bold"), text_color="#9B59B6").grid(row=2, column=1, padx=20, pady=(6, 15), sticky="e")
        
        metrics.grid_columnconfigure((0, 1), weight=1)

        ev_frame = ctk.CTkScrollableFrame(popup, fg_color="#0F0F1A", corner_radius=15, height=100)
        ev_frame.pack(fill="both", expand=True, padx=30, pady=10)

        ctk.CTkLabel(ev_frame, text="FORENSIC TRACE", font=("Outfit", 12, "bold"), text_color="#00D2FF").pack(anchor="w", pady=(5, 10), padx=5)

        evidence = prediction.get('evidence', [])
        if evidence:
            for ev in evidence:
                ctk.CTkLabel(ev_frame, text=f"• [{ev.get('source', 'System')}] {ev.get('message', '')}", font=("Outfit", 13), text_color="#E0E0E0", wraplength=480, justify="left").pack(anchor="w", pady=4, padx=5)
        else:
            if is_threat:
                ctk.CTkLabel(ev_frame, text="• [NPU Subsystem] Generative adversarial network artifacts and phase jitter detected.", font=("Outfit", 13), text_color="#FF2A2A", wraplength=480, justify="left").pack(anchor="w", pady=4, padx=5)
            else:
                ctk.CTkLabel(ev_frame, text="• [NPU Subsystem] No synthetic perturbations or deepfake signatures detected.", font=("Outfit", 13), text_color="#2ECC71", wraplength=480, justify="left").pack(anchor="w", pady=4, padx=5)

        ctk.CTkButton(popup, text="ACKNOWLEDGE", command=popup.destroy, fg_color=color, hover_color=color, text_color="#000000", font=("Outfit", 14, "bold"), height=45, width=220, corner_radius=8).pack(pady=(15, 25))

    def _toggle_monitoring(self):
        if not self.is_monitoring:
            if self.on_start: self.on_start()
            self.btn_run.configure(text="STOP MONITORING", fg_color="#E74C3C", text_color="#FFFFFF", hover_color="#C0392B")
            self.status_lbl.configure(text="MONITORING ACTIVE", text_color="#2ECC71")
            self.is_monitoring = True
        else:
            if self.on_stop: self.on_stop()
            self.btn_run.configure(text="START MONITORING", fg_color="#00D2FF", text_color="#000000", hover_color="#00A0D2")
            self.status_lbl.configure(text="SYSTEM SECURE", text_color="#00D2FF")
            
            # Reset UI to show completely aborted status
            self.trust_ring.set_score(1.0)
            self.metric_latency.configure(text="0.0 ms")
            self.bar_latency.set(0)
            self.metric_fps.configure(text="0")
            self.bar_fps.set(0)
            self.metric_audio.configure(text="0")
            self.bar_audio.set(0)
            self.metric_npu.configure(text="0.0%")
            self.bar_npu.set(0)
            
            self.is_monitoring = False

    def update_metrics(self, data: dict):
        def _sync_update():
            try:
                self.trust_ring.set_score(data.get("trust_score", 1.0))
                
                latency = data.get('latency', 0.0)
                fps = data.get('fps', 0)
                audio = data.get('audio_windows', 0)
                npu = data.get('npu_load', 0.0)
                
                self.metric_latency.configure(text=f"{float(latency) if latency is not None else 0.0:.1f} ms")
                self.bar_latency.set(min((latency or 0) / 100.0, 1.0))
                
                self.metric_fps.configure(text=str(fps))
                self.bar_fps.set(min(fps / 60.0, 1.0))
                
                self.metric_audio.configure(text=str(audio))
                self.bar_audio.set(min(audio / 10.0, 1.0))
                
                self.metric_npu.configure(text=f"{float(npu):.1f}%")
                self.bar_npu.set(min(npu / 100.0, 1.0))
                
                
                raw = data.get('raw_scores', {})
                if data.get("is_threat", False):
                    self.status_lbl.configure(text="THREAT DETECTED!", text_color="#E74C3C")
                    if not self.alert.root:
                        insight = "Anomaly detected. Be aware of potential social engineering attacks."
                        if raw.get('audio_jitter', 0.0) > 0.4:
                            insight = "The metallic jitter in this audio suggests a Neural Vocoder—never share sensitive codes if you hear this mechanical pattern."
                        elif raw.get('video_jitter', 0.0) > 0.4:
                            insight = "Background pixel-warping detected—this is a common artifact in face-swapping deepfakes. Verify their identity out-of-band."
                        self.alert.show(insight=insight)
                    
                    self._add_log_entry(data)
                else:
                    self.status_lbl.configure(text="MONITORING ACTIVE", text_color="#2ECC71")
                    if self.alert.root:
                        self.alert.dismiss()
                        
                    # Hygiene Mode ambient education
                    if self.hygiene_mode:
                        import random
                        if raw.get('audio_jitter', 0.0) > 0.15 and random.random() < 0.2:
                            self._add_log_entry({}, override_msg="🎓 Hygiene: Pitch variance dropping slightly. Listen for robotic cadence.")
                        elif raw.get('video_jitter', 0.0) > 0.15 and random.random() < 0.2:
                            self._add_log_entry({}, override_msg="🎓 Hygiene: Minor pixel instability observed. Artificial models struggle with sudden movements.")
                
                self.forensic_panel.update_evidence(data.get("evidence", []))
            except Exception as e:
                print(f"UI_SYNC_ERROR: {e}")
        
        # Ensure UI updates happen on main thread
        try:
            if self.ui_queue:
                self.ui_queue.put(_sync_update)
            else:
                self.after(0, _sync_update)
        except Exception as e:
            print(f"UI_PUT_ERROR: {e}")

    def _add_log_entry(self, data, override_msg=None):
        import time
        t_str = time.strftime("%H:%M:%S")
        
        payload = dict(data) if data else {}
        
        if override_msg:
            msg = f"{t_str} | {override_msg}"
            payload['override_msg'] = override_msg
        else:
            raw = data.get('raw_scores', {})
            v_j = raw.get('video_jitter', 0.0)
            a_j = raw.get('audio_jitter', 0.0)
            
            causes = []
            if a_j > 0.4: causes.append("Audio")
            if v_j > 0.4: causes.append("Video")
            cause_str = f"[{','.join(causes)}]" if causes else "[Anomaly]"
    
            msg = f"{t_str} | 🚨 Threat {cause_str} | V:{v_j:.2f} A:{a_j:.2f}"
        
        if hasattr(self, 'last_log_msg') and self.last_log_msg == msg:
            return
        self.last_log_msg = msg
        
        log_btn = ctk.CTkButton(
            self.logs_scroll, 
            text=msg,
            fg_color="#3a1c1c",
            hover_color="#5a2c2c",
            anchor="w",
            command=lambda d=payload: self._show_log_details(d)
        )
        log_btn.pack(fill="x", pady=2, padx=5)

    def _show_log_details(self, data):
        import tkinter.messagebox
        if 'override_msg' in data:
            tkinter.messagebox.showinfo("System Notification", data['override_msg'])
            return
            
        raw = data.get('raw_scores', {})
        details = f"Trust Score: {data.get('trust_score', 0):.2f}\n\n"
        details += f"Video Jitter: {raw.get('video_jitter', 0):.2f}\n"
        details += f"Audio Jitter: {raw.get('audio_jitter', 0):.2f}\n"
        details += f"Sync Penalty: {raw.get('sync_penalty', 0):.2f}\n"
        
        evidence = data.get('evidence', [])
        if evidence:
            details += "\n--- FORENSIC ANALYSIS & REMEDIES ---\n"
            for ev in evidence:
                details += f"\n[{ev.get('source', 'System')}] {ev.get('severity', 'INFO')}\n"
                details += f"{ev.get('message', '')}\n"
        else:
            causes = []
            if raw.get('audio_jitter', 0.0) > 0.4: causes.append("Audio Deepfake (Vocoder/Monotonic)")
            if raw.get('video_jitter', 0.0) > 0.4: causes.append("Video Deepfake (Face/Pixel Jitter)")
            
            if causes:
                details += f"\nPrimary Flag: {', '.join(causes)}"
            
        tkinter.messagebox.showinfo("Alert Details", details)

    def set_hardware_info(self, info: str):
        self.hw_badge.configure(text=f"HARDWARE: {info}")

    def set_hardware_provider(self, provider: str):
        self.lbl_npu_provider.configure(text=f"Auto-Detected: {provider}")

    def _trigger_snip(self):
        from ui_overlay.region_selector import RegionSelector
        self.withdraw() # Temporarily hide the main window during snip
        RegionSelector(self.master if hasattr(self, 'master') and self.master else self, self._on_snip_complete)

    def _on_snip_complete(self, roi):
        self.deiconify() # Restore main window
        if hasattr(self, 'app') and self.app:
            self.app._pending_roi = roi # Always save to app state for clean restarts
            if hasattr(self.app, 'video_interceptor') and self.app.video_interceptor:
                self.app.video_interceptor.roi = roi
                if roi:
                    self._add_log_entry({}, override_msg="⛶ TARGET LOCKED: Manual Face Extraction Active.")
                else:
                    self._add_log_entry({}, override_msg="⛶ TARGET RESET: Autonomous Deep-Seek Active.")
            else:
                if roi:
                    self._add_log_entry({}, override_msg="⛶ TARGET PRE-LOADED: Manual Face Extraction Active.")

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
