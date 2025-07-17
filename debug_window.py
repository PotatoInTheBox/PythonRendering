# debug_window.py

import dearpygui.dearpygui as dpg
import threading
import time


class DebugWindow:
    def __init__(self):
        dpg.create_context()
        dpg.create_viewport(title='Dynamic UI Example', width=600, height=200)
        dpg.setup_dearpygui()
        self.controls = {}

        with dpg.window(label="Debug", width=400, height=400) as self.window_id:
            dpg.add_text("Hello, world")
            pass  # We'll add widgets dynamically
        
        dpg.show_viewport()
    
    # === INPUT ELEMENTS ===
    def create_number_input(self, label, config_ref, on_change=None):
        self.controls[label] = config_ref
        dpg.add_input_float(label=label, default_value=config_ref.val,
                            callback=lambda s, a: self._update_config(label, a, on_change),
                            parent=self.window_id)

    def create_slider_input_int(self, label, config_ref, min_val=0, max_val=10, on_change=None):
        self.controls[label] = config_ref
        dpg.add_slider_int(label=label, default_value=config_ref.val, min_value=min_val, max_value=max_val,
                            callback=lambda s, a: self._update_config(label, a, on_change),
                            parent=self.window_id)

    def create_slider_input_float(self, label, config_ref, min_val=0.0, max_val=1.0, on_change=None):
        self.controls[label] = config_ref
        dpg.add_slider_float(label=label, default_value=config_ref.val, min_value=min_val, max_value=max_val,
                            callback=lambda s, a: self._update_config(label, a, on_change),
                            parent=self.window_id)

    def create_checkbox(self, label, config_ref, on_change=None):
        self.controls[label] = config_ref
        dpg.add_checkbox(label=label, default_value=config_ref.val,
                        callback=lambda s, a: self._update_config(label, a, on_change),
                        parent=self.window_id)

    def create_debug_label(self, label, tracked_val):
        self.controls[label] = tracked_val
        dpg.add_text(default_value=f"{label}: {tracked_val.val}",
                    tag=f"dbg_{label}",
                    parent=self.window_id)

    # === INTERNAL UPDATE LOGIC ===
    def _update_config(self, label, value, on_change=None):
        """Updates the referenced variable in real time."""
        ref = self.controls[label]
        ref.val = value  # Mutate in place (config_ref is a list or mutable container)
        if on_change:
            on_change(value)

    def update_debug_labels(self):
        """Updates all debug labels with the latest values."""
        for label, ref in self.controls.items():
            if label.startswith("Debug_"):  # You can define custom naming if needed
                dpg.set_value(f"dbg_{label}", f"{label}: {ref.val}")
    
    def render_ui(self):
        """Render the UI for this frame (call inside your renderer loop)."""

        # self.update_debug_labels()
        # dpg.render_dearpygui_frame()
        dpg.render_dearpygui_frame()

    def shutdown(self):
        dpg.destroy_context()
