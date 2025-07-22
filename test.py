import dearpygui.dearpygui as dpg
import time

dpg.create_context()
dpg.create_viewport(title='Dynamic UI Example', width=600, height=200)
dpg.setup_dearpygui()

with dpg.window(label="Example Window", tag="main_window"):
    dpg.add_text("Hello, world")

dpg.show_viewport()

frame_count = 0
while dpg.is_dearpygui_running():
    frame_count += 1

    # Dynamically add an item after some frames
    if frame_count == 120:  # ~2 seconds later at 60 FPS
        dpg.add_text("Added at runtime!", parent="main_window")
    if frame_count == 240:
        dpg.add_button(label="Click Me!", parent="main_window")

    dpg.render_dearpygui_frame()
    time.sleep(1/60)  # optional to limit loop speed

dpg.destroy_context()