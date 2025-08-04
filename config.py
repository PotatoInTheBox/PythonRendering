# config.py

from typing import Generic, TypeVar

import numpy as np

T = TypeVar('T')

class ConfigEntry(Generic[T]):
    def __init__(self, default_val: T, name=None, mutable=True):
        self._mutable = True  # Allow it to be mutable at the start
        if name == None:
            name = f"UnnamedConfigEntry_{id(self)}"
        self.name = name
        self.val = default_val
        self._mutable = mutable  # Then decide whether to remain mutable

    @property
    def val(self) -> T:
        return self._val

    @val.setter
    def val(self, new_val: T):
        if not self._mutable:
            raise AttributeError(f"{self.name} is immutable")
        self._val = new_val

class Config:
    def __init__(self):
        # TODO: In the future these values may be moved to a config file which
        # can then be loaded. This way instead of modifying the code we can modify
        # a file. However, at the moment we are programming in Python which is more
        # of a scripting language so configs in the code are probably acceptable
        # as long as they are easy to find/read/write.
        
        # === Initial settings ===
        # These settings are set once. They cannot be changed afterwards
        # (changing them will either do nothing or cause issues)
        START_DISTANCE = 4.0
        CAMERA_POSITION = np.array([0.0,0.0,float(START_DISTANCE)])
        self.CAMERA_START_POSITION = ConfigEntry(CAMERA_POSITION, mutable=False)
        CAMERA_ROTATION = np.array([0.0,0.0,0])
        self.CAMERA_START_ROTATION = ConfigEntry(CAMERA_ROTATION, mutable=False)
        
        # === Screen settings ===
        self.screen_width = ConfigEntry(1200, mutable=False)
        self.screen_height = ConfigEntry(800, mutable=False)
        self.cell_size = ConfigEntry(6, mutable=True)

        # === Camera settings ===
        self.camera_speed = ConfigEntry(0.1)
        self.camera_sensitivity = ConfigEntry(1.0)
        self.fov = ConfigEntry(90.0)  # In degrees, how much can the camera see from left to right?

        # === Rendering toggles ===
        self.draw_faces = ConfigEntry(True)
        self.draw_lines = ConfigEntry(False)
        self.draw_z_buffer = ConfigEntry(False)
        
        # Wave Settings
        self.wave_amplitude = ConfigEntry(0.4)  # bigger number = taller wave
        self.wave_period = ConfigEntry(3.0)  # bigger number = shorter wave
        self.wave_speed = ConfigEntry(0.01)  # The speed/increment of the wave, based on frame count
        
        # Camera transform
        self.camera_position = ConfigEntry(CAMERA_POSITION)
        self.camera_rotation = ConfigEntry(CAMERA_ROTATION)

        # === Debug values (runtime-only) ===
        # self.frame_time = 0.0

    def reset_defaults(self):
        """Resets all configs to their default values."""
        self.__init__()  # Simple way to restore defaults
    



# Global instance
config = Config()
