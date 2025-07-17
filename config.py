# config.py

from typing import Generic, TypeVar

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
        # === Screen settings ===
        self.screen_width = ConfigEntry(1200, mutable=False)
        self.screen_height = ConfigEntry(800, mutable=False)
        self.cell_size = ConfigEntry(6, mutable=True)

        # === Camera settings ===
        self.camera_speed = ConfigEntry(0.1)
        # self.start_distance = 4.0
        # self.camera_position = [0.0, 0.0, 4.0]  # Mutable vector
        # self.camera_rotation = [0.0, 0.0, 0.0]
        # self.fov = 90

        # === Rendering toggles ===
        self.draw_faces = ConfigEntry(True)
        self.draw_lines = ConfigEntry(False)
        self.draw_z_buffer = ConfigEntry(False)

        # === Debug values (runtime-only) ===
        # self.frame_time = 0.0

    def reset_defaults(self):
        """Resets all configs to their default values."""
        self.__init__()  # Simple way to restore defaults
    



# Global instance
config = Config()
