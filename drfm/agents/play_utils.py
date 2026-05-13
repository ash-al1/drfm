# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali

import logging

import numpy as np

log = logging.getLogger(__name__)

# World-frame offset from drone position to camera eye.
_CAM_OFFSET = np.array([-5.0, 10.0, 20.0])


class CameraFollower:
    """Locks the viewport camera onto a moving target until the user manually moves it.

    Lock releases the first time a mouse button is held and the cursor moves.
    Once released, the camera is never re-locked.
    """

    def __init__(self) -> None:
        # carb/omni modules must be imported after SimulationApp is instantiated;
        # Carbonite's plugin system is not initialised before that point.
        import carb.input
        import omni.appwindow

        self._locked = True
        self._carb_input = carb.input
        self._iinput = carb.input.acquire_input_interface()

        app_window = omni.appwindow.get_default_app_window()
        self._mouse = app_window.get_mouse()

        # Subscribe to mouse events; release lock on first drag (button held + move).
        self._sub_id = self._iinput.subscribe_to_mouse_events(self._mouse, self._on_mouse_event)
        self._button_held = False

    def _on_mouse_event(self, event) -> bool:
        if not self._locked:
            return True
        if event.type in (
            self._carb_input.MouseEventType.LEFT_BUTTON_DOWN,
            self._carb_input.MouseEventType.RIGHT_BUTTON_DOWN,
            self._carb_input.MouseEventType.MIDDLE_BUTTON_DOWN,
        ):
            self._button_held = True
        elif event.type in (
            self._carb_input.MouseEventType.LEFT_BUTTON_UP,
            self._carb_input.MouseEventType.RIGHT_BUTTON_UP,
            self._carb_input.MouseEventType.MIDDLE_BUTTON_UP,
        ):
            self._button_held = False
        elif event.type == self._carb_input.MouseEventType.MOVE and self._button_held:
            self._locked = False
            self._iinput.unsubscribe_to_mouse_events(self._mouse, self._sub_id)
            log.info("Camera lock released by user input.")
        return True

    def update(self, drone_pos: np.ndarray) -> None:
        """Move the viewport camera to follow drone_pos, or skip if lock is released."""
        if not self._locked:
            return

        # isaacsim.core must be imported after SimulationApp is instantiated;
        # Carbonite's plugin system is not initialised before that point.
        from isaacsim.core.utils.viewports import set_camera_view

        eye = drone_pos + _CAM_OFFSET
        set_camera_view(eye=eye, target=drone_pos)
