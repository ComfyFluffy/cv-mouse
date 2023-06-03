class MouseController:
    last_update_time = 0
    last_coordinates: tuple[float, float]

    left_button_pressed = False
    right_button_pressed = False

    def update_position(self, x: float, y: float):
        '''
        Move the mouse according to the new position (x, y).
        The movement is relative to the last position.
        If the last_update_time is zero, we will only update the values and not move the mouse.
        The actual movement of mouse is computed by compute_acceleration().
        '''

    def compute_with_acceleration(self, dx: float, dy: float,
                                  dt: float) -> tuple[float, float]:
        '''
        Compute the relative movement of the mouse according to the position change (dx, dy) and time change dt.
        The faster the movement, the larger the acceleration.
        '''
