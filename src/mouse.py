import math
import time
from pyautogui import mouseDown, mouseUp, moveRel


class MouseController:
    last_update_time = 0
    last_coordinates: tuple[float, float]

    __left_button_pressed = False
    __right_button_pressed = False

    def update_position(self, x: float, y: float, aspect_ratio: float):
        '''
        Move the mouse according to the new position (x, y).
        The value of x and y should be in [0, 1].
        The aspect_ratio is x/y in pixel.
        The movement is relative to the last position.
        If the last_update_time is zero, we will only update the values and not move the mouse.
        The actual movement of mouse is computed by compute_with_acceleration().
        '''

        # Check if this is the first update
        if self.last_update_time == 0:
            self.last_update_time = time.time()
            self.last_coordinates = (x, y)
            return

        current_time = time.time()

        dt = current_time - self.last_update_time

        # Compute the relative position change (dx, dy)
        dx = x - self.last_coordinates[0]
        dy = y - self.last_coordinates[1]

        # Adjust for aspect ratio
        dx *= aspect_ratio

        # Update the last known values
        self.last_update_time = current_time
        self.last_coordinates = (x, y)

        # Compute the pixel movement
        mouse_x, mouse_y = self.compute_with_acceleration(dx, dy, dt)

        moveRel(mouse_x, mouse_y)

    def compute_with_acceleration(self, dx: float, dy: float,
                                  dt: float) -> tuple[int, int]:
        '''
        Compute the relative movement of the mouse according to the position change (dx, dy) and time change dt.
        The faster the movement, the larger the acceleration.

        ## Returns

        The pixel movement of the mouse.
        '''

        # Compute the speed
        speed = math.sqrt(dx * dx + dy * dy) / dt

        # Compute acceleration as a function of speed. This is a simple
        # linear acceleration - if you move twice as fast, the mouse moves
        # twice as far. Adjust as necessary for your specific needs.
        acceleration = speed * 50

        # Apply acceleration to movement
        pixel_dx = int(dx * acceleration)
        pixel_dy = int(dy * acceleration)

        return pixel_dx, pixel_dy

    def __set_left_pressed(self, value: bool):
        if value:
            if not self.__left_button_pressed:
                mouseDown()
        elif self.__left_button_pressed:
            mouseUp()
        self.__left_button_pressed = value

    def __set_right_pressed(self, value: bool):
        if value:
            if not self.__right_button_pressed:
                mouseDown(button='right')
        elif self.__right_button_pressed:
            mouseUp(button='right')
        self.__right_button_pressed = value

    left_pressed = property(lambda self: self.__left_button_pressed,
                            __set_left_pressed)
    right_pressed = property(lambda self: self.__right_button_pressed,
                             __set_right_pressed)
