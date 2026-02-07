import pyautogui
import screeninfo
import time


class VirtualMouse:
    def __init__(self):
        pyautogui.FAILSAFE = False
        monitor = screeninfo.get_monitors()[0]
        self.screen_w = monitor.width
        self.screen_h = monitor.height

        self.smooth_factor = 6  # Adjust for feel
        self.prev_x, self.prev_y = None, None

        # Gesture action cooldowns to prevent multiple triggers
        self.last_click_time = 0
        self.last_right_click_time = 0
        self.last_scroll_time = 0
        self.click_cooldown = 0.5  # 500ms cooldown between clicks
        self.scroll_cooldown = 0.1  # 100ms cooldown between scrolls

    def move_cursor(self, index_tip, frame_w, frame_h):
        """Move cursor based on index finger tip position"""
        # Normal feed + inverted X for correct left/right
        x = (1 - index_tip.x) * self.screen_w  # Invert X to fix opposite direction
        y = index_tip.y * self.screen_h

        # Smoothing
        if self.prev_x is not None:
            x = self.prev_x + (x - self.prev_x) / self.smooth_factor
            y = self.prev_y + (y - self.prev_y) / self.smooth_factor

        pyautogui.moveTo(x, y)
        self.prev_x, self.prev_y = x, y

    def click(self):
        """Perform left click with cooldown"""
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            pyautogui.click()
            self.last_click_time = current_time
            return True
        return False

    def double_click(self):
        """Perform double click with cooldown"""
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            pyautogui.doubleClick()
            self.last_click_time = current_time
            return True
        return False

    def right_click(self):
        """Perform right click with cooldown"""
        current_time = time.time()
        if current_time - self.last_right_click_time > self.click_cooldown:
            pyautogui.rightClick()
            self.last_right_click_time = current_time
            return True
        return False

    def scroll_up(self, amount=3):
        """Scroll up with cooldown"""
        current_time = time.time()
        if current_time - self.last_scroll_time > self.scroll_cooldown:
            pyautogui.scroll(amount)
            self.last_scroll_time = current_time
            return True
        return False

    def scroll_down(self, amount=3):
        """Scroll down with cooldown"""
        current_time = time.time()
        if current_time - self.last_scroll_time > self.scroll_cooldown:
            pyautogui.scroll(-amount)
            self.last_scroll_time = current_time
            return True
        return False

    def drag_start(self):
        """Start dragging (press and hold)"""
        pyautogui.mouseDown()

    def drag_end(self):
        """End dragging (release)"""
        pyautogui.mouseUp()