# Disable the pylint error for the next line
# pylint: disable=missing-module-docstring
import ctypes
import os
import pathlib
import string
import subprocess
import time
from typing import Dict, Tuple

import cv2
import keyboard
import numpy as np
import pyautogui
import pymem
import pywinauto
import win32con
import win32gui
import win32ui
from PIL import Image
# Disable the pylint error for the next line
# pylint: disable=no-name-in-module
from cv2 import cuda
from numpy import ndarray

from game_inputs import GameInputs
from gps import GPS
from lap_progress import LapProgress
from lap_time import LapTime
from speedometer import Speedometer
from strategy.gps.gps_strategy_enum import GPSStrategyEnum
from strategy.gps_image_recognition.a_gps_ircgn_strategy import AGpsImageRecognitionStrategy
from strategy.gps_image_recognition.gps_ircgn_strategy_cpu import GpsImageRecognitionStrategyCPU
from strategy.gps_image_recognition.gps_ircgn_strategy_gpu import GpsImageRecognitionStrategyGPU
from utils.enums.restart_states_enum import RestartStateEnum
from utils.singleton.controls import Controls


# Pylint disable warnings about too many attributes and methods
# pylint: disable=too-many-instance-attributes,too-many-public-methods
class Game:
    """
    Class Which Acts as an API with the game Need for Speed: Most Wanted (2005)
    """
    a_threshold = 100
    a_ratio = 3

    a_loop_time = time.time()
    a_width: int
    a_height: int
    a_image: ndarray
    a_screenshot: ndarray
    a_interest_rect_vert: list[tuple[float, float]]

    a_game_filename: string = r'F:\Games Folder\Electronic Arts\Need for Speed Most ' \
                              r'Wanted\speed.exe'
    a_cheat_engine_filename: string = r'C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64.exe'

    a_gps: GPS

    a_speedometer: Speedometer

    a_lap_progress: LapProgress

    a_lap_time: LapTime

    a_is_recording: bool

    a_user23 = ctypes.windll.user32

    a_list_bitmap: []

    a_controls: Controls

    a_speed: int

    a_car_distance_offset: float

    a_car_direction_offset: int

    a_race_initialised: bool

    a_cycles_passed: int

    a_cuda_device: None

    a_gps_img_rcg_strategy: AGpsImageRecognitionStrategy

    a_gps_strategy_enum: GPSStrategyEnum

    def __init__(self) -> None:

        self.comparable_images: Dict[str, np.ndarray] = {}
        self.load_comparable_images()
        self.a_race_initialised = False
        self.a_car_distance_offset = 0
        self.a_car_direction_offset = 0
        self.a_cycles_passed = 0

        cuda.printCudaDeviceInfo(0)

        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                self.a_gps_img_rcg_strategy = GpsImageRecognitionStrategyGPU()
                self.a_gps_strategy_enum = GPSStrategyEnum.GPU
            else:
                self.a_gps_img_rcg_strategy = GpsImageRecognitionStrategyCPU()
                self.a_gps_strategy_enum = GPSStrategyEnum.CPU
        except cv2.error:
            GpsImageRecognitionStrategyCPU()
            self.a_gps_strategy_enum = GPSStrategyEnum.CPU

    def init_game_memory_objects(self) -> None:
        """
        Initializes Speedometer, Lap Progress and Lap Time From Game Memory Reading
        """
        self.a_speedometer.construct()
        self.a_lap_progress.construct()
        self.a_lap_time.construct()

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    def main_loop(self, par_game_inputs: GameInputs):
        """
        Main Loop That Controls All The Game Logic

        Args:
            par_game_inputs (GameInputs): An instance of the GameInputs class containing the
                inputs for the game.

        Returns:
            None: This method doesn't return anything.
        """
        self.a_speed = 3

        self.a_list_bitmap = []

        self.a_controls = Controls()

        self.start_game()
        # self.start_cheat_engine()

        print(cv2.__version__)

        while not self.process_exists("speed.exe"):
            print("Waiting for Game to Start")
            time.sleep(3)

        self.a_speedometer = Speedometer()

        self.a_lap_progress = LapProgress()

        self.a_lap_time = LapTime()

        self.a_gps = GPS(self.a_gps_strategy_enum)

        self.start_cheat_engine()
        self.set_speed(self.a_speed)

        self.a_is_recording = False

        tmp_frame_counter: int = 0

        tmp_start_time: float = time.time()

        tmp_speed_constant = 1 / self.a_speed

        self.focus_on_game()

        time.sleep(5)

        self.init_game_memory_objects()

        self.reconfigure_speed(4)

        self.init_game_race(0.7 / float(4), 0.1 / float(4))

        self.reconfigure_speed(self.a_speed)

        time.sleep(3)

        self.a_race_initialised = True

        self.a_cycles_passed = 0

        par_game_inputs.game_initialization_inputs.put((self.a_race_initialised, self.a_speed))

        while True:

            # Capture screenshot and convert to numpy array
            self.a_screenshot, self.a_width, self.a_height = self.window_capture()
            self.a_screenshot = np.array(self.a_screenshot)

            # Define the region of interest
            self.a_interest_rect_vert = [
                (0, self.a_height),
                (self.a_width / 2, self.a_height / 2),
                (self.a_width, self.a_height)
            ]

            # Check for quit key
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

            # Check for record key
            if keyboard.is_pressed('r'):
                self.a_is_recording = True
                self.a_list_bitmap = []

            tmp_speed_mph = self.a_speedometer.return_speed_mph()
            tmp_lap_progress = self.a_lap_progress.return_lap_completed_percent()

            # tmp_car_offset, tmp_contour = self.calc_car_offset(self.a_screenshot)
            tmp_car_offset_distance: float
            tmp_contour: list
            tmp_car_offset_direction: int
            tmp_car_offset_distance, tmp_contour, tmp_car_offset_direction = \
                self.a_gps_img_rcg_strategy.calc_car_offset(self.a_gps, self.a_screenshot)

            self.a_car_distance_offset = tmp_car_offset_distance
            self.a_car_direction_offset = tmp_car_offset_direction

            # tmp_lap_time = self.a_lap_time.return_lap_time()
            if tmp_contour is not None:
                cv2.drawContours(self.a_screenshot, [tmp_contour], -1, (255, 0, 255), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (159, 43, 104)
            thickness = 2
            line_type = 2

            bottom_left_corner_of_text = (10, int(self.a_height / 4))
            cv2.putText(self.a_screenshot, "Speed: " + str(tmp_speed_mph) + " / mph",
                        bottom_left_corner_of_text,
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        line_type)

            bottom_left_corner_of_text = (10, int(self.a_height / 3))
            cv2.putText(self.a_screenshot,
                        "Road Offset: " + str(round(tmp_car_offset_distance, 2)) + "",
                        bottom_left_corner_of_text,
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        line_type)

            bottom_left_corner_of_text = (10, int(self.a_height / 2.4))
            cv2.putText(self.a_screenshot, "Completed: " + str(round(tmp_lap_progress, 2)) + "%",
                        bottom_left_corner_of_text,
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        line_type)

            bottom_left_corner_of_text = (10, int(self.a_height / 2))
            cv2.putText(self.a_screenshot,
                        "Incline: " + str(self.a_gps.translate_direction_offset_to_string(
                            tmp_car_offset_direction))
                        + "",
                        bottom_left_corner_of_text,
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        line_type)

            cv2.imshow('Main Vision', self.a_screenshot)

            tmp_frame_counter += tmp_speed_constant

            if (time.time() - tmp_start_time) > tmp_speed_constant:
                print("FPS: ", tmp_frame_counter / (time.time() - tmp_start_time))
                tmp_frame_counter = 0
                tmp_start_time = time.time()

            self.a_cycles_passed += 1

            while not par_game_inputs.game_restart_inputs.empty():
                tmp_needs_restart: bool = par_game_inputs.game_restart_inputs.get()
                if tmp_needs_restart:
                    par_game_inputs.game_restart_inputs.put(tmp_needs_restart)
                    self.reset_game_race(0.7 / float(self.a_speed), 0.01 / float(self.a_speed))
                    par_game_inputs.game_restart_inputs.get()

            # .empty() returns False or True, it is not function to empty the Queue
            # Maybe some python coders can use function declaration, when they know that it
            # returns bool to use naming like isEmpty? :)
            while not par_game_inputs.agent_inputs_state.empty():
                par_game_inputs.agent_inputs_state.get()
            par_game_inputs.agent_inputs_state.put(
                (self.get_speed_mph(), self.get_car_distance_offset(),
                 self.a_lap_progress.return_lap_completed_percent(),
                 self.a_car_direction_offset))

    def is_race_initialised(self) -> bool:
        """
        Checks is Race is Already Initialised
        :return: Return True is Race is Initialised otherwise False
        """
        return self.a_race_initialised

    def load_comparable_images(self) -> bool:
        """
        Loads Images which are used for comparison e.g. matchTemplate
        :return: Returns True if loaded correctly otherwise False
        """
        self.comparable_images: Dict[str, np.ndarray] = {}
        directory: str = "comparable_images"
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                img: np.ndarray = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_COLOR)
                name: str = os.path.splitext(filename)[0]
                self.comparable_images[name] = img
        return True

    def get_lap_progress(self) -> float:
        """
        Gets Lap Progress as Float
        :return: Lap Progress as Float Value
        """
        return self.a_lap_progress.return_lap_completed_percent()

    def get_car_distance_offset(self) -> float:
        """
        Get Distance of Car from the center of the Road
        Possible Values:
        <-0, -inf) - The Car is outside road contour
        0 - The Car is in the border of road contour
        (0, inf) - The Car is inside the road contour
        :return: Distance as Float Value
        """
        return self.a_car_distance_offset

    def get_car_direction_offset(self) -> int:
        """
        Get Direction from which is car inclined from the centre of the road
        :return: -1 if the car is to the Left of the road centre
        1 if the car is to the Right of the road centre
        0 if the car is pointed straight to the road centre
        """
        return self.a_car_direction_offset

    def get_speed_mph(self) -> float:
        """
        Get Car Speed in Mph
        :return: Float value describing Car Speed in Mph
        """
        return self.a_speedometer.return_speed_mph()

    def make_canny(self, par_grayscale: ndarray):
        """
          Performs Canny edge detection on the input grayscale image.

          Args:
              par_grayscale: A grayscale image as a NumPy array.

          Returns:
              A binary edge map as a NumPy array.
          """
        # return cv2.Canny(par_grayscale, self.a_threshold, self.a_threshold * self.a_ratio)

        # Convert input to a GpuMat
        gpu_input = cv2.cuda_GpuMat(par_grayscale)

        # Perform Canny edge detection on the GPU
        gpu_output = cv2.cuda.createCannyEdgeDetector(self.a_threshold,
                                                      self.a_threshold * self.a_ratio)
        gpu_edges = gpu_output.detect(gpu_input)

        # Download the result to CPU memory
        edges = gpu_edges.download()

        return edges

    def region_of_interest(self, par_img: ndarray, par_vertices):
        """
        Applies a region of interest mask to the input image.

        Args:
            par_img: The input image as a NumPy array.
            par_vertices: A list of vertices defining the region of interest polygon.

        Returns:
            A masked image as a NumPy array, where only the pixels inside the region of interest 
            are retained.
        """
        mask = np.zeros_like(par_img)
        # color_channel_count = img.shape[2]
        create_matched_color_mask = 255
        # Fills Polygons That Are Not For Our Interest
        cv2.fillPoly(mask, par_vertices, create_matched_color_mask)
        # Return Image Where Only The Mask Pixel Matches
        masked_image = cv2.bitwise_and(par_img, mask)
        return masked_image

    def window_capture(self) -> Tuple[np.ndarray, int, int]:
        """
           Capture the game window and return the captured image along with its width and height.

           Returns:
               Tuple containing:
               - np.ndarray: Captured image
               - int: Width of the captured image
               - int: Height of the captured image
        """
        # Find the game window
        hwnd = win32gui.FindWindow(None, 'Need for Speed\u2122 Most Wanted')

        # Get the window device context
        w_dc = win32gui.GetWindowDC(hwnd)

        rect = win32gui.GetWindowRect(hwnd)

        width = rect[2] - rect[0]
        height = rect[3] - rect[1]

        border_pixels = 8
        titlebar_pixels = 30
        width = width - (border_pixels * 2)
        height = height - titlebar_pixels - border_pixels
        x_cropped = border_pixels
        y_cropped = titlebar_pixels

        # Win Image Data
        dc_obj = win32ui.CreateDCFromHandle(w_dc)
        c_dc = dc_obj.CreateCompatibleDC()
        data_bit_map = win32ui.CreateBitmap()
        data_bit_map.CreateCompatibleBitmap(dc_obj, width, height)
        c_dc.SelectObject(data_bit_map)
        c_dc.BitBlt((0, 0), (width, height), dc_obj, (x_cropped, y_cropped), win32con.SRCCOPY)

        # Save Screen
        # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)

        signed_ints_array = data_bit_map.GetBitmapBits(True)
        img = np.fromstring(signed_ints_array, dtype='uint8')

        img.shape = (height, width, 4)

        # Free Resources
        dc_obj.DeleteDC()
        c_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, w_dc)
        win32gui.DeleteObject(data_bit_map.GetHandle())

        img = img[..., :3]
        img = np.ascontiguousarray(img)

        if self.a_is_recording:
            self.record(img, self.a_is_recording)

        try:
            win32gui.SetForegroundWindow(hwnd)
        except win32gui.error as win32gui_exception:
            print("An error occurred while setting the foreground window (Probably Debugging): ",
                  win32gui_exception)

        return img, width, height

    def test_file_exec(self, par_file: string) -> bool:
        """
            Checks if the file exists and is executable.

            Args:
                par_file (str): The filename to check.

            Returns:
                bool: True if the file exists and is executable, False otherwise.
        """
        if not os.path.exists(par_file):
            return False
        if not os.access(par_file, os.X_OK):
            return False
        return True

    def start_game(self):
        """
         Starts the game application.
         """
        if self.test_file_exec(self.a_game_filename):
            start_game_bat = str(pathlib.Path.cwd()) + '\\utils\\start.bat'
            print(start_game_bat)
            subprocess.run([start_game_bat], check=False)
            return
        print('Game From File: ' + self.a_game_filename + ' cannot be started')
        return

    def process_exists(self, par_process_name: string) -> bool:
        """
          Checks if a process with the given name is currently running.

          Args:
              par_process_name (str): The name of the process to check.

          Returns:
              bool: True if the process is running, False otherwise.
        """
        call = f'TASKLIST /FI "imagename eq {par_process_name}"'
        # use buildin check_output right away
        output = subprocess.check_output(call).decode()
        # check in last line for process name
        last_line = output.strip().rsplit('\r\n', maxsplit=1)[-1]
        # because Fail message could be translated
        return last_line.lower().startswith(par_process_name.lower())

    def capture_race(self, par_lap_progress: float):
        """
            Captures a screenshot of the game screen and saves it.

          Args:
              par_lap_progress (float): The lap progress in the game.

          Returns:
              None
        """
        if par_lap_progress is not None:
            self.save_image()

    def start_cheat_engine(self) -> None:
        """
            Starts the Cheat Engine 7.4 application and focuses on its window.

            Returns:
                None
        """
        app = None

        process_is_running = True
        try:
            app = pywinauto.Application().connect(title="Cheat Engine 7.4", found_index=1)
        except pywinauto.application.ProcessNotFoundError:
            process_is_running = False
        except pywinauto.findwindows.ElementNotFoundError:
            process_is_running = False

        if not process_is_running:
            app = pywinauto.Application().start(
                r'C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64-SSE4-AVX2.exe')

        window = app.window(title="Cheat Engine 7.4", found_index=0)
        window.set_focus()

        text_boxes = []
        controls = []
        windows = []

        # Iterate over all windows in the app
        for window in app.windows():
            # Iterate over all controls in the window
            for control in window.children():
                # Check if the control is a text box

                controls.append(control)
                windows.append(window)

                if isinstance(control, pywinauto.controls.uiawrapper.UIAWrapper):
                    # Add the control to the list of text boxes
                    text_boxes.append(control)

        # Get the main window of the application

        main_window = app.window(title="Cheat Engine 7.4", found_index=0)

        app_rect = main_window.rectangle()
        app_position_process = (app_rect.left + (app_rect.right - app_rect.left) * 1 / 40,
                                app_rect.top + (app_rect.bottom - app_rect.top) * 1 / 10)
        pyautogui.click(app_position_process[0], app_position_process[1])

        file_menu = app.window(title="Process List")

        file_menu.print_control_identifiers()

        test = file_menu.children()

        list_apps = test[0]

        items = list_apps.texts()

        # ruff-disable-line
        game_name: str = 'Need for Speed\u2122 Most Wanted'

        index = 0
        index_found = -1
        for string_item in items:
            if game_name in string_item:
                index_found = index
                break
            index = index + 1

        list_apps.select(index_found - 1)

        file_menu.child_window(title="Open").click()

    def set_speed(self, par_speed: int):
        """
        Sets Game Speed Via Cheat Engine.
        The Speed increases the Tick Rate of the Game.
        Meaning everything will be sped up.
        :param par_speed: Speed to set described as integer
        """
        app = pywinauto.Application().connect(title="Cheat Engine 7.4", found_index=1)

        if app is None:
            app = pywinauto.Application().start(
                r'C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64-SSE4-AVX2.exe')

        main_window = app.window(title="Cheat Engine 7.4", found_index=0)
        # Find the control by its name
        control = main_window.child_window(title="Enable Speedhack")
        control.click_input()

        speed_edit = main_window.child_window(class_name="Edit", found_index=0)
        speed_edit.set_edit_text(par_speed)

        speed_edit = main_window.child_window(title="Apply")
        speed_edit.click()
        
    def reconfigure_speed(self, par_speed: int):
        """
        Reconfigures The Game Speed via Already Running Cheat Engine
        The Speed increases the Tick Rate of the Game.
        Meaning everything will be sped up.
        :param par_speed: Speed to set described as integer
        """
        app = pywinauto.Application().connect(title="Cheat Engine 7.4", found_index=1)

        main_window = app.window(title="Cheat Engine 7.4", found_index=0)

        speed_edit = main_window.child_window(class_name="Edit", found_index=0)
        speed_edit.set_edit_text(par_speed)

        speed_edit = main_window.child_window(title="Apply")
        speed_edit.click()

    def get_game_speed(self) -> int:
        """
          Get Game Speed Via Cheat Engine.
          The Speed increases the Tick Rate of the Game.
          Meaning everything will be sped up.
          """
        app = pywinauto.Application().connect(title="Cheat Engine 7.4", found_index=1)

        if app is None:
            app = pywinauto.Application().start(
                r'C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64-SSE4-AVX2.exe')

        main_window = app.window(title="Cheat Engine 7.4", found_index=0)

        speed_edit = main_window.child_window(class_name="Edit", found_index=0)

        return int(speed_edit.window_text())

    def record(self, bitmap: ndarray, recording: bool):
        """
        Records bitmap frames while a lap is being completed and saves them as images.

        Args:
        bitmap (numpy.ndarray): A bitmap image frame to be recorded.
        recording (bool): A flag indicating whether the recording is in progress.

        Returns:
        None
        """
        if self.a_lap_progress.return_lap_completed_percent() < 2 and recording:
            self.a_list_bitmap.append(bitmap)
            self.a_controls.press_key(self.a_controls.UP_KEY)
        elif recording:
            self.a_controls.release_key(self.a_controls.UP_KEY)
            self.save_image()

    def save_image(self):
        """
        Saves Image To The Recording Folder
        """
        # pylint: disable=consider-using-enumerate
        for i in range(len(self.a_list_bitmap)):
            img = Image.fromarray(self.a_list_bitmap[i])
            img.save("record/" + str(self.get_game_speed()) + "/" + str(i) + ".png")
        self.a_is_recording = False
        self.a_list_bitmap = []

    def focus_on_game(self) -> None:
        """
        Forces Windows to Focus On Game (Bring It To Front)
        """
        hwnd = win32gui.FindWindow(None, 'Need for Speed\u2122 Most Wanted')
        win32gui.SetForegroundWindow(hwnd)

    def match_template(self, par_original_image: ndarray, par_template_to_match: ndarray) -> bool:
        """
            Matches a template image with an original image using OpenCV's template matching.

            Args:
                par_original_image: A numpy array representing the original image.
                par_template_to_match: A numpy array representing the template image to match.

            Returns:
                A boolean value indicating whether the template image was found in the original
                 image.
        """
        # Convert to grayscale
        screen_gray = cv2.cvtColor(par_original_image, cv2.COLOR_BGR2GRAY)
        restart_state_gray = cv2.cvtColor(par_template_to_match, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        res = cv2.matchTemplate(screen_gray, restart_state_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        print(loc)

        # Return boolean based on whether match was found or not
        return len(loc[0]) > 0

    def is_in_correct_restart_state(self, par_screen_image: ndarray) -> RestartStateEnum:
        """
        Check if the screen image is in the correct restart state.

        Args:
            par_screen_image: A numpy array of the screen image.

        Returns:
            A value of RestartStateEnum indicating whether the screen is in the restart state, 
            standings state or an unknown state.
        """
        restart_menu: str = 'resume_race_text'
        standing_menu: str = 'standings_menu'
        if restart_menu not in self.comparable_images:
            print(f"No image found with name {restart_menu}")
            return RestartStateEnum.UNKNOWN_STATE
        if standing_menu not in self.comparable_images:
            print(f"No image found with name {standing_menu}")
            return RestartStateEnum.UNKNOWN_STATE

        if self.match_template(par_screen_image, self.comparable_images[restart_menu]):
            return RestartStateEnum.RESTART_STATE

        if self.match_template(par_screen_image, self.comparable_images[standing_menu]):
            return RestartStateEnum.STANDINGS_STATE

        return RestartStateEnum.UNKNOWN_STATE

    def reset_game_race(self, par_sleep_time_delay: float, par_sleep_time_key_press: float) -> None:
        """
        Reset the game race to the initial state.

        Args:
            par_sleep_time_delay: The amount of time to sleep after each key press.
            par_sleep_time_key_press: The amount of time to press each key.

        Returns:
            None
        """
        time.sleep(par_sleep_time_delay)
        self.a_controls.release_all_keys()

        # First Press To Pause Menu

        tmp_current_state: RestartStateEnum = RestartStateEnum.UNKNOWN_STATE

        while tmp_current_state != RestartStateEnum.RESTART_STATE:
            tmp_current_state = self.is_in_correct_restart_state(self.window_capture()[0])
            if tmp_current_state == RestartStateEnum.UNKNOWN_STATE:
                self.a_controls.press_and_release_key(self.a_controls.ESCAPE,
                                                      par_sleep_time_key_press, True)
                time.sleep(par_sleep_time_delay)
            elif tmp_current_state == RestartStateEnum.STANDINGS_STATE:
                self.a_controls.press_and_release_key(self.a_controls.ENTER,
                                                      par_sleep_time_key_press, True)
                time.sleep(par_sleep_time_delay)

        time.sleep(par_sleep_time_delay)
        # Then We Proceed to The Right - Restart Button
        self.a_controls.press_and_release_key(self.a_controls.RIGHT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        # Press Enter - Restarts The Race
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)

        # Then Prompt Will Appear - We Move To The OK Button
        self.a_controls.press_and_release_key(self.a_controls.LEFT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        # Press Enter - Accepts The Prompt
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)

        time.sleep(3)

    def init_game_race(self, par_sleep_time_delay: float, par_sleep_time_key_press: float):
        """
        Initializes a race in the game by navigating through the game's menu system.

        Args:
            par_sleep_time_delay (float): The delay time between menu selections.
            par_sleep_time_key_press (float): The delay time between key presses.

        Returns:
            None
        """
        par_sleep_time_delay = par_sleep_time_delay * 2
        try:
            self.get_speed_mph()
        except pymem.exception.MemoryReadError:
            print("Race Not Yet Initialised")
        else:
            print("Race Already Initialised")
            return

        # First Press To Go to Menu
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)

        # Accept Prompt
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)
        # Now We are in the Main Menu
        self.a_controls.press_and_release_key(self.a_controls.RIGHT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        # Now the selection will be on Challenge Series
        self.a_controls.press_and_release_key(self.a_controls.RIGHT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        # Now the selection will be on Quick Race

        # Quick Race Menu
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)
        time.sleep(par_sleep_time_delay)
        # Now the selection will be on Custom Race
        self.a_controls.press_and_release_key(self.a_controls.RIGHT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)

        # Custom Race Mode Select
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)
        time.sleep(par_sleep_time_delay)
        # Now the selection will be on Sprint
        self.a_controls.press_and_release_key(self.a_controls.RIGHT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)

        # Sprint Track Select
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)
        time.sleep(par_sleep_time_delay)

        # Sprint Options
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)
        # Traffic Level - None
        self.a_controls.press_and_release_key(self.a_controls.LEFT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        # Set Opponents to None
        self.a_controls.press_and_release_key(self.a_controls.DOWN_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        self.a_controls.press_and_release_key(self.a_controls.LEFT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        self.a_controls.press_and_release_key(self.a_controls.LEFT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        self.a_controls.press_and_release_key(self.a_controls.LEFT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        # Set Difficulty to Easy
        self.a_controls.press_and_release_key(self.a_controls.DOWN_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        self.a_controls.press_and_release_key(self.a_controls.LEFT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        # Catch Up Off
        self.a_controls.press_and_release_key(self.a_controls.DOWN_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        self.a_controls.press_and_release_key(self.a_controls.LEFT_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)

        # Accept
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)
        time.sleep(par_sleep_time_delay)
        # Set Car
        self.a_controls.press_and_release_key(self.a_controls.DOWN_KEY, par_sleep_time_key_press,
                                              True)
        time.sleep(par_sleep_time_delay)
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)
        # Transmission Auto
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
        time.sleep(par_sleep_time_delay)

        # Should Wait One Second to Start the Race
        time.sleep(par_sleep_time_delay * 5)
        # Press Enter To Speed Up The Starting
        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
