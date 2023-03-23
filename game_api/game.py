"""
This module provides a Class Which Acts as an API with the game Need for Speed: Most Wanted (2005)
"""
import ctypes
import os
import pathlib
import string
import subprocess
import time
# Disable the pylint error for the next line
# pylint: disable=no-name-in-module
from queue import Empty
from typing import Tuple

import cv2
import keyboard
import numpy as np
import pymem
import win32con
import win32gui
import win32ui
from PIL import Image
# noinspection PyProtectedMember
from cv2 import cuda
from numpy import ndarray

from game_api.cheat_engine import CheatEngine
from game_api.font_settings import FontSettings
from game_api.image_manipulation import ImageManipulation
from game_inputs import GameInputs
from game_memory_reading.lap_progress import LapProgress
from game_memory_reading.lap_time import LapTime
from game_memory_reading.revolutions_per_minute import RevolutionsPerMinute
from game_memory_reading.speedometer import Speedometer
from game_memory_reading.wrong_way import WrongWay
from gps import GPS
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

    a_game_filename: string = r'F:\Games Folder\Electronic Arts\Need for Speed Most ' \
                              r'Wanted\speed.exe'
    a_cheat_engine_filename: string = r'C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64.exe'

    a_gps: GPS

    a_speedometer: Speedometer

    a_lap_progress: LapProgress

    a_lap_time: LapTime

    a_revolutions_per_minute: RevolutionsPerMinute

    a_wrong_way: WrongWay

    a_is_recording: bool

    a_user23 = ctypes.windll.user32

    a_list_bitmap: []

    a_controls: Controls

    a_font_settings: FontSettings

    a_speed: int

    a_car_distance_offset: float

    a_car_direction_offset: int

    a_race_initialised: bool

    a_cycles_passed: int

    a_cuda_device: None

    a_gps_img_rcg_strategy: AGpsImageRecognitionStrategy

    a_gps_strategy_enum: GPSStrategyEnum

    a_cheat_engine: CheatEngine

    a_image_manipulation: ImageManipulation

    def __init__(self) -> None:
        self.a_image_manipulation = ImageManipulation()
        self.a_image_manipulation.load_comparable_images()
        self.a_race_initialised = False
        self.a_car_distance_offset = 0
        self.a_car_direction_offset = 0
        self.a_cycles_passed = 0
        self.a_cheat_engine = CheatEngine()
        self.a_list_bitmap = []
        self.a_speed = 3
        self.a_controls = Controls()
        self.a_font_settings = FontSettings(
            par_font=cv2.FONT_HERSHEY_SIMPLEX,
            par_font_scale=1,
            par_font_thickness=2,
            par_line_type=2
        )

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

        self.a_gps = GPS(self.a_gps_strategy_enum)

    def init_game_memory_objects(self) -> None:
        """
        Initializes Speedometer, Lap Progress and Lap Time From Game Memory Reading
        """
        self.a_speedometer.construct()
        self.a_lap_progress.construct()
        self.a_lap_time.construct()
        self.a_revolutions_per_minute.construct()
        self.a_wrong_way.construct()

    def initialize_game(self, par_game_inputs: GameInputs) -> None:
        """
        Initializes the game by starting the game, waiting for it to start, creating and 
            initializing required game objects,and setting the game speed and cheat engine.

        Args:
            par_game_inputs (GameInputs): an instance of GameInputs class containing the 
                game inputs.

        Returns:
            None
        """
        self.start_game()
        # self.start_cheat_engine()

        print(cv2.__version__)

        while not self.process_exists("speed.exe"):
            print("Waiting for Game to Start")
            time.sleep(3)

        self.a_speedometer = Speedometer()

        self.a_lap_progress = LapProgress()

        self.a_lap_time = LapTime()

        self.a_revolutions_per_minute = RevolutionsPerMinute()

        self.a_wrong_way = WrongWay()

        self.a_cheat_engine.start_cheat_engine()
        self.a_cheat_engine.set_speed(self.a_speed)

        self.a_is_recording = False

        self.focus_on_game()

        time.sleep(5)

        self.init_game_memory_objects()

        self.a_cheat_engine.reconfigure_speed(4)

        self.init_game_race(0.7 / float(4), 0.1 / float(4))

        self.a_cheat_engine.reconfigure_speed(self.a_speed)

        time.sleep(3)

        self.a_race_initialised = True

        self.a_cycles_passed = 0

        par_game_inputs.game_initialization_inputs.put((self.a_race_initialised, self.a_speed))

    def main_loop(self, par_game_inputs: GameInputs):
        """
        Main Loop That Controls All The Game Logic

        Args:
            par_game_inputs (GameInputs): An instance of the GameInputs class containing the
                inputs for the game.

        Returns:
            None: This method doesn't return anything.
        """

        self.initialize_game(par_game_inputs)

        tmp_start_time = time.time()
        tmp_speed_constant = 1 / self.a_speed
        tmp_frame_counter: int = 0

        while True:
            # Capture screenshot and convert to numpy array
            self.a_screenshot, self.a_width, self.a_height = self.window_capture()
            self.a_screenshot = np.array(self.a_screenshot)

            # Check for quit key -> !! WARNING - Without this all the windows will be BLANK GREY !!!
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

            # Check for record key
            if keyboard.is_pressed('r'):
                self.a_is_recording = True
                self.a_list_bitmap = []

            tmp_speed_mph: int = self.a_speedometer.return_speed_mph()
            tmp_lap_progress: float = self.a_lap_progress.return_lap_completed_percent()
            tmp_revolutions_per_minute: float = \
                self.a_revolutions_per_minute.return_revolutions_per_minute()
            tmp_is_wrong_way: bool = self.a_wrong_way.return_is_wrong_way()

            # tmp_car_offset, tmp_contour = self.calc_car_offset(self.a_screenshot)
            tmp_car_offset_distance: float
            tmp_contour: list
            tmp_car_offset_direction: int
            tmp_car_offset_distance, tmp_contour, tmp_car_offset_direction = \
                self.a_gps_img_rcg_strategy.calc_car_offset(self.a_gps, self.a_screenshot)

            self.a_car_distance_offset = tmp_car_offset_distance
            self.a_car_direction_offset = tmp_car_offset_direction

            if tmp_contour is not None:
                cv2.drawContours(self.a_screenshot, [tmp_contour], -1, (255, 0, 255), -1)

            strings_to_show: ndarray = np.array([
                str(tmp_speed_mph),
                str(round(tmp_car_offset_distance, 2)),
                str(round(tmp_lap_progress, 2)),
                str(self.a_gps.translate_direction_offset_to_string(tmp_car_offset_direction)),
                str(round(tmp_revolutions_per_minute, 2)),
                str(tmp_is_wrong_way)
            ])

            self.show_texts_on_image(par_image=self.a_screenshot,
                                     par_font_color=(159, 43, 104),
                                     par_array_of_text=strings_to_show
                                     )

            cv2.imshow('Main Vision', self.a_screenshot)

            self.show_graph(par_image_path=
                            'h:/diplomka_vysledky/results/short_race/fourth_iteration_training' \
                            '/scatter_plot.png')

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
            while not par_game_inputs.agent_inputs_state.qsize() == 0:
                try:
                    par_game_inputs.agent_inputs_state.get_nowait()
                except Empty:
                    pass  # This case is possible qsize() is unreliable, it is expected behaviour
            par_game_inputs.agent_inputs_state.put(
                (self.get_speed_mph(), self.get_car_distance_offset(),
                 self.a_lap_progress.return_lap_completed_percent(),
                 self.a_car_direction_offset, self.get_revolutions_per_minute(),
                 self.get_is_wrong_way()))

    def is_race_initialised(self) -> bool:
        """
        Checks is Race is Already Initialised
        :return: Return True is Race is Initialised otherwise False
        """
        return self.a_race_initialised

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

    def get_revolutions_per_minute(self) -> float:
        """
        Get Car Revolutions per Minute (RPM)
        :return: Float value describing Car Revolutions per Minute (RPM)
        """
        return self.a_revolutions_per_minute.return_revolutions_per_minute()

    def get_is_wrong_way(self) -> bool:
        """
        Gets Car Wrong Way
        :return: bool value describing if the car is going in a wrong way
        """
        return self.a_wrong_way.return_is_wrong_way()

    def show_graph(self, par_image_path: str) -> None:
        """
        Displays an image of a graph on the screen.

        Args:
            par_image_path: A string representing the path to the image file.
        Returns:
            None.
        """
        image = None
        if os.path.exists(os.path.abspath(par_image_path)):
            # Load the image
            image = cv2.imread(os.path.abspath(par_image_path))
        if image is not None:
            cv2.imshow('Graph: ', image)

    def show_texts_on_image(self, par_image: ndarray,
                            par_font_color: tuple[int, int, int],
                            par_array_of_text: np.ndarray
                            ) -> None:
        """
        Displays multiple texts on an image with specified font, font size, color, thickness, 
            and line type.

        Args:
            par_image: An ndarray representing the input image.
            par_font_color: The color of the font in RGB format.
            par_array_of_text: A numpy ndarray containing the text to be displayed.

        Returns:
            None
        """

        tmp_new_font_scale: float = self.a_font_settings.font_scale

        if par_array_of_text.size > 5:
            tmp_new_font_scale -= 0.2 * (par_array_of_text.size - 5)

        x_bottom_left_text_coordinate: int = 10

        texts_to_show: list[str] = [
            "Speed: " + par_array_of_text[0] + " MPH",
            "Road Offset: " + par_array_of_text[1] + "",
            "Completed: " + par_array_of_text[2] + "%",
            "Incline: " + par_array_of_text[3] + "",
            "RPM: " + par_array_of_text[4] + "",
            "Wrong Way: " + par_array_of_text[5] + ""
        ]

        text_size, _ = \
            cv2.getTextSize(
                "Text",
                self.a_font_settings.font,
                tmp_new_font_scale,
                self.a_font_settings.thickness
            )
        text_height = text_size[1] + self.a_font_settings.thickness
        text_splitter_height = int(text_height / 2)
        y_bottom_left_text_coordinate: int = int(self.a_height / 3.5)

        for text_value in texts_to_show:
            bottom_left_corner_of_text: tuple[int, int] = (
                x_bottom_left_text_coordinate,
                y_bottom_left_text_coordinate
            )
            # Draw the text with an outline in white color
            cv2.putText(par_image,
                        text_value,
                        bottom_left_corner_of_text,
                        self.a_font_settings.font,
                        tmp_new_font_scale,
                        (255, 255, 255),
                        abs(self.a_font_settings.thickness) + 3,
                        self.a_font_settings.line_type)
            cv2.putText(par_image, text_value,
                        bottom_left_corner_of_text,
                        self.a_font_settings.font,
                        tmp_new_font_scale,
                        par_font_color,
                        self.a_font_settings.thickness,
                        self.a_font_settings.line_type)
            y_bottom_left_text_coordinate += text_height + text_splitter_height

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

        width, height, x_cropped, y_cropped = \
            self.__calculate_cropped_dimensions(
                par_rect=rect, par_border_pixels=8, par_tile_bar_pixels=30
            )

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

    def __calculate_cropped_dimensions(self, par_rect: Tuple[int, int, int, int],
                                       par_border_pixels: int = 8,
                                       par_tile_bar_pixels: int = 30) -> Tuple[int, int, int, int]:
        """
        Calculates the dimensions of a cropped rectangle given the dimensions of an original 
            rectangle and the size of its border and title bar.

        Args:
            par_rect: A tuple of four integers representing the coordinates of the top-left and 
                bottom-right corners of the original rectangle in the form (x1, y1, x2, y2).
            par_border_pixels: An integer representing the width of the border around the original 
                rectangle in pixels (default 8).
            par_tile_bar_pixels: An integer representing the height of the title bar at the top of 
                the original rectangle in pixels (default 30).

        Returns:
            A tuple of four integers representing the width, height, x_cropped and y_cropped

        """
        width = (par_rect[2] - par_rect[0]) - (par_border_pixels * 2)
        height = (par_rect[3] - par_rect[1]) - par_tile_bar_pixels - par_border_pixels
        x_cropped = par_border_pixels
        y_cropped = par_tile_bar_pixels
        return width, height, x_cropped, y_cropped

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
            img.save("record/" + str(self.a_cheat_engine.get_game_speed()) + "/" + str(i) + ".png")
        self.a_is_recording = False
        self.a_list_bitmap = []

    def focus_on_game(self) -> None:
        """
        Forces Windows to Focus On Game (Bring It To Front)
        """
        hwnd = win32gui.FindWindow(None, 'Need for Speed\u2122 Most Wanted')
        win32gui.SetForegroundWindow(hwnd)

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
        if restart_menu not in self.a_image_manipulation.comparable_images:
            print(f"No image found with name {restart_menu}")
            return RestartStateEnum.UNKNOWN_STATE
        if standing_menu not in self.a_image_manipulation.comparable_images:
            print(f"No image found with name {standing_menu}")
            return RestartStateEnum.UNKNOWN_STATE

        if self.a_image_manipulation.match_template(par_screen_image,
                                                    self.a_image_manipulation.comparable_images[
                                                        restart_menu]):
            return RestartStateEnum.RESTART_STATE

        if self.a_image_manipulation.match_template(par_screen_image,
                                                    self.a_image_manipulation.comparable_images[
                                                        standing_menu]):
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

        keys_to_press: list[int] = [self.a_controls.RIGHT_KEY, self.a_controls.ENTER,
                                    self.a_controls.LEFT_KEY, self.a_controls.ENTER]
        # Then We Proceed to The Right - Restart Button
        # Press Enter - Restarts The Race
        # Then Prompt Will Appear - We Move To The OK Button
        # Press Enter - Accepts The Prompt
        for key_to_press in keys_to_press:
            self.a_controls.press_and_release_key(key_to_press, par_sleep_time_key_press, True)
            time.sleep(par_sleep_time_delay)

        time.sleep(1 * self.a_speed)

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

        keys_to_press: list[int] = [
            self.a_controls.ENTER,  # First Press To Go to Menu
            self.a_controls.ENTER,  # Accept Prompt to be int the Main Menu
            self.a_controls.RIGHT_KEY,  # Now the selection will be on Challenge Series
            self.a_controls.RIGHT_KEY,  # Now the selection will be on Quick Race
            self.a_controls.ENTER,  # Quick Race Menu
            self.a_controls.RIGHT_KEY,  # Now the selection will be on Custom Race
            self.a_controls.ENTER,  # Custom Race Mode Select
            self.a_controls.RIGHT_KEY,  # Now the selection will be on Sprint
            self.a_controls.ENTER,  # Sprint Track Select
            self.a_controls.ENTER,  # Sprint Options
            self.a_controls.LEFT_KEY,  # Traffic Level - None
            self.a_controls.DOWN_KEY,  # Set Opponents Option
            self.a_controls.LEFT_KEY,
            self.a_controls.LEFT_KEY,
            self.a_controls.LEFT_KEY,  # Set Opponents to None
            self.a_controls.DOWN_KEY,  # Set Difficulty Option
            self.a_controls.LEFT_KEY,  # Set Difficulty to Easy
            self.a_controls.DOWN_KEY,  # Catch Up Option
            self.a_controls.LEFT_KEY,  # Catch Up Off
            self.a_controls.ENTER,  # Accept Settings
            self.a_controls.DOWN_KEY,  # Set Car
            self.a_controls.ENTER,  # Accept Car Selection
            self.a_controls.ENTER,  # Transmission Auto
            self.a_controls.ENTER,  # Press Enter To Speed Up The Starting
        ]

        for index, key_to_press in enumerate(keys_to_press):
            self.a_controls.press_and_release_key(key_to_press, par_sleep_time_key_press,
                                                  True)
            time.sleep(par_sleep_time_delay)
            if index in [4, 6, 8, 19]:
                time.sleep(par_sleep_time_delay)
            if index == 22:
                # Should Wait to Start the Race
                time.sleep(par_sleep_time_delay * 5)
