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
from typing import Tuple, Optional

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

from car_states.car_state import CarState
from car_states.enabled_game_api_values import EnabledGameApiValues
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
from state.a_game_state import AGameState
from state.game_state_not_connected import GameStateNotConnected
from state.game_state_restarting import GameStateRestarting
from state.game_state_starting import GameStateStarting
from state.game_state_training import GameStateTraining
from strategy.gps.gps_strategy_enum import GPSStrategyEnum
from strategy.gps_image_recognition.a_gps_ircgn_strategy import AGpsImageRecognitionStrategy
from strategy.gps_image_recognition.gps_ircgn_strategy_cpu import GpsImageRecognitionStrategyCPU
from strategy.gps_image_recognition.gps_ircgn_strategy_gpu import GpsImageRecognitionStrategyGPU
from utils.enums.restart_states_enum import RestartStateEnum
from utils.print_utils.printer import Printer
from utils.singleton.controls import Controls


# Pylint disable warnings about too many attributes and methods
# pylint: disable=too-many-instance-attributes,too-many-public-methods
class Game:
    """
    Class Which Acts as an API with the game Need for Speed: Most Wanted (2005)
    """
    api_settings = {
        'game_path': r'F:\Games Folder\Electronic Arts\Need for Speed Most '
                     r'Wanted\speed.exe',
        'cheat_engine_path': r'C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64.exe',
        'game_process_name': 'Need for Speed\u2122 Most Wanted',
        'game_title_name': 'Need for Speed\u2122 Most Wanted',
    }
    a_threshold = 100
    a_ratio = 3

    a_loop_time = time.time()
    a_width: int
    a_height: int
    a_image: ndarray
    a_screenshot: ndarray

    a_game_filename: string = api_settings['game_path']
    a_cheat_engine_filename: string = api_settings['cheat_engine_path']

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

    a_dictionary_menus: dict[str, str]

    a_enabled_game_api_values: EnabledGameApiValues

    a_car_state: CarState

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
        self.a_game_state = GameStateNotConnected()
        self.a_dictionary_menus = {
            'restart_menu': 'resume_race_text',
            'standing_menu': 'standings_menu',
            'attention_restart': 'attention_restart'
        }

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

        self.a_gps_img_rcg_strategy = GpsImageRecognitionStrategyCPU()
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

    def initialize_game(self, par_game_inputs: GameInputs,
                        par_enabled_game_api_values: EnabledGameApiValues) -> None:
        """
        Initializes the game by starting the game, waiting for it to start, creating and 
            initializing required game objects,and setting the game speed and cheat engine.

        Args:
            par_game_inputs (GameInputs): an instance of GameInputs class containing the 
                game inputs.
            par_enabled_game_api_values (EnabledGameApiValues): an instance of EnabledGameApiValues
                class containing the enabled game api values. 

        Returns:
            None
        """
        self.a_game_state = GameStateStarting()
        self.a_enabled_game_api_values = par_enabled_game_api_values
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

        self.a_cheat_engine.start_cheat_engine(self.api_settings['game_process_name'])
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

        self.a_controls.release_all_keys()

        self.a_cycles_passed = 0

        par_game_inputs.game_initialization_inputs.put((
            self.a_race_initialised,
            self.a_speed
        ))

        self.a_car_state = self.create_empty_car_state()

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    def main_loop(self, par_game_inputs: GameInputs,
                  par_results_path: str,
                  par_enabled_game_api_values: EnabledGameApiValues):
        """
        Main Loop That Controls All The Game Logic

        Args:
            par_game_inputs (GameInputs): An instance of the GameInputs class containing the
                inputs for the game.
            par_results_path (str): Path of the folder containing results including graph images
            par_enabled_game_api_values (EnabledGameApiValues): an instance of EnabledGameApiValues
                class containing the enabled game api values. 

        Returns:
            None: This method doesn't return anything.
        """

        self.initialize_game(par_game_inputs, par_enabled_game_api_values)

        tmp_start_time = time.time()
        tmp_speed_constant = 1 / self.a_speed
        tmp_frame_counter: int = 0

        tmp_wrong_way_value: int = -1

        while tmp_wrong_way_value == -1:
            try:
                tmp_wrong_way_value = self.a_wrong_way.return_is_wrong_way()
            # pylint: disable=broad-except
            except Exception as exception:
                Printer.print_info(f"Waiting for pointers to initialize {exception}", "GAME")
                time.sleep(1)

        while True:
            # Capture screenshot and convert to numpy array
            self.a_screenshot, self.a_width, self.a_height = self.window_capture()
            self.a_screenshot = np.array(self.a_screenshot)

            self.a_car_state.reset_car_state()

            # Check for quit key -> !! WARNING - Without this all the windows will be BLANK GREY !!!
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

            # Check for record key
            if keyboard.is_pressed('r'):
                self.a_is_recording = True
                self.a_list_bitmap = []

            grayscale_image, gps_mask, gps_center, gps_size = \
                self.a_gps_img_rcg_strategy.gps_data_with_greyscale(self.a_gps, self.a_screenshot)

            tmp_speed_mph: int = -1
            if par_enabled_game_api_values.enabled_car_speed_mph:
                tmp_speed_mph = self.a_speedometer.return_speed_mph()
            tmp_lap_progress: float = -1
            if par_enabled_game_api_values.enabled_lap_progress:
                tmp_lap_progress = self.a_lap_progress.return_lap_completed_percent()
            tmp_wrong_way_indicator: int = -1
            if par_enabled_game_api_values.enabled_wrong_way_indicator:
                tmp_wrong_way_indicator = self.a_wrong_way.return_is_wrong_way()
            tmp_revolutions_per_minute: float = -1
            if par_enabled_game_api_values.enabled_revolutions_per_minute:
                tmp_revolutions_per_minute = \
                    self.a_revolutions_per_minute.return_revolutions_per_minute()
            tmp_gps_cropped_greyscale: Optional[ndarray] = None
            if par_enabled_game_api_values.enabled_mini_map:
                while tmp_gps_cropped_greyscale is None:
                    tmp_gps_cropped_greyscale = \
                        self.a_gps_img_rcg_strategy.get_half_gps_greyscale(
                            self.a_screenshot,
                            grayscale_image,
                            gps_mask,
                            gps_center,
                            gps_size
                        )
            tmp_car_offset_distance: float = -1
            tmp_car_offset_direction: int = -1
            tmp_contour: Optional[list] = None
            if par_enabled_game_api_values.enabled_distance_incline_center or \
                    par_enabled_game_api_values.enabled_distance_offset_center:
                tmp_car_offset_distance, tmp_contour, tmp_car_offset_direction = \
                    self.a_gps_img_rcg_strategy.calc_car_offset(
                        par_gps=self.a_gps,
                        par_image=self.a_screenshot,
                        par_gps_mask=gps_mask,
                        par_gps_center=gps_center
                    )

            self.a_car_distance_offset = tmp_car_offset_distance
            self.a_car_direction_offset = tmp_car_offset_direction

            if tmp_contour is not None:
                cv2.drawContours(self.a_screenshot, [tmp_contour], -1, (255, 0, 255), -1)

            backup_screenshot: ndarray = self.a_screenshot
            self.show_graph(par_image_path=par_results_path + 'scatter_plot.png')

            tmp_frame_counter += tmp_speed_constant

            if (time.time() - tmp_start_time) > tmp_speed_constant:
                print("FPS: ", tmp_frame_counter / (time.time() - tmp_start_time))
                tmp_frame_counter = 0
                tmp_start_time = time.time()

            self.a_cycles_passed += 1

            while not par_game_inputs.game_restart_inputs.empty():
                tmp_needs_restart: bool = par_game_inputs.game_restart_inputs.get()
                if tmp_needs_restart:
                    self.a_game_state = GameStateRestarting()
                    self.update_state_on_screen(self.a_screenshot)

                    par_game_inputs.game_restart_inputs.put(tmp_needs_restart)
                    self.reset_game_race(0.7 / float(self.a_speed), 0.01 / float(self.a_speed))

                    self.a_game_state = GameStateTraining()
                    par_game_inputs.game_restart_inputs.get()

            # .empty() returns False or True, it is not function to empty the Queue
            # Maybe some python coders can use function declaration, when they know that it
            # returns bool to use naming like isEmpty? :)
            while not par_game_inputs.agent_inputs_state.qsize() == 0:
                try:
                    par_game_inputs.agent_inputs_state.get_nowait()
                except Empty:
                    pass  # This case is possible qsize() is unreliable, it is expected behaviour

            self.a_car_state.assign_values(
                par_speed_mph=tmp_speed_mph,
                par_distance_offset_center=self.get_car_distance_offset(),
                par_incline_center=self.get_car_direction_offset(),
                par_lap_progress=tmp_lap_progress,
                par_wrong_way_indicator=tmp_wrong_way_indicator,
                par_revolutions_per_minute=tmp_revolutions_per_minute,
                par_mini_map=tmp_gps_cropped_greyscale
            )

            par_game_inputs.agent_inputs_state.put(self.a_car_state, )

            self.show_texts_on_image(par_image=backup_screenshot,
                                     par_font_color=(159, 43, 104),
                                     par_car_state=self.a_car_state
                                     )

            self.show_state_on_image(par_image=backup_screenshot,
                                     par_game_state=self.a_game_state)

            cv2.imshow('Main Vision', backup_screenshot)

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

    def update_state_on_screen(self, par_image: ndarray) -> None:
        """
        Updates the game state on the screen.

        Args:
            par_image (ndarray): The image to display the game state on.
    
        Returns:
            None
        """
        self.show_state_on_image(par_image=par_image,
                                 par_game_state=self.a_game_state)

        cv2.imshow('Main Vision', self.a_screenshot)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()

    def show_state_on_image(self, par_image: ndarray,
                            par_game_state: AGameState) -> None:
        """
        Displays the game state on an image.

        Args:
            par_image (ndarray): The image to display the game state on.
            par_game_state (AGameState): The current game state object.

        Returns:
            None
        """
        tmp_new_font_scale: float = self.a_font_settings.font_scale

        test_to_show: str = par_game_state.return_state_text()

        text_size, _ = \
            cv2.getTextSize(
                test_to_show,
                self.a_font_settings.font,
                tmp_new_font_scale,
                self.a_font_settings.thickness
            )

        text_height = text_size[1] + self.a_font_settings.thickness
        text_splitter_height = int(text_height / 2)
        x_bottom_left_text_coordinate: int = int(self.a_width / 2) - int(text_size[0] / 2)
        y_bottom_left_text_coordinate: int = self.a_height - text_height

        bottom_left_corner_of_text: tuple[int, int] = (
            x_bottom_left_text_coordinate,
            y_bottom_left_text_coordinate
        )
        # Draw the text with an outline in white color
        cv2.putText(par_image,
                    test_to_show,
                    bottom_left_corner_of_text,
                    self.a_font_settings.font,
                    tmp_new_font_scale,
                    (255, 255, 255),
                    abs(self.a_font_settings.thickness) + 3,
                    self.a_font_settings.line_type)
        cv2.putText(par_image, test_to_show,
                    bottom_left_corner_of_text,
                    self.a_font_settings.font,
                    tmp_new_font_scale,
                    par_game_state.return_color_representation(),
                    self.a_font_settings.thickness,
                    self.a_font_settings.line_type)
        y_bottom_left_text_coordinate += text_height + text_splitter_height

    def show_texts_on_image(self, par_image: ndarray,
                            par_font_color: tuple[int, int, int],
                            par_car_state: CarState,
                            ) -> None:
        """
        Displays multiple texts on an image with specified font, font size, color, thickness, 
            and line type.

        Args:
            par_image: An ndarray representing the input image.
            par_font_color: The color of the font in RGB format.
            par_car_state: A car state with values to display.

        Returns:
            None
        """

        tmp_new_font_scale: float = self.a_font_settings.font_scale

        tmp_count_of_values_to_display: int = self.a_enabled_game_api_values.count_enabled_values
        if self.a_enabled_game_api_values.enabled_mini_map:
            tmp_count_of_values_to_display -= 1

        if tmp_count_of_values_to_display > 5:
            tmp_new_font_scale -= 0.2 * (tmp_count_of_values_to_display - 5)

        x_bottom_left_text_coordinate: int = 10

        texts_to_show: list[str] = []

        if self.a_enabled_game_api_values.enabled_car_speed_mph:
            texts_to_show.append("Speed: " + str(par_car_state.speed_mph) + " MPH")
        if self.a_enabled_game_api_values.enabled_distance_offset_center:
            texts_to_show.append(
                "Road Offset: " + str(round(par_car_state.distance_offset_center, 2)) + ""
            )
        if self.a_enabled_game_api_values.enabled_lap_progress:
            texts_to_show.append("Completed: " + str(round(par_car_state.lap_progress, 2)) + "%")
        if self.a_enabled_game_api_values.enabled_distance_incline_center:
            texts_to_show.append(
                "Incline: " + str(
                    self.a_gps.translate_direction_offset_to_string(
                        int(par_car_state.incline_center)
                    )
                ) + ""
            )
        if self.a_enabled_game_api_values.enabled_wrong_way_indicator:
            texts_to_show.append("Wrong way: " + str(par_car_state.wrong_way_indicator) + "")
        if self.a_enabled_game_api_values.enabled_revolutions_per_minute:
            texts_to_show.append("RPM: " + str(par_car_state.revolutions_per_minute) + "")

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
        hwnd = win32gui.FindWindow(None, self.api_settings['game_title_name'])

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
        hwnd = win32gui.FindWindow(None, self.api_settings['game_title_name'])
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

        for menu_enum_name, menu_image_string_name in self.a_dictionary_menus.items():
            if menu_image_string_name not in self.a_image_manipulation.comparable_images:
                Printer.print_error(f"No image found with name {menu_image_string_name}", "GAME")
                return RestartStateEnum.UNKNOWN_STATE

            if self.a_image_manipulation.match_template(par_screen_image,
                                                        self.a_image_manipulation.comparable_images[
                                                            menu_image_string_name]):
                if menu_enum_name == 'restart_menu':
                    return RestartStateEnum.RESTART_STATE
                if menu_enum_name == 'standing_menu':
                    return RestartStateEnum.STANDINGS_STATE
                if menu_enum_name == 'attention_restart':
                    return RestartStateEnum.ATTENTION_RESTART_STATE

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
            if tmp_current_state in [RestartStateEnum.UNKNOWN_STATE,
                                     RestartStateEnum.ATTENTION_RESTART_STATE]:
                self.a_controls.press_and_release_key(self.a_controls.ESCAPE,
                                                      par_sleep_time_key_press, True)
                time.sleep(par_sleep_time_delay)
            elif tmp_current_state == RestartStateEnum.STANDINGS_STATE:
                self.a_controls.press_and_release_key(self.a_controls.ENTER,
                                                      par_sleep_time_key_press, True)
                time.sleep(par_sleep_time_delay)

        time.sleep(par_sleep_time_delay)

        tmp_is_not_in_attention_state: bool = True
        while tmp_is_not_in_attention_state:

            keys_to_press: list[int] = [self.a_controls.RIGHT_KEY, self.a_controls.ENTER,
                                        self.a_controls.LEFT_KEY]
            # Then We Proceed to The Right - Restart Button
            # Press Enter - Restarts The Race
            # Then Prompt Will Appear - We Move To The OK Button
            for key_to_press in keys_to_press:
                self.a_controls.press_and_release_key(key_to_press, par_sleep_time_key_press, True)
                time.sleep(par_sleep_time_delay)

            tmp_restart_state = self.is_in_correct_restart_state(self.window_capture()[0])
            if tmp_restart_state == RestartStateEnum.ATTENTION_RESTART_STATE:
                tmp_is_not_in_attention_state = False
            else:
                self.a_controls.press_and_release_key(self.a_controls.ESCAPE,
                                                      par_sleep_time_key_press, True)
                time.sleep(par_sleep_time_delay * 2)
                tmp_is_not_in_attention_state = True

        # If the prompt with Attention (Do you Really Want to Restart) appears press Enter
        time.sleep(par_sleep_time_delay)

        self.a_controls.press_and_release_key(self.a_controls.ENTER, par_sleep_time_key_press, True)
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

    def create_empty_car_state(self) -> CarState:
        """
        Initializes car state with default values
        Returns:
            An CarState object with default values
        """
        return CarState()
