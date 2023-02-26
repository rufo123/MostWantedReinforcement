import ctypes
import multiprocessing
import os
import pathlib
import string
import subprocess

import time

import cv2
import keyboard
import win32gui
import win32ui
import win32con
import numpy as np
from numpy import ndarray

from GPS import GPS
from LapProgress import LapProgress
from LapTime import LapTime
from Speedometer import Speedometer

from PIL import Image
import pywinauto
import pyautogui

from strategy.gps.gps_strategy_enum import GPSStrategyEnum
from strategy.gps_image_recognition.a_gps_ircgn_strategy import AGpsImageRecognitionStrategy
from strategy.gps_image_recognition.gps_ircgn_strategy_cpu import GpsImageRecognitionStrategyCPU
from strategy.gps_image_recognition.gps_ircgn_strategy_gpu import GpsImageRecognitionStrategyGPU
from Utils.controls import Controls
from cv2 import cuda


class Game:
    a_threshold = 100
    a_ratio = 3

    a_loop_time = time.time()
    a_width: int
    a_height: int
    a_image: ndarray
    a_screenshot: ndarray
    a_interest_rect_vert: list[tuple[float, float]]

    a_game_filename: string = r'F:\Games Folder\Electronic Arts\Need for Speed Most Wanted\speed.exe'
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
        except:
            GpsImageRecognitionStrategyCPU()
            self.a_gps_strategy_enum = GPSStrategyEnum.CPU

    def init_game_memory_objects(self) -> None:
        self.a_speedometer.construct()
        self.a_lap_progress.construct()
        self.a_lap_time.construct()

    def main_loop(self, par_queue_agent_inputs: multiprocessing.Queue,
                  par_queue_game_started_inputs: multiprocessing.Queue,
                  par_queue_restart_game_input: multiprocessing.Queue):

        self.a_speed = 3

        self.a_list_bitmap = []

        self.a_controls = Controls()

        self.start_game()
        # self.start_cheat_engine()

        print(cv2.__version__)

        while not self.process_exists("speed.exe"):
            print("Waiting for Game to Start")
            time.sleep(2)

        self.a_speedometer = Speedometer()

        self.a_lap_progress = LapProgress()

        self.a_lap_time = LapTime()

        self.init_game_memory_objects()

        self.a_gps = GPS(self.a_gps_strategy_enum)

        tmp_grayscale: None

        self.start_cheat_engine()
        self.set_speed(self.a_speed)

        self.a_is_recording = False

        tmp_frame_counter: int = 0

        tmp_start_time: float = time.time()

        tmp_speed_constant = 1 / self.a_speed

        self.focus_on_game()

        time.sleep(5)

        self.init_game_race(0.7 / float(self.a_speed), 0.1 / float(self.a_speed))

        time.sleep(3)

        self.a_race_initialised = True

        self.a_cycles_passed = 0

        par_queue_game_started_inputs.put((self.a_race_initialised, self.a_speed))

        while True:

            # Capture screenshot and convert to numpy array
            self.a_screenshot, self.a_width, self.a_height, rect = self.window_capture()
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
            tmp_car_offset_distance, tmp_contour, tmp_car_offset_direction = self.a_gps_img_rcg_strategy.calc_car_offset(
                self.a_gps, self.a_screenshot)

            self.a_car_distance_offset = tmp_car_offset_distance
            self.a_car_direction_offset = tmp_car_offset_direction

            # tmp_lap_time = self.a_lap_time.return_lap_time()
            if tmp_contour is not None:
                cv2.drawContours(self.a_screenshot, [tmp_contour], -1, (255, 0, 255), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            fontColor = (159, 43, 104)
            thickness = 2
            lineType = 2

            bottomLeftCornerOfText = (10, int(self.a_height / 4))
            cv2.putText(self.a_screenshot, "Speed: " + str(tmp_speed_mph) + " / mph",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            bottomLeftCornerOfText = (10, int(self.a_height / 3))
            cv2.putText(self.a_screenshot, "Road Offset: " + str(round(tmp_car_offset_distance, 2)) + "",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            bottomLeftCornerOfText = (10, int(self.a_height / 2.4))
            cv2.putText(self.a_screenshot, "Completed: " + str(round(tmp_lap_progress, 2)) + "%",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            bottomLeftCornerOfText = (10, int(self.a_height / 2))
            cv2.putText(self.a_screenshot,
                        "Incline: " + str(self.a_gps.translate_direction_offset_to_string(tmp_car_offset_direction))
                        + "",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            cv2.imshow('Main Vision', self.a_screenshot)

            tmp_frame_counter += tmp_speed_constant

            if (time.time() - tmp_start_time) > tmp_speed_constant:
                print("FPS: ", tmp_frame_counter / (time.time() - tmp_start_time))
                tmp_frame_counter = 0
                tmp_start_time = time.time()

            self.a_cycles_passed += 1

            while not par_queue_restart_game_input.empty():
                tmp_needs_restart: bool = par_queue_restart_game_input.get()
                if tmp_needs_restart:
                    par_queue_restart_game_input.put(tmp_needs_restart)
                    self.reset_game_race(0.7 / float(self.a_speed), 0.01 / float(self.a_speed))
                    par_queue_restart_game_input.get()

            # .empty() returns False or True, it is not function to empty the Queue
            # Maybe some python coders can use function declaration, when they know that it
            # returns bool to use naming like isEmpty? :)
            while not par_queue_agent_inputs.empty():
                par_queue_agent_inputs.get()
            par_queue_agent_inputs.put((self.get_speed_mph(), self.get_car_distance_offset(),
                                        self.a_lap_progress.return_lap_completed_percent(),
                                        self.a_car_direction_offset))

    def is_race_initialised(self) -> bool:
        return self.a_race_initialised

    def get_lap_progress(self) -> float:
        return self.a_lap_progress.return_lap_completed_percent()

    def get_car_distance_offset(self) -> float:
        return self.a_car_distance_offset

    def get_car_direction_offset(self) -> int:
        return self.a_car_direction_offset

    def get_speed_mph(self) -> float:
        return self.a_speedometer.return_speed_mph()

    def make_canny(self, par_grayscale):
        # return cv2.Canny(par_grayscale, self.a_threshold, self.a_threshold * self.a_ratio)

        # Convert input to a GpuMat
        gpu_input = cv2.cuda_GpuMat(par_grayscale)

        # Perform Canny edge detection on the GPU
        gpu_output = cv2.cuda.createCannyEdgeDetector(self.a_threshold, self.a_threshold * self.a_ratio)
        gpu_edges = gpu_output.detect(gpu_input)

        # Download the result to CPU memory
        edges = gpu_edges.download()

        return edges

    def region_of_interest(self, par_img, par_vertices):
        mask = np.zeros_like(par_img)
        # color_channel_count = img.shape[2]
        create_matched_color_mask = 255
        # Fills Polygons That Are Not For Our Interest
        cv2.fillPoly(mask, par_vertices, create_matched_color_mask)
        # Return Image Where Only The Mask Pixel Matches
        masked_image = cv2.bitwise_and(par_img, mask)
        return masked_image

    def window_capture(self):
        w = 800  # set this
        h = 600  # set this
        bmpfilenamename = "debug.bmp"  # set this

        hwnd = win32gui.FindWindow(None, 'Need for Speed™ Most Wanted')

        # win32gui.MoveWindow(hwnd, 0, 0, 800, 600, True)

        wDC = win32gui.GetWindowDC(hwnd)

        rect = win32gui.GetWindowRect(hwnd)

        w = rect[2] - rect[0]
        h = rect[3] - rect[1]

        border_pixels = 8
        titlebar_pixels = 30
        w = w - (border_pixels * 2)
        h = h - titlebar_pixels - border_pixels
        x_cropped = border_pixels
        y_cropped = titlebar_pixels

        offset_x = rect[0] + x_cropped
        offset_y = rect[1] + y_cropped

        # Win Image Data
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (w, h), dcObj, (x_cropped, y_cropped), win32con.SRCCOPY)

        # Save Screen
        # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')

        img.shape = (h, w, 4)

        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[..., :3]
        img = np.ascontiguousarray(img)

        if self.a_is_recording:
            self.record(img, self.a_is_recording)

        try:
            win32gui.SetForegroundWindow(hwnd)
        except win32gui.error as e:
            print("An error occurred while setting the foreground window (Probably Debugging): ", e)

        return img, w, h, rect

    def test_file_exec(self, par_file: string) -> bool:
        if not os.path.exists(par_file):
            return False
        if not os.access(par_file, os.X_OK):
            return False
        return True

    def start_cheat_engine(self):
        if self.test_file_exec(self.a_cheat_engine_filename):
            start_cheat_bat = str(pathlib.Path.cwd()) + '\\Utils\\start_cheat_engine.bat'
            print(start_cheat_bat)
            game_path = self.a_cheat_engine_filename
            subprocess.run([start_cheat_bat])
            return
        print('Cheat Engine From File: ' + self.a_cheat_engine_filename + ' cannot be started')
        return

    def start_game(self):
        if self.test_file_exec(self.a_game_filename):
            start_game_bat = str(pathlib.Path.cwd()) + '\\Utils\\start.bat'
            print(start_game_bat)
            game_path = self.a_game_filename
            subprocess.run([start_game_bat])
            return
        print('Game From File: ' + self.a_filename + ' cannot be started')
        return

    def process_exists(self, par_process_name: string) -> bool:
        call = 'TASKLIST', '/FI', 'imagename eq %s' % par_process_name
        # use buildin check_output right away
        output = subprocess.check_output(call).decode()
        # check in last line for process name
        last_line = output.strip().split('\r\n')[-1]
        # because Fail message could be translated
        return last_line.lower().startswith(par_process_name.lower())

    def capture_race(self, par_lap_progress: float, par_game_speed: int, par_rect: any):
        if par_lap_progress is not None:
            self.save_image(par_rect, "record/" + str(par_game_speed) + "/" + str(par_lap_progress))

    def start_cheat_engine(self) -> None:
        app = None

        process_is_running = True
        try:
            app = pywinauto.Application().connect(title="Cheat Engine 7.4", found_index=1)
        except pywinauto.application.ProcessNotFoundError:
            process_is_running = False
        except pywinauto.findwindows.ElementNotFoundError:
            process_is_running = False

        if not process_is_running:
            app = pywinauto.Application().start('C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64-SSE4-AVX2.exe')

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

        index = 0
        index_found = -1
        for string in items:
            if 'Need for Speed™ Most Wanted' in string:
                index_found = index
                break
            index = index + 1

        list_apps.select(index_found - 1)

        file_menu.child_window(title="Open").click()

    def set_speed(self, par_speed: int):
        app = pywinauto.Application().connect(title="Cheat Engine 7.4", found_index=1)

        if app is None:
            app = pywinauto.Application().start('C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64-SSE4-AVX2.exe')

        main_window = app.window(title="Cheat Engine 7.4", found_index=0)
        # Find the control by its name
        control = main_window.child_window(title="Enable Speedhack")
        control.click_input()

        speed_edit = main_window.child_window(class_name="Edit", found_index=0)
        speed_edit.set_edit_text(par_speed)

        speed_edit = main_window.child_window(title="Apply")
        speed_edit.click()

    def get_game_speed(self) -> int:
        app = pywinauto.Application().connect(title="Cheat Engine 7.4", found_index=1)

        if app is None:
            app = pywinauto.Application().start('C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64-SSE4-AVX2.exe')

        main_window = app.window(title="Cheat Engine 7.4", found_index=0)

        speed_edit = main_window.child_window(class_name="Edit", found_index=0)

        return int(speed_edit.window_text())

    def record(self, bitmap, recording):
        if self.a_lap_progress.return_lap_completed_percent() < 2 and recording:
            self.a_list_bitmap.append(bitmap)
            self.a_controls.PressKey(self.a_controls.UP_KEY)
        elif recording:
            self.a_controls.ReleaseKey(self.a_controls.UP_KEY)
            self.save_image()

    def save_image(self):
        for i in range(len(self.a_list_bitmap)):
            img = Image.fromarray(self.a_list_bitmap[i])
            img.save("record/" + str(self.get_game_speed()) + "/" + str(i) + ".png")
        self.a_is_recording = False
        self.a_list_bitmap = []

    def focus_on_game(self) -> None:
        hwnd = win32gui.FindWindow(None, 'Need for Speed™ Most Wanted')
        win32gui.SetForegroundWindow(hwnd)

    def reset_game_race(self, par_sleep_time_delay: float, par_sleep_time_key_press: float):
        # First Press To Pause Menu
        self.a_controls.PressAndReleaseKey(self.a_controls.ESCAPE, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Also We Press The Left Key - To Debug The Behaviour, if the agent action didn't end correctly
        # we will "unstuck" it
        self.a_controls.PressAndReleaseKey(self.a_controls.LEFT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)

        # Then We Proceed to The Right - Restart Button
        self.a_controls.PressAndReleaseKey(self.a_controls.RIGHT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Press Enter - Restarts The Race
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)

        # Then Prompt Will Appear - We Move To The OK Button
        self.a_controls.PressAndReleaseKey(self.a_controls.LEFT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Press Enter - Accepts The Prompt
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)

        time.sleep(3)

    def init_game_race(self, par_sleep_time_delay: float, par_sleep_time_key_press: float):

        try:
            self.get_speed_mph()
        except Exception as e:
            print("Race Not Yet Initialised")
        else:
            print("Race Already Initialised")
            return

        # First Press To Go to Menu
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)

        # Accept Prompt
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Now We are in the Main Menu
        self.a_controls.PressAndReleaseKey(self.a_controls.RIGHT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Now the selection will be on Challenge Series
        self.a_controls.PressAndReleaseKey(self.a_controls.RIGHT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Now the selection will be on Quick Race

        # Quick Race Menu
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        time.sleep(par_sleep_time_delay)
        # Now the selection will be on Custom Race
        self.a_controls.PressAndReleaseKey(self.a_controls.RIGHT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)

        # Custom Race Mode Select
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        time.sleep(par_sleep_time_delay)
        # Now the selection will be on Sprint
        self.a_controls.PressAndReleaseKey(self.a_controls.RIGHT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)

        # Sprint Track Select
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        time.sleep(par_sleep_time_delay)

        # Sprint Options
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Traffic Level - None
        self.a_controls.PressAndReleaseKey(self.a_controls.LEFT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Set Opponents to None
        self.a_controls.PressAndReleaseKey(self.a_controls.DOWN_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        self.a_controls.PressAndReleaseKey(self.a_controls.LEFT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        self.a_controls.PressAndReleaseKey(self.a_controls.LEFT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        self.a_controls.PressAndReleaseKey(self.a_controls.LEFT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Set Difficulty to Easy
        self.a_controls.PressAndReleaseKey(self.a_controls.DOWN_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        self.a_controls.PressAndReleaseKey(self.a_controls.LEFT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Catch Up Off
        self.a_controls.PressAndReleaseKey(self.a_controls.DOWN_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        self.a_controls.PressAndReleaseKey(self.a_controls.LEFT_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)

        # Accept
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        time.sleep(par_sleep_time_delay)
        # Set Car
        self.a_controls.PressAndReleaseKey(self.a_controls.DOWN_KEY, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)
        # Transmission Auto
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
        time.sleep(par_sleep_time_delay)

        # Should Wait One Second to Start the Race
        time.sleep(par_sleep_time_delay * 5)
        # Press Enter To Speed Up The Starting
        self.a_controls.PressAndReleaseKey(self.a_controls.ENTER, par_sleep_time_key_press)
