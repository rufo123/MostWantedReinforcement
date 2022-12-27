import ctypes
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

from PIL import Image, ImageGrab
import pywinauto
import pyautogui
import Utils.direct_input
from Utils.controls import Controls


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



    def __init__(self):
        pass

    def main_loop(self):



        self.a_list_bitmap = []

        self.a_controls = Controls()

        self.start_game()
        # self.start_cheat_engine()

        self.a_speedometer = Speedometer()

        self.a_lap_progress = LapProgress()

        self.a_lap_time = LapTime()

        self.a_gps = GPS()

        tmp_grayscale: None

        self.start_cheat_engine()
        self.set_speed(1)

        self.a_is_recording = False

        tmp_frame_counter: int = 0

        tmp_start_time: float = time.time()

        while True:

            self.a_screenshot, self.a_width, self.a_height, rect = self.window_capture()
            self.a_screenshot = np.array(self.a_screenshot)

            self.a_interest_rect_vert = [
                (0, self.a_height),
                (self.a_width / 2, self.a_height / 2),
                (self.a_width, self.a_height)
            ]

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

            if keyboard.is_pressed('r'):
                self.a_is_recording = True
                self.a_list_bitmap = []

            tmp_grayscale = self.make_grayscale(self.a_screenshot)
            tmp_gps_final = cv2.equalizeHist(tmp_grayscale)
            tmp_gps, tmp_gps_center = self.a_gps.get_gps_mask(tmp_gps_final)
            tmp_gps = cv2.bitwise_and(self.a_screenshot, self.a_screenshot, mask=tmp_gps)

            tmp_gps_hsv = cv2.cvtColor(tmp_gps, cv2.COLOR_BGR2HSV)

            cv2.imshow('Main DKO', tmp_grayscale)

            # tmp_gps_mask_lines = cv2.inRange(tmp_gps_hsv, np.array([0, 0, 87]), np.array([179, 136, 123]))
            tmp_gps_mask_lines = cv2.inRange(tmp_gps_hsv, np.array([0, 0, 174]), np.array([179, 10, 255]))

            tmp_contour = self.a_gps.make_gps_contour(tmp_gps_mask_lines, self.a_screenshot, tmp_gps_center)

            tmp_speed_mph = self.a_speedometer.return_speed_mph()

            tmp_car_pos = self.a_gps.get_car_point(self.a_screenshot, tmp_gps_center)

            tmp_car_offset = self.a_gps.polygon_contour_test(tmp_contour, tmp_car_pos)

            tmp_lap_progress = self.a_lap_progress.return_lap_completed_percent()

            # tmp_lap_time = self.a_lap_time.return_lap_time()

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
            cv2.putText(self.a_screenshot, "Road Offset: " + str(round(tmp_car_offset, 2)) + "",
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

            cv2.imshow('Main Vision', self.a_screenshot)

            tmp_frame_counter += 1

            if (time.time() - tmp_start_time) > 1:
                print("FPS: ", tmp_frame_counter / (time.time() - tmp_start_time))
                tmp_frame_counter = 0
                tmp_start_time = time.time()



    def make_grayscale(self, par_image):
        grayscale = cv2.cvtColor(par_image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(grayscale, (5, 5), 0)

    def make_canny(self, par_grayscale):
        return cv2.Canny(par_grayscale, self.a_threshold, self.a_threshold * self.a_ratio)

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

        if self.a_is_recording:
            self.record(dataBitMap, self.a_is_recording)

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

    def save_image(self, par_rect_window: any, par_filename: str):
        img = ImageGrab.grab(par_rect_window)
        img.save(par_filename + ".png")

    def capture_race(self, par_lap_progress: float, par_game_speed: int, par_rect: any):
        if par_lap_progress is not None:
            self.save_image(par_rect, "record/" + str(par_game_speed) + "/" + str(par_lap_progress))

    def start_cheat_engine(self):
        app = pywinauto.Application().connect(title="Cheat Engine 7.4", found_index=1)

        if app is None:
            app = pywinauto.Application().start('C:\Program Files\Cheat Engine 7.4\cheatengine-x86_64-SSE4-AVX2.exe')

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






