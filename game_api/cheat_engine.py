"""
Module provides functionality to interact with the Cheat Engine application.
"""
import pywinauto
import pyautogui


class CheatEngine:
    """
    A class that provides functionality to interact with the Cheat Engine application.
    """

    a_cheat_engine_filename: str

    def __init__(self):
        """
        Initialize a new instance of the CheatEngine class.
        """

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
        
        # Get the main window of the application

        main_window = app.window(title="Cheat Engine 7.4", found_index=0)

        app_rect = main_window.rectangle()
        app_position_process = (app_rect.left + (app_rect.right - app_rect.left) * 1 / 40,
                                app_rect.top + (app_rect.bottom - app_rect.top) * 1 / 10)
        pyautogui.click(app_position_process[0], app_position_process[1])

        file_menu = app.window(title="Process List")

        list_apps = file_menu.children()[0]

        items = list_apps.texts()

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
