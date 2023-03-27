"""
A module that provides a Printer class for printing colored text to the console.
"""
from utils.print_utils.print_colors_enum import PrintColorsEnum


class Printer:
    """
    A class that provides a method for printing colored text to the console.

    Methods:
        print_utils(par_string_to_print, par_color=PrintColorsEnum.RESET):
            Prints the given string in the specified color to the console.
            If no color is specified, the default color is used.
    """

    @staticmethod
    def print(par_string_to_print: str, par_color: PrintColorsEnum = PrintColorsEnum.RESET) -> None:
        """
        Prints the given string in the specified color to the console.

        Args:
            par_string_to_print: The string to print_utils.
            par_color: The color to print_utils the string in. Defaults to the default color.

        Returns:
            None
        """
        print(f'{par_color.value}{par_string_to_print}{PrintColorsEnum.RESET.value}')

    @staticmethod
    def print_info(par_string_to_print: str, par_caller_as_string: str = "") -> None:
        """
        Prints an informational message in orange to the console.

        Args:
            par_string_to_print: The informational message to print.
            par_caller_as_string: Caller information that should be printed, if specified.

        Returns:
            None
        """
        if par_caller_as_string != "":
            Printer.print(par_caller_as_string + ": " + par_string_to_print, PrintColorsEnum.ORANGE)
        else:
            Printer.print(par_string_to_print, PrintColorsEnum.ORANGE)

    @staticmethod
    def print_error(par_error_message: str, par_caller_as_string: str = "",
                    par_exception: Exception = None) -> None:
        """
        Prints an error message in red-orange to the console.

        Args:
            par_error_message: The informational message to print.
            par_caller_as_string: Optional - Caller information that should be printed, if 
             specified
            par_exception: Optional - Exception to be printed

        Returns:
            None
        """
        error_message = par_error_message

        if par_caller_as_string != "":
            error_message = f"{par_caller_as_string}: {error_message}"

        Printer.print(error_message, PrintColorsEnum.RED_ORANGE)

        if par_exception is not None:
            Printer.print("Exception: " + str(par_exception), PrintColorsEnum.RED_ORANGE)

    @staticmethod
    def print_success(par_success_message: str, par_caller_as_string: str = "") -> None:
        """
        Prints a success message in green to the console.

        Args:
            par_success_message: The success message to print.
            par_caller_as_string: Caller information that should be printed, if specified.

        Returns:
            None
        """
        if par_caller_as_string != "":
            Printer.print(par_caller_as_string + ": " + par_success_message, PrintColorsEnum.GREEN)
        else:
            Printer.print(par_success_message, PrintColorsEnum.GREEN)

    @staticmethod
    def print_basic(par_basic_message: str, par_caller_as_string: str = "") -> None:
        """
        Prints a basic message in white to the console.

        Args:
            par_basic_message: The basic message to print.
            par_caller_as_string: Caller information that should be printed, if specified.

        Returns:
            None
        """
        if par_caller_as_string != "":
            Printer.print(par_caller_as_string + ": " + par_basic_message, PrintColorsEnum.RESET)
        else:
            Printer.print(par_basic_message, PrintColorsEnum.RESET)
