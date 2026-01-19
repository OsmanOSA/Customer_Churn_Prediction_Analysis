import sys

from src.logging import logger


class ChurnErrorException(Exception):

    def __init__(self, 
                 error_message: str, 
                 error_details: sys):
        
        self.error_message = error_message
        _, _, exc_table = exec.info()

        self.lineno = exc_table.tb_lineno
        self.filename = exc_table.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.filename, self.lineno, str(self.error_message)
        )