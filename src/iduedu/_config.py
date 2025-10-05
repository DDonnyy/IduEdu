import sys
from typing import Literal

from loguru import logger


class Config:
    """
    A configuration class to manage global settings for the application, such as Overpass API URL, timeouts, and logging options.

    Methods
    -------
    change_logger_lvl(lvl: Literal["TRACE", "DEBUG", "INFO", "WARN", "ERROR"])
        Changes the logging level to the specified value.
    set_overpass_url(url: str)
        Sets a new Overpass API URL.
    set_timeout(timeout: int)
        Sets the timeout for API requests.
    set_enable_tqdm(enable: bool)
        Enables or disables progress bars in the application.
    """

    def __init__(self):
        self.overpass_url = "http://lz4.overpass-api.de/api/interpreter"
        self.timeout = 120
        self.enable_tqdm_bar = True
        self.logger = logger
        self.drive_useful_edges_attr = {"highway", "name", "lanes"}
        self.walk_useful_edges_attr = {"highway", "name"}
        self.transport_useful_edges_attr = {"name"}
        self.overpass_min_interval = 1

    def set_logger_lvl(self, lvl: Literal["TRACE", "DEBUG", "INFO", "WARN", "ERROR"]):
        self.logger.remove()
        self.logger.add(sys.stderr, level=lvl)

    def set_overpass_url(self, url: str):
        # TODO ping new url
        self.overpass_url = url

    def set_timeout(self, timeout: int):
        self.timeout = timeout

    def set_enable_tqdm(self, enable: bool):
        self.enable_tqdm_bar = enable

    def set_drive_useful_edges_attr(self, attr: set):
        self.drive_useful_edges_attr = set(attr)

    def set_walk_useful_edges_attr(self, attr: set):
        self.walk_useful_edges_attr = set(attr)


config = Config()
config.set_logger_lvl("INFO")
