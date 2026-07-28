import logging
import datetime
import os
import pytz

class Logger:
    """
    Wrapper for logging
    """
    def __init__(self, log: logging.Logger, rank: int):
        self.log = log
        self.set_rank(rank)
    
    def set_rank(self, rank):
        self.rank = rank

    def info(self, msg: str, all = False):
        if all:
            self.log.info(msg)
        elif self.rank == 0:
            self.log.info(msg)

    def debug(self, msg: str):
        if self.rank == 0:
            self.log.debug(msg)

    def warning(self, msg: str):
        self.log.warning(f"rank {self.rank} - {msg}")

    def error(self, msg: str):
        self.log.error(f"rank {self.rank} - {msg}")

    def critical(self, msg: str):
        self.log.critical(f"rank {self.rank} - {msg}")


def setup_logger(log_dir: str, rank: int, out=True):

    tz = pytz.timezone("America/New_York")
    logging.Formatter.converter = (
        lambda *args: datetime.datetime.now(tz).timetuple()
    )

    handlers = [logging.StreamHandler()]

    if out:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.makedirs(os.path.join(log_dir, "logs"), exist_ok=True)
        handlers.append(logging.FileHandler(f"{log_dir}/logs/{now}.log"))

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    logger = logging.getLogger()
    logger.info("logger initialized")
    return Logger(logger, rank)