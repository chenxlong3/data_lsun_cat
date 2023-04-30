import os, logging
from datetime import datetime
def init_logs():
    # Log Directory
    cur_time = datetime.now()
    log_dir = f"./logs/{cur_time.month}-{cur_time.day}-{cur_time.hour}-{cur_time.minute}/"
    os.makedirs(log_dir, exist_ok=True)

    log_file = log_dir + "/log_file.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return log_dir, log_file