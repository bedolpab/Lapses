import time


def time_stamp(msg: str, time):
    print(f'{msg} - {time}')


def get_time() -> time.ctime:
    return time.ctime(time.time())
