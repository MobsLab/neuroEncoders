#!/usr/bin/env python3

import functools
import time


def timing(func):
    """
    A decorator that prints the time taken by a function to execute.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"⏱ Starting '{func.__name__}'...")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"✅ Finished '{func.__name__}' in {end - start:.2f} seconds")
        return result

    return wrapper
