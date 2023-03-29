""" Utility functions
"""
import time
import settings
import os

def vprnt(*args):
    """ Thin wrapper around the standard print function
    Only prints if settings.verbose == True
    """
    if settings.verbose:
        print(*args)

def tic():
    """ Start timing. Returns a list of times with one entry.
    I wrote these tic & toc functions for NMLAB
    See https://github.com/jonkb/NMLab/blob/main/src/util.py
    """
    times = []
    times.append(time.time())
    return times

def toc(times, msg=None, total=False):
    """ Log the current time in the times list.
    If msg is provided, print out a message with the time.
      the string f" time: {t}" will be appended to the message.
    If total, then print out the elapsed time since the start.
      Else, print out only the last time interval.
    """
    times.append(time.time())
    if msg is not None:
        t = times[-1] - times[0] if total else times[-1] - times[-2]
        vprnt(f"{msg} time: {t:.6f} s")

def file_cleanup(types, dirpath=None):
    """ Remove all files of the given types from a directory.
    Default directory: os.getcwd()
    """
    if dirpath is None:
        dirpath = os.getcwd()
    for filename in os.listdir(dirpath):
        if os.path.isfile(os.path.join(dirpath, filename)):
            if any([filename.endswith('.' + ext) for ext in types]):
                os.remove(os.path.join(dirpath, filename))
