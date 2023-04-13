# use pyatuogui to click and drag mouse from one point to another
import pyautogui
import time

def pan():
    # wait for 3 seconds
    time.sleep(3)

    # move the mouse to the bottom right of screen
    pyautogui.moveTo(1900, 1078, duration=.25)

    # drag the mouse left to pan
    pyautogui.dragTo(200, 1078, button='left', duration=2)
    pyautogui.moveTo(1900, 1078, duration=0)
    pyautogui.dragTo(200, 1078, button='left', duration=2)
    pyautogui.moveTo(1900, 1078, duration=0)
    pyautogui.dragTo(200, 1078, button='left', duration=2)

def calibrate():
    # get the screen size
    width, height = pyautogui.size()

    print(width, height)

    # print the current mouse position
    while True: print(pyautogui.position())


if __name__ == "__main__":
    # calibrate()
    pan()