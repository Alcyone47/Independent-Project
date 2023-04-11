import pygaze
from pygaze.display import Display
from pygaze.screen import Screen
from pygaze.eyetracker import EyeTracker

# Create a display
disp = Display(disptype='psychopy')

# Create a screen
scr = Screen()

# Create an eye tracker
tracker = EyeTracker(disp)

# Calibrate the eye tracker
tracker.calibrate()

# Start tracking the eyes
tracker.start_recording()

# Get the eye position
x, y = tracker.sample()

# Stop tracking the eyes
tracker.stop_recording()

# Close the eye tracker and display
tracker.close()
disp.close()
