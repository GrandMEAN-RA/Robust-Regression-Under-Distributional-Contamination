
# Import sys module
import sys
from blessed import Terminal


# Save current cursor position
_current_cursor_location = "\033[s"














# Text styling
reset = "\033[0m"
bold = "\033[1m"
green = "\033[32m"
cyan = "\033[36m"
yellow = "\033[33m"
red = "\033[31m"
magenta = "\033[32m"

def get_current_position():

    term = Terminal()
    # Captures the exact absolute cursor location safely inside VS Code
    with term.location():
        anchor_pos = term.get_location()

    return anchor_pos

# Marks the current terminal line as the upper boundary
# for dynamic updates.
def set_anchor():
    sys.stdout.write(_current_cursor_location)
    sys.stdout.flush()

# Wipes out all text below the marked anchor and resets the
# cursor to it.
def refresh_terminal():

    _restore_cursor_and_clear = "\033[u\033[j" # Restore cursor position and clear downward
    sys.stdout.write(_restore_cursor_and_clear)
    sys.stdout.flush()
def refresh_line():

    _restore_cursor_and_clear = "\033[u\033[k" # Restore cursor position and clear forward
    sys.stdout.write(_restore_cursor_and_clear)
    sys.stdout.flush()

def restore():
    _restore_cursor = "\033[u" # Restore cursor to last saved position
    sys.stdout.write(_restore_cursor)
    sys.stdout.flush()

def status(info: str):
    """Prints a status message at the bottom of the terminal."""
    sys.stdout.write(f"{green}Status: {info}{reset}\n")
    sys.stdout.flush()

def clear_below():
    """Clears everything from the cursor down to the bottom of the terminal."""
    _clrdown = '\033[J' # Clear console from current cursor position downward
    sys.stdout.write(_clrdown)
    sys.stdout.flush()

def clear_line():
    """Clears everything from the cursor down to the bottom of the terminal."""
    _clrline = '\033[K' # Clear line from cursor to end of line
    sys.stdout.write(_clrline)
    sys.stdout.flush()

def move_home():
    """Clears everything from the cursor down to the bottom of the terminal."""
    _home = '\033[H' #Move cursor to Home (top-left corner, 0,0)
    sys.stdout.write(_home)
    sys.stdout.flush()

def move_up(lines):
    """Moves the cursor up a specific number of lines."""
    _up = f'\033[{lines}A' #Move cursor Up 1 line
    sys.stdout.write(_up)
    sys.stdout.flush()

def move_down(lines):
    """Moves the cursor up a specific number of lines."""
    _down = f'\033[{lines}B' # Move cursor Down 1 line
    sys.stdout.write(_down)
    sys.stdout.flush()

def move_right(lines):
    """Moves the cursor up a specific number of lines."""
    _right = f'\033[{lines}C' # Move cursor Forward/Right 1 column
    sys.stdout.write(_right)
    sys.stdout.flush()

def move_left(lines):
    """Moves the cursor up a specific number of lines."""
    _left = f'\033[{lines}D' # Move cursor Backward/Left 1 column
    sys.stdout.write(_left)
    sys.stdout.flush()

def move_xypoint(lines, columns):
    """ """
    _xypoint = f'\033[{lines};{columns}H' #Move cursor to Exact Position
    sys.stdout.write(_xypoint)
    sys.stdout.flush()

    