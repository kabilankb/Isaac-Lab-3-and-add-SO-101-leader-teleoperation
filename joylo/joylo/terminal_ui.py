"""Terminal output helpers for JoyLo teleoperation (colors, progress, controls)."""


# Color constants for terminal output
class Colors:
    """ANSI escape codes for colored terminal output."""
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

def print_progress_text(current: int, total: int, prefix: str = "Progress", color: str = Colors.GREEN) -> None:
    """Print a progress line ("current/total (percent%)") in the given terminal color.

    Args:
        current (int): Current progress value.
        total (int): Total progress value.
        prefix (str): Label shown before the progress numbers.
        color (str): ANSI color escape used for the line.
    """
    if total <= 0:
        percent = 0.0
    else:
        percent = min(100.0, (current / total) * 100.0)

    print(f"{color}[INFO] {prefix}: {current}/{total} ({percent:.1f}%){Colors.RESET}")


def print_controls():
    """Print control instructions."""
    print("\n" + "="*80)
    print("JOYLO TELEOPERATION CONTROL")
    print("="*80)
    print("JOYCON CONTROLS:")
    print("• ZL (hold):  Close left gripper continuously")
    print("• ZR (hold):  Close right gripper continuously")
    print("• L (hold):   Open left gripper continuously")
    print("• R (hold):   Open right gripper continuously")
    print("• X (1st press):  Start recording trajectory")
    print("• X (2nd press):  Save trajectory")
    print("• HOME:           Reset task (discard current trajectory)")
    print("• Other buttons:  Print notification (no functionality yet)")
    print()
    print("DATA COLLECTION WORKFLOW:")
    print("1. On startup: Simulation syncs to current JoyLo position")
    print("2. Move JoyLo freely to position arms for the task")
    print("3. Press X to start recording trajectory")
    print("4. Perform demonstration by moving physical JoyLo arms")
    print("5. Press X again to save trajectory")
    print("6. Press HOME to reset task (discards unsaved trajectory)")
    print("7. Repeat until all demos collected")
    print("="*80)
