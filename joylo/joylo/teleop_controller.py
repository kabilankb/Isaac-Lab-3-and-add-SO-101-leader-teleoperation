"""Controller mapping JoyLo hardware input to the IsaacLab follower simulation."""

import time

import numpy as np

from joylo.agents import YAMGelloAgent, JoyConAgent
from joylo.follower_client import BimanualFollowerClient
from joylo.terminal_ui import Colors, print_progress_text


class JoyLoTeleopController:
    """
    Controls teleoperation from JoyLo hardware to IsaacLab simulation.
    Handles button presses for recording, reset, and diagnostics.
    """

    def __init__(
        self,
        arm_agent: YAMGelloAgent,
        joycon_agent: JoyConAgent,
        follower: BimanualFollowerClient,
        enable_recording: bool = False,
        control_frequency: float = 30.0,  # Hz
    ):
        """Initialize the JoyLo teleoperation controller.

        Args:
            arm_agent (YAMGelloAgent): Arm-control agent (reads Dynamixel motors).
            joycon_agent (JoyConAgent): JoyCon agent (reads buttons and grippers).
            follower (BimanualFollowerClient): RPC client for the sim follower.
            enable_recording (bool): Whether to enable data recording.
            control_frequency (float): Control loop frequency in Hz.
        """
        self.arm_agent = arm_agent
        self.joycon_agent = joycon_agent
        self.follower = follower
        self.enable_recording = enable_recording
        self.control_dt = 1.0 / control_frequency

        # Recording state
        self.recording_active = False
        self.x_button_action = "start"  # "start" or "save"

        # Joint state (always unlocked - free movement at all times)
        self.sync_required = True  # Flag to indicate sim needs to sync with real

        # Button state tracking (for edge detection)
        self.prev_button_states = {}

        # Exit flag
        self.exit_requested = False
        # Stage success polling (to mirror sim messages into JoyLo terminal)
        self._stage1_success_prev = None
        self._stage2_success_prev = None
        self._last_stage_poll_time = 0.0
        self._stage_poll_interval = 0.25  # seconds

        print(f"[INFO] JoyLoTeleopController initialized:")
        print(f"[INFO] - Control frequency: {control_frequency} Hz")
        print(f"[INFO] - Recording enabled: {enable_recording}")

    def run_teleoperation_step(self):
        """Execute one step of teleoperation - always send JoyLo positions to sim."""
        # Handle initial sync if needed
        if self.sync_required:
            print(f"[INFO] Syncing simulation to current JoyLo position...")
            print(f"[INFO] Physical hardware will NOT move - only simulation syncs")
            self.sync_required = False
            print(f"[INFO] Sync complete. You can freely move JoyLo arms to control simulation")
            print(f"{'='*60}")
            print(f"{Colors.YELLOW}[INFO] Press X to start recording when ready{Colors.RESET}")
            return

        # Read JoyLo state and send to simulation (free movement at all times)
        arm_joints = self.arm_agent.get_joints()  # 12 DOF
        left_gripper, right_gripper = self.joycon_agent.get_gripper_commands()

        # Split arm joints for left and right arms
        left_arm_joints = arm_joints[:6]  # First 6 joints
        right_arm_joints = arm_joints[6:12]  # Next 6 joints

        # Combine arm joints + gripper for each arm (7 DOF total per arm)
        left_command = np.concatenate([left_arm_joints, [left_gripper]])
        right_command = np.concatenate([right_arm_joints, [right_gripper]])

        # Send bimanual command to follower (14 DOF total: 7 per arm)
        bimanual_command = np.concatenate([left_command, right_command])
        self.follower.command_bimanual_joint_pos(bimanual_command)

        # Poll stage success state occasionally and print in JoyLo terminal whenever it changes,
        # recording or not, so the operator can see grasp/stage detection during free teleop too.
        now = time.time()
        if now - self._last_stage_poll_time >= self._stage_poll_interval:
            try:
                task_info = self.follower.get_task_info()
                result = task_info[0]
                inter_success = result.get('intermediate_success', {}) or {}
                # Build a colorized string for current state
                parts = []
                for k, v in inter_success.items():
                    bool_val = bool(v)
                    color = Colors.GREEN if bool_val else Colors.YELLOW
                    parts.append(f"{k}: {color}{bool_val}{Colors.RESET}")
                status_str = "; ".join(parts) if parts else ""
                # Track previous snapshot to detect change (preserve original order)
                current_snapshot = tuple((k, bool(v)) for k, v in inter_success.items())
                prev_snapshot = getattr(self, '_prev_intermediate_snapshot', None)
                if prev_snapshot is None or current_snapshot != prev_snapshot:
                    if status_str:
                        print(f"[TASK PROGRESS] {status_str}")
                        # Check if task is successful and show celebration message
                        task_success = inter_success.get('task_success', False)
                        if task_success:
                            print(f"{Colors.GREEN}[TASK PROGRESS] 🎉 Task is successful! Please save this trajectory{Colors.RESET}")
                    self._prev_intermediate_snapshot = current_snapshot
            except Exception:
                pass
            finally:
                self._last_stage_poll_time = now

    def handle_button_presses(self):
        """Handle button press events from JoyCons."""
        # Update button cooldowns
        self.joycon_agent.update_button_cooldowns()

        # Get current button states
        button_states = self.joycon_agent.get_button_states()

        # Detect button press edges (transition from False to True)
        # Skip L/R — they are used for continuous gripper control in get_gripper_commands()
        for button_name, pressed in button_states.items():
            if button_name in ("l", "r"):
                continue
            was_pressed = self.prev_button_states.get(button_name, False)

            # Edge detection: button just pressed AND not in cooldown
            if pressed and not was_pressed:
                if self.joycon_agent.is_button_ready(button_name):
                    self._handle_button_press(button_name)
                    # Set cooldown for this button
                    self.joycon_agent.set_button_cooldown(button_name)
                else:
                    # Button pressed but still in cooldown
                    cooldown_remaining = self.joycon_agent.button_cooldowns[button_name]
                    print(f"[DEBUG] {button_name.upper()} button pressed but in cooldown ({cooldown_remaining} frames remaining)")

        # Update previous states
        self.prev_button_states = button_states.copy()

    def _handle_button_press(self, button_name: str):
        """Handle a single button press event."""
        if button_name == "x":
            self._handle_x_button()
        elif button_name == "home":
            self._handle_home_button()
        elif button_name in ["y", "b", "a", "capture", "minus", "plus",
                             "left_arrow", "right_arrow", "up", "down"]:
            print(f"[INFO] {button_name.upper()} button pressed. No functionality yet!")
        # Note: ZL/ZR (close) and L/R (open) are handled by continuous gripper control

    def _handle_x_button(self):
        """Handle X button press: toggle between start recording and save trajectory."""
        if self.x_button_action == "start":
            # Start recording
            success = self.follower.start_recording()
            if success:
                self.recording_active = True
                self.x_button_action = "save"

                # Get current demo count for the message
                task_info = self.follower.get_task_info()
                result = task_info[0]
                demo_count = result.get('demo_count', 0)
                demos_per_asset = result.get('demos_per_asset', 1)
                next_demo = demo_count + 1

                print(f"\n{'='*60}")
                print(f"{Colors.GREEN}📹 EPISODE RECORDING STARTED{Colors.RESET}")
                print(f"{'='*60}")
                print(f"{Colors.YELLOW}[INFO] 🎯 Starting to collect {next_demo}-th demo out of {demos_per_asset}{Colors.RESET}")
                print(f"[INFO] Recording trajectory - perform the demonstration")
                print(f"[INFO] Recording state: {Colors.GREEN}ACTIVE{Colors.RESET}")
                print(f"[INFO] Press X again to save trajectory")
                print(f"[INFO] Press HOME to discard and reset\n")
            else:
                print(f"[WARNING] Failed to start recording")

        elif self.x_button_action == "save":
            # Save trajectory
            success = self.follower.save_trajectory()
            if success:
                self.recording_active = False
                self.x_button_action = "start"
                print(f"\n{'='*60}")
                print(f"{Colors.GREEN}💾 TRAJECTORY SAVED{Colors.RESET}")
                print(f"{'='*60}")
                print(f"[INFO] Trajectory has been saved to dataset")

                # Show updated progress and check for completion
                try:
                    task_info = self.follower.get_task_info()
                    result = task_info[0]
                    demo_count = result.get('demo_count', 0)
                    demos_per_asset = result.get('demos_per_asset', 1)
                    total_demos = result.get('total_demos_collected', demo_count)

                    print_progress_text(demo_count, demos_per_asset, prefix="Demo Progress", color=Colors.GREEN)
                    if total_demos != demo_count:
                        print_progress_text(total_demos, demos_per_asset, prefix="Total Progress", color=Colors.GREEN)

                    # Check if data collection is complete
                    if demo_count >= demos_per_asset:
                        print(f"\n{'='*60}")
                        print(f"✅ DATA COLLECTION COMPLETE!")
                        print(f"{'='*60}")
                        print(f"[INFO] All {demos_per_asset} demos have been collected")
                        print(f"[INFO] Exiting teleoperation...")
                        print(f"{'='*60}\n")
                        self.exit_requested = True
                        return

                except Exception as e:
                    print(f"[WARNING] Could not get progress info: {e}")

                # Reset grippers to open position for next recording
                self.joycon_agent.reset_grippers()

                print(f"[INFO] Environment automatically reset for next demonstration")
                print(f"{Colors.YELLOW}[INFO] Ready to start next recording - Press X when ready\n{Colors.RESET}")
            else:
                print(f"[WARNING] Failed to save trajectory")

    def _handle_home_button(self):
        """Handle HOME button press: discard current recording and reset task.

        Unlike save (X button), this does NOT advance the pose schedule.
        Objects will reset to the same scheduled pose.
        """
        print(f"\n{'='*60}")
        print(f"{Colors.RED}🔄 RESET TASK{Colors.RESET}")
        print(f"{'='*60}")

        # Check if recording was active and inform user
        if self.recording_active:
            print(f"[INFO] Discarding current recording")
            print(f"[INFO] Trajectory will not be saved")

        # Reset task in simulation (this automatically discards any active recording)
        result, is_complete = self.follower.reset_task()

        # Reset grippers to open position
        self.joycon_agent.reset_grippers()

        # Reset recording state
        self.recording_active = False
        self.x_button_action = "start"

        print(f"[INFO] Task reset complete")
        print(f"[INFO] - Demo: {result['demo_count']}/{result['demos_per_asset']}")
        print(f"[INFO] - Total demos: {result.get('total_demos', result['demo_count'])}")
        print(f"[INFO] - Recording state: {Colors.RED}STOPPED{Colors.RESET}")

        # Show current progress
        demo_count = result.get('demo_count', 0)
        demos_per_asset = result.get('demos_per_asset', 1)
        total_demos = result.get('total_demos_collected', demo_count)

        print_progress_text(demo_count, demos_per_asset, prefix="Demo Progress", color=Colors.RED)
        if total_demos != demo_count:
            print_progress_text(total_demos, demos_per_asset, prefix="Total Progress", color=Colors.RED)

        if is_complete:
            print(f"\n{'='*60}")
            print(f"✅ COMPLETED ALL DEMOS!")
            print(f"{'='*60}\n")
            self.exit_requested = True
            return

        print(f"[INFO] You can freely move JoyLo to control simulation")
        print(f"{'='*60}")
        print(f"{Colors.YELLOW}[INFO] Press X to start recording when ready{Colors.RESET}")


    def cleanup(self):
        """Clean up resources."""
        print("[INFO] Cleaning up JoyLo controller...")

        # Close data collector if enabled
        if self.enable_recording:
            self.follower.close_data_collector()

        print("[INFO] JoyLo controller cleanup complete")
