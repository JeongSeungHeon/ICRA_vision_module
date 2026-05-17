#!/usr/bin/env python3
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import time
import argparse
import threading
from datetime import datetime
import numpy as np
import cv2
import pygame

import pyzed.sl as sl
from anyskin import AnySkinProcess

try:
    import rerun as rr
except ImportError:
    rr = None

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None


# =========================
# Config
# =========================
TACTILE_W = 400
CAM_W = 960
CAM_H = 540
WINDOW_W = TACTILE_W + CAM_W
WINDOW_H = 720

FPS = 30

NUM_MAGS = 5
NO_CONTACT_NORM_TH = 30
AUTO_RELEASE_TACTILE_DELTA_TH = 30
AUTO_RELEASE_REF_DELAY_S = 0.3
GRASP_DONE_STATUSES = {
    "stopped after extra close",
    "stopped by gOBJ",
    "object detected",
    "closed",
}
GRASP_ACTIVE_STATUSES = {
    "closing",
    "grasping",
    "tactile detected, extra closing",
}
DEFAULT_OBJECT_NAME = "object"
RECORD_BUTTON_RECT = pygame.Rect(24, 650, 170, 38)
BASE_POSE_BUTTON_RECT = pygame.Rect(TACTILE_W + 18, WINDOW_H - 106, 150, 36)

DEFAULT_ROBOT_IP = "192.168.1.218"
BASE_POSE = {
    "x": 547,
    "y": 189,
    "z": 176,
    "roll": -88.8,
    "pitch": 88.6,
    "yaw": -101.6,
}
BASE_POSE_SPEED = 100
GRIPPER_SLAVE_ID = 0x09
GRIPPER_CLOSE_POS = 0xFF
GRIPPER_OPEN_POS = 0x00
GRIPPER_CLOSE_SPEED = 0xFF
GRIPPER_CLOSE_FORCE = 0x80
GRIPPER_OPEN_SPEED = 0x80
GRIPPER_OPEN_FORCE = 0x20
AUTO_BASELINE_RESET_DELAY_S = 1.5

TELEOP_CONTROL_HZ = 10
TELEOP_CONTROL_DT = 1.0 / TELEOP_CONTROL_HZ
TELEOP_VC_DURATION = 0.1
TELEOP_VEL_ALPHA = 0.4
TELEOP_VEL_DEADBAND = 0.5

TELEOP_KEY_BINDINGS = {
    pygame.K_e: ("z", 1),
    pygame.K_q: ("z", -1),
}


# 네 코드에서 쓰던 chip 위치 기반
CHIP_LOCATIONS = np.array([
    [180, 270],
    [180, 200],
    [180, 340],
    [110, 270],
    [250, 270],
], dtype=np.float32)

CHIP_XY_ROTATIONS = np.array([
    -np.pi / 2,
    -np.pi / 2,
    np.pi,
    np.pi / 2,
    0.0
])


# =========================
# ZED Camera
# =========================
class ZEDCamera:
    def __init__(self, width=CAM_W, height=CAM_H):
        self.width = width
        self.height = height
        self.zed = sl.Camera()
        self.image = sl.Mat()

    def open(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")

        print("[ZED] Camera opened.")

    def get_rgb_frame(self):
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            return None

        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        frame_rgba = self.image.get_data()  # BGRA or RGBA-like ndarray

        # ZED image is usually RGBA. Convert to RGB for pygame.
        frame_rgb = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2RGB)
        frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))

        return frame_rgb

    def close(self):
        self.zed.close()
        print("[ZED] Camera closed.")


# =========================
# Tactile Sensor
# =========================
class TactileSensor:
    def __init__(self, port="/dev/ttyACM0"):
        self.port = port
        self.stream = AnySkinProcess(num_mags=NUM_MAGS, port=port)
        self.baseline = None
        self.latest = np.zeros(NUM_MAGS * 3, dtype=np.float32)

    def start(self):
        self.stream.start()
        time.sleep(1.0)
        self.reset_baseline()
        print("[Tactile] Sensor stream started.")

    def reset_baseline(self):
        baseline_data = self.stream.get_data(num_samples=5)
        baseline_data = np.array(baseline_data)[:, 1:]
        self.baseline = np.mean(baseline_data, axis=0)
        print("[Tactile] Baseline reset.")

    def read(self):
        try:
            sensor_data = self.stream.get_data(num_samples=1)[0][1:]
            sensor_data = np.array(sensor_data, dtype=np.float32)
            self.latest = sensor_data - self.baseline
            return self.latest
        except Exception as e:
            print("[Tactile] Read error:", e)
            return self.latest

    def close(self):
        self.stream.pause_streaming()
        self.stream.join()
        print("[Tactile] Sensor stream stopped.")


# =========================
# xArm XYZ Teleoperation
# =========================
class XArmXYZTeleop:
    def __init__(self, arm, lock, speed=80.0):
        self.arm = arm
        self.lock = lock
        self.speed = speed
        self.running = False
        self.thread = None
        self.target_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.current_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pose = None
        self.last_code = 0
        self.last_error = ""

    def start(self):
        with self.lock:
            self.arm.motion_enable(True)
            self.arm.set_mode(5)
            self.arm.set_state(0)
            self.arm.set_cartesian_velo_continuous(True)

        self.running = True
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.thread.start()
        print("[Teleop] XYZ velocity control started.")

    def stop_motion(self):
        with self.lock:
            self.arm.vc_set_cartesian_velocity(
                [0, 0, 0, 0, 0, 0],
                is_radian=False,
                is_tool_coord=False,
                duration=0.1,
            )

    def send_velocity(self, velocity):
        with self.lock:
            ret = self.arm.vc_set_cartesian_velocity(
                velocity,
                is_radian=False,
                is_tool_coord=False,
                duration=TELEOP_VC_DURATION,
            )

        if isinstance(ret, (list, tuple)):
            return int(ret[0]) if len(ret) > 0 else -999
        return int(ret) if ret is not None else -999

    def read_pose(self):
        with self.lock:
            code, pose = self.arm.get_position(is_radian=False)
        return pose if code == 0 else None

    def set_target_velocity(self, velocity):
        self.target_velocity = list(velocity)

    def _control_loop(self):
        current_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        next_time = time.perf_counter()
        last_pose_query = time.time()

        while self.running:
            target_velocity = list(self.target_velocity)

            for i in range(6):
                current_velocity[i] = (
                    TELEOP_VEL_ALPHA * target_velocity[i]
                    + (1.0 - TELEOP_VEL_ALPHA) * current_velocity[i]
                )

                if abs(current_velocity[i]) < TELEOP_VEL_DEADBAND and abs(target_velocity[i]) < TELEOP_VEL_DEADBAND:
                    current_velocity[i] = 0.0

            try:
                self.last_code = self.send_velocity(current_velocity)
                self.last_error = ""
            except Exception as e:
                self.last_code = -999
                self.last_error = str(e)

            if time.time() - last_pose_query > 0.5:
                try:
                    pose = self.read_pose()
                    if pose is not None:
                        self.pose = pose
                except Exception as e:
                    self.last_error = f"get_position error: {e}"
                last_pose_query = time.time()

            self.current_velocity = list(current_velocity)

            next_time += TELEOP_CONTROL_DT
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()

        try:
            self.stop_motion()
        except Exception as e:
            print("[Teleop] stop_motion error:", e)

        print("[Teleop] XYZ velocity control stopped.")

    def stop(self):
        self.running = False
        self.target_velocity = [0, 0, 0, 0, 0, 0]
        if self.thread is not None:
            self.thread.join(timeout=2.0)

        with self.lock:
            self.arm.set_cartesian_velo_continuous(False)
            self.arm.set_mode(0)
            self.arm.set_state(0)


# =========================
# Gripper Control
# =========================
class RobotiqGripper:
    def __init__(self, robot_ip=DEFAULT_ROBOT_IP, arm=None, lock=None):
        if XArmAPI is None:
            raise RuntimeError("xarm package is not installed. Install xArm-Python-SDK to use --gripper.")

        self.robot_ip = robot_ip
        self.arm = arm
        self.owns_arm = arm is None
        self.lock = lock if lock is not None else threading.Lock()
        self.cancel_event = threading.Event()
        self.last_status = "disconnected"
        self.last_cmd_pos = GRIPPER_OPEN_POS
        self.close_thread = None
        self.last_gripper_status = None

    def connect(self):
        self.last_status = "connecting"
        if self.arm is None:
            self.arm = XArmAPI(self.robot_ip)
        time.sleep(0.2)

        with self.lock:
            if self.arm.warn_code != 0:
                self.arm.clean_warn()
            if self.arm.error_code != 0:
                self.arm.clean_error()

            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.motion_enable(True)

            ret = self.arm.core.set_modbus_timeout(100)
        print("[Gripper] set modbus timeout ret:", self._ret_code(ret))

        with self.lock:
            ret = self.arm.core.set_modbus_baudrate(115200)
        print("[Gripper] set modbus baudrate ret:", self._ret_code(ret))

        self.activate()
        self.last_status = "ready"
        print("[Gripper] Connected and activated.")

    def _ret_code(self, ret):
        return ret[0] if isinstance(ret, (list, tuple)) and len(ret) > 0 else ret

    def _write_regs(self, b0, b1, b2, b3, b4, b5):
        frame = [
            GRIPPER_SLAVE_ID,
            0x10,
            0x03, 0xE8,
            0x00, 0x03,
            0x06,
            b0 & 0xFF, b1 & 0xFF, b2 & 0xFF,
            b3 & 0xFF, b4 & 0xFF, b5 & 0xFF,
        ]
        ret = self.arm.core.tgpio_set_modbus(frame, len(frame))
        return self._ret_code(ret), ret

    def _read_status_regs(self, reg_count=2):
        frame = [
            GRIPPER_SLAVE_ID,
            0x04,
            0x07, 0xD0,
            0x00, reg_count & 0xFF,
        ]
        ret = self.arm.core.tgpio_set_modbus(frame, len(frame))
        return self._ret_code(ret), ret

    def _parse_status(self, ret):
        if ret is None or len(ret) < 9:
            return None

        data = ret[5:9]
        byte0, byte1, byte2, byte3 = data
        status = {
            "gOBJ": (byte0 >> 6) & 0x03,
            "gSTA": (byte0 >> 4) & 0x03,
            "gGTO": (byte0 >> 3) & 0x01,
            "gACT": byte0 & 0x01,
            "pos": byte3,
            "raw_data": data,
        }
        self.last_gripper_status = status
        return status

    def activate(self):
        with self.lock:
            code, _ = self._write_regs(0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
            print("[Gripper] clear rACT code:", code)
            time.sleep(0.2)

            code, _ = self._write_regs(0x01, 0x00, 0x00, 0x00, 0x00, 0x00)
            print("[Gripper] set rACT code:", code)
            time.sleep(2.0)

    def goto_position(self, pos, speed=0x80, force=0x80):
        with self.lock:
            code, _ = self._write_regs(0x09, 0x00, 0x00, pos, speed, force)
        self.last_cmd_pos = pos
        print(f"[Gripper] goto pos=0x{pos:02X} code:", code)
        return code

    def open(self):
        self.cancel_event.set()
        self.last_status = "opening"
        code = self.goto_position(GRIPPER_OPEN_POS, speed=GRIPPER_OPEN_SPEED, force=GRIPPER_OPEN_FORCE)
        self.last_status = "open" if code == 0 else f"open error {code}"

    def close_until_object_or_fully_closed(self, timeout_s=5.0, poll_dt=0.03):
        self.cancel_event.clear()
        self.last_status = "closing"
        self.goto_position(GRIPPER_CLOSE_POS, speed=GRIPPER_CLOSE_SPEED, force=GRIPPER_CLOSE_FORCE)

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.cancel_event.is_set():
                self.last_status = "close canceled"
                return

            with self.lock:
                _, ret = self._read_status_regs(reg_count=2)

            status = self._parse_status(ret)
            if status is None:
                time.sleep(poll_dt)
                continue

            gobj = status["gOBJ"]
            if gobj == 2:
                self.last_status = "object detected"
                print("[Gripper] Object detected while closing.")
                return
            if gobj == 3:
                self.last_status = "closed"
                print("[Gripper] Reached requested position.")
                return

            time.sleep(poll_dt)

        self.last_status = "close timeout"
        print("[Gripper] Timeout while closing.")

    def stop_grasp(self, status=None):
        status = status or self.last_gripper_status
        hold_pos = status["pos"] if status is not None else self.last_cmd_pos

        # rACT=1, rGTO=0 tells the Robotiq gripper to stop the current motion.
        # This is used for tactile-triggered stops so it does not keep closing.
        with self.lock:
            code, _ = self._write_regs(
                0x01,
                0x00,
                0x00,
                hold_pos,
                GRIPPER_CLOSE_SPEED,
                GRIPPER_CLOSE_FORCE,
            )

        print(f"[Gripper] stop grasp at pos=0x{hold_pos:02X} code:", code)

    def close_until_object_or_tactile_contact(self, get_tactile_norm, extra_grasp_pos=10, poll_dt=0.03):
        self.cancel_event.clear()
        self.last_status = "grasping"
        self.goto_position(GRIPPER_CLOSE_POS, speed=GRIPPER_CLOSE_SPEED, force=GRIPPER_CLOSE_FORCE)

        while True:
            if self.cancel_event.is_set():
                self.last_status = "grasp canceled"
                return

            tactile_norm = get_tactile_norm()

            with self.lock:
                _, ret = self._read_status_regs(reg_count=2)

            status = self._parse_status(ret)
            if status is not None and status["gOBJ"] == 2:
                self.last_status = "stopped by gOBJ"
                self.stop_grasp(status)
                print("[Gripper] Stopped grasp: Robotiq object detected.")
                return

            if tactile_norm > NO_CONTACT_NORM_TH:
                if status is None:
                    self.last_status = "stopped by tactile"
                    self.stop_grasp(status)
                    print(f"[Gripper] Stopped grasp: tactile norm {tactile_norm:.2f} > {NO_CONTACT_NORM_TH}, status unavailable.")
                    return

                contact_pos = status["pos"]
                extra_grasp_pos = max(0, int(extra_grasp_pos))
                target_pos = min(contact_pos + extra_grasp_pos, GRIPPER_CLOSE_POS)
                self.last_status = "tactile detected, extra closing"
                print(
                    "[Gripper] Tactile contact detected. "
                    f"norm={tactile_norm:.2f}, contact_pos=0x{contact_pos:02X}, "
                    f"target_pos=0x{target_pos:02X}, raw_data={status['raw_data']}"
                )

                self.goto_position(target_pos, speed=GRIPPER_CLOSE_SPEED, force=GRIPPER_CLOSE_FORCE)

                while True:
                    if self.cancel_event.is_set():
                        self.last_status = "grasp canceled"
                        return

                    with self.lock:
                        _, ret = self._read_status_regs(reg_count=2)

                    extra_status = self._parse_status(ret)
                    if extra_status is None:
                        time.sleep(poll_dt)
                        continue

                    extra_pos = extra_status["pos"]
                    if extra_status["gOBJ"] == 2 or extra_pos >= target_pos:
                        self.last_status = "stopped after extra close"
                        self.stop_grasp(extra_status)
                        print(
                            "[Gripper] Stopped after extra close. "
                            f"pos=0x{extra_pos:02X}, target_pos=0x{target_pos:02X}, "
                            f"gOBJ={extra_status['gOBJ']}, raw_data={extra_status['raw_data']}"
                        )
                        return

                    time.sleep(poll_dt)

            if status is not None and status["gOBJ"] == 3:
                self.last_status = "fully closed"
                print("[Gripper] Reached requested position before tactile contact.")
                return

            time.sleep(poll_dt)

    def close_async(self):
        if self.close_thread is not None and self.close_thread.is_alive():
            self.last_status = "already closing"
            return

        self.close_thread = threading.Thread(
            target=self.close_until_object_or_fully_closed,
            daemon=True,
        )
        self.close_thread.start()

    def grasp_async(self, get_tactile_norm, extra_grasp_pos=10):
        if self.close_thread is not None and self.close_thread.is_alive():
            self.last_status = "already grasping"
            return

        self.close_thread = threading.Thread(
            target=self.close_until_object_or_tactile_contact,
            args=(get_tactile_norm, extra_grasp_pos),
            daemon=True,
        )
        self.close_thread.start()

    def open_async(self):
        self.cancel_event.set()
        thread = threading.Thread(target=self.open, daemon=True)
        thread.start()

    def close(self):
        if self.arm is not None and self.owns_arm:
            self.arm.disconnect()
            print("[Gripper] Disconnected.")


# =========================
# Drawing Utils
# =========================
def draw_text(surface, text, x, y, size=24, color=(20, 20, 20)):
    font = pygame.font.Font(None, size)
    img = font.render(text, True, color)
    surface.blit(img, (x, y))


def sanitize_filename_text(text):
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or DEFAULT_OBJECT_NAME


def get_object_name(window, clock):
    object_name = ""
    prompt = "Recording object name:"
    active = True

    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_RETURN:
                    active = False
                elif event.key == pygame.K_BACKSPACE:
                    object_name = object_name[:-1]
                elif event.unicode and event.unicode.isprintable():
                    object_name += event.unicode

        window.fill((245, 247, 250))
        draw_text(window, prompt, 80, 240, 44, (20, 20, 20))

        input_rect = pygame.Rect(80, 300, WINDOW_W - 160, 58)
        pygame.draw.rect(window, (255, 255, 255), input_rect, border_radius=6)
        pygame.draw.rect(window, (30, 90, 160), input_rect, width=2, border_radius=6)

        shown_text = object_name if object_name else DEFAULT_OBJECT_NAME
        text_color = (20, 20, 20) if object_name else (150, 150, 150)
        draw_text(window, shown_text, input_rect.x + 16, input_rect.y + 15, 34, text_color)
        draw_text(window, "Press Enter to start / ESC to quit", 80, 385, 26, (80, 80, 80))

        pygame.display.flip()
        clock.tick(FPS)

    return sanitize_filename_text(object_name)


def draw_tactile_panel(surface, tactile_data, scaling=10.0, teleop_enabled=False):
    # panel background
    pygame.draw.rect(surface, (255, 255, 255), (0, 0, TACTILE_W, WINDOW_H))

    draw_text(surface, "Tactile Sensor", 24, 20, 34)
    draw_text(surface, "b: reset baseline", 24, 55, 24)
    draw_text(surface, "ESC: quit", 24, 80, 24)

    data = tactile_data.reshape(NUM_MAGS, 3).copy()

    # 네 기존 코드처럼 x/y flip
    data[:, :2] *= -1

    data_mag = np.linalg.norm(data, axis=1)
    total_norm = np.linalg.norm(data.flatten())

    if total_norm < NO_CONTACT_NORM_TH:
        draw_text(surface, "No Contact", 24, 115, 36, color=(200, 0, 0))
    else:
        draw_text(surface, "Contact", 24, 115, 36, color=(0, 130, 0))

    # chip visualization
    for magid, chip_location in enumerate(CHIP_LOCATIONS):
        x, y = chip_location.astype(int)

        # z-axis magnitude circle
        z_val = data[magid, 2]
        radius = max(3, int(abs(z_val) / scaling))
        width = 2 if z_val < 0 else 0

        pygame.draw.circle(surface, (255, 80, 70), (x, y), radius, width)

        # xy arrow
        rot = CHIP_XY_ROTATIONS[magid]
        rotation_mat = np.array([
            [np.cos(rot), -np.sin(rot)],
            [np.sin(rot),  np.cos(rot)],
        ])

        data_xy = rotation_mat @ data[magid, :2]
        end_x = int(x + data_xy[0] / scaling)
        end_y = int(y + data_xy[1] / scaling)

        pygame.draw.line(surface, (0, 180, 0), (x, y), (end_x, end_y), 6)
        pygame.draw.circle(surface, (20, 20, 20), (x, y), 5)

        draw_text(surface, f"M{magid}", x - 12, y - 35, 22)

    # numeric values
    base_y = 400
    draw_text(surface, f"Total norm: {total_norm:.2f}", 24, base_y, 26)

    for i in range(NUM_MAGS):
        bx, by, bz = tactile_data[i*3:i*3+3]
        mag = np.linalg.norm(tactile_data[i*3:i*3+3])
        line = f"M{i}: x={bx:7.1f}, y={by:7.1f}, z={bz:7.1f}, |B|={mag:7.1f}"
        draw_text(surface, line, 24, base_y + 32 + i * 24, 22)


def draw_gripper_status(surface, gripper, grasp_enabled=False):
    close_label = "c: grasp until contact" if grasp_enabled else "c: close gripper"
    draw_text(surface, close_label, 24, 548, 24)
    draw_text(surface, "v: open gripper", 24, 573, 24)
    status = gripper.last_status if gripper is not None else "disabled"
    draw_text(surface, f"Gripper: {status}", 24, 603, 26, (20, 90, 150))


def draw_auto_release_status(surface, auto_release_armed, auto_release_triggered, hold_tactile_ref):
    if auto_release_triggered:
        status = "triggered"
        color = (180, 40, 40)
    elif auto_release_armed:
        status = "armed"
        color = (30, 120, 70)
    else:
        status = "off"
        color = (90, 90, 90)

    ref_text = "none" if hold_tactile_ref is None else f"{hold_tactile_ref:.1f}"
    draw_text(
        surface,
        f"Auto release: {status}  ref={ref_text}  d>{AUTO_RELEASE_TACTILE_DELTA_TH}",
        24,
        628,
        22,
        color,
    )


def compute_xyz_velocity(keys, speed):
    velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for key, (axis, direction) in TELEOP_KEY_BINDINGS.items():
        if not keys[key]:
            continue

        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        velocity[axis_idx] += direction * speed

    return velocity


def draw_teleop_status(surface, teleop, teleop_enabled):
    if not teleop_enabled:
        draw_text(surface, "Teleop: disabled", TACTILE_W + 18, WINDOW_H - 32, 24, (70, 70, 70))
        return

    if teleop is None:
        draw_text(surface, "Teleop: connecting", TACTILE_W + 18, WINDOW_H - 32, 24, (180, 80, 20))
        return

    velocity = teleop.current_velocity
    draw_text(
        surface,
        "Z: hold e/q | gripper: c/v | ESC quit",
        TACTILE_W + 18,
        WINDOW_H - 58,
        24,
        (20, 20, 20),
    )
    draw_text(
        surface,
        f"z vel={velocity[2]:.1f} mm/s | code={teleop.last_code}",
        TACTILE_W + 18,
        WINDOW_H - 32,
        24,
        (20, 90, 150) if teleop.last_code == 0 else (180, 40, 40),
    )


def draw_base_pose_button(surface, status, enabled):
    color = (45, 105, 165) if enabled else (150, 150, 150)
    pygame.draw.rect(surface, color, BASE_POSE_BUTTON_RECT, border_radius=6)
    draw_text(
        surface,
        "Base Pose",
        BASE_POSE_BUTTON_RECT.x + 20,
        BASE_POSE_BUTTON_RECT.y + 9,
        24,
        (255, 255, 255),
    )
    draw_text(
        surface,
        f"Base: {status}",
        BASE_POSE_BUTTON_RECT.right + 14,
        BASE_POSE_BUTTON_RECT.y + 9,
        24,
        (20, 90, 150) if enabled else (100, 100, 100),
    )


def draw_recording_status(surface, is_recording, object_name):
    color = (190, 35, 35) if is_recording else (35, 110, 70)
    label = "Stop REC" if is_recording else "Start REC"

    pygame.draw.rect(surface, color, RECORD_BUTTON_RECT, border_radius=6)
    draw_text(surface, label, RECORD_BUTTON_RECT.x + 18, RECORD_BUTTON_RECT.y + 9, 26, (255, 255, 255))
    
    if is_recording:
        draw_text(surface, f"REC  Object: {object_name}", 205, 657, 24, (180, 0, 0))


def frame_to_pygame_surface(frame_rgb):
    # pygame expects shape (W, H, 3), so transpose
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
    return frame_surface


def pygame_surface_to_bgr_frame(surface):
    frame_rgb = pygame.surfarray.array3d(surface)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def log_rerun_frame(frame_rgb, visualizer_rgb, tactile_data, gripper, teleop, object_name, frame_idx, elapsed_s):
    if rr is None:
        return

    rr.set_time("frame", sequence=frame_idx)
    rr.set_time("time", duration=elapsed_s)

    rr.log("camera/rgb", rr.Image(frame_rgb).compress())
    rr.log("visualizer/window", rr.Image(visualizer_rgb).compress())

    tactile_values = np.array(tactile_data, dtype=np.float32).reshape(NUM_MAGS, 3)
    total_norm = float(np.linalg.norm(tactile_values.flatten()))
    rr.log("tactile/total_norm", rr.Scalars(total_norm))

    for mag_idx in range(NUM_MAGS):
        bx, by, bz = tactile_values[mag_idx]
        mag_norm = float(np.linalg.norm(tactile_values[mag_idx]))
        rr.log(f"tactile/mag_{mag_idx}/x", rr.Scalars(float(bx)))
        rr.log(f"tactile/mag_{mag_idx}/y", rr.Scalars(float(by)))
        rr.log(f"tactile/mag_{mag_idx}/z", rr.Scalars(float(bz)))
        rr.log(f"tactile/mag_{mag_idx}/norm", rr.Scalars(mag_norm))

    rr.log("recording/object_name_live", rr.TextLog(object_name))

    if gripper is not None:
        rr.log("gripper/status", rr.TextLog(gripper.last_status))
        if gripper.last_gripper_status is not None:
            status = gripper.last_gripper_status
            rr.log("gripper/gOBJ", rr.Scalars(float(status["gOBJ"])))
            rr.log("gripper/gSTA", rr.Scalars(float(status["gSTA"])))
            rr.log("gripper/pos", rr.Scalars(float(status["pos"])))

    if teleop is not None:
        velocity = teleop.current_velocity
        rr.log("robot/velocity/x", rr.Scalars(float(velocity[0])))
        rr.log("robot/velocity/y", rr.Scalars(float(velocity[1])))
        rr.log("robot/velocity/z", rr.Scalars(float(velocity[2])))

        if teleop.pose is not None:
            pose = teleop.pose
            labels = ["x", "y", "z", "roll", "pitch", "yaw"]
            for idx, label in enumerate(labels):
                rr.log(f"robot/pose/{label}", rr.Scalars(float(pose[idx])))


# =========================
# Main App
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--scaling", type=float, default=10.0)
    parser.add_argument("--no_tactile", action="store_true")
    parser.add_argument("-r", "--record", action="store_true", help="kept for compatibility; use r or the button to start/stop mp4 recording")
    parser.add_argument("--gripper", action="store_true", help="enable xArm Robotiq gripper control")
    parser.add_argument("--grasp", action="store_true", help="stop closing when Robotiq detects an object or tactile total norm exceeds the contact threshold")
    parser.add_argument("--extra_grasp_pos", type=int, default=10, help="extra Robotiq close position after tactile contact is detected")
    parser.add_argument("--teleop", action="store_true", help="enable continuous xArm z-axis keyboard teleoperation")
    parser.add_argument("--teleop_speed", type=float, default=80.0, help="teleop Cartesian target velocity in mm/s")
    parser.add_argument("--robot_ip", type=str, default=DEFAULT_ROBOT_IP, help="xArm robot IP for gripper control")
    args = parser.parse_args()
    if args.grasp:
        args.gripper = True
    if args.teleop:
        args.gripper = True

    pygame.init()
    window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("ZED + Tactile Visualization")
    clock = pygame.time.Clock()

    zed = ZEDCamera()
    tactile = None
    video_writer = None
    rerun_enabled = False
    rerun_filename = None
    recording_start_time = None
    recording_frame_idx = 0
    gripper = None
    teleop = None
    shared_arm = None
    shared_arm_lock = threading.Lock()
    object_name = DEFAULT_OBJECT_NAME
    video_filename = None
    close_command_sent = False
    auto_baseline_reset_at = None
    auto_release_armed = False
    auto_release_triggered = False
    hold_tactile_ref = None
    hold_ref_capture_at = None
    hold_ref_source_status = None
    base_pose_state = {
        "status": "ready",
        "thread": None,
    }

    def start_recording():
        nonlocal video_writer, object_name, video_filename
        nonlocal rerun_enabled, rerun_filename, recording_start_time, recording_frame_idx
        if video_writer is not None:
            return

        if teleop is not None:
            teleop.set_target_velocity([0, 0, 0, 0, 0, 0])

        next_object_name = get_object_name(window, clock)
        if next_object_name is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_prefix = f"tactile_sensing_{timestamp}_{next_object_name}"
        next_video_filename = f"{recording_prefix}.mp4"
        next_rerun_filename = f"{recording_prefix}.rrd"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        next_video_writer = cv2.VideoWriter(
            next_video_filename,
            fourcc,
            FPS,
            (WINDOW_W, WINDOW_H),
        )
        if not next_video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {next_video_filename}")

        object_name = next_object_name
        video_filename = next_video_filename
        video_writer = next_video_writer
        rerun_filename = next_rerun_filename
        rerun_enabled = False
        recording_start_time = time.time()
        recording_frame_idx = 0

        if rr is not None:
            rr.init("tactile_sensing", recording_id=recording_prefix, spawn=False)
            rr.save(rerun_filename)
            rr.log("recording/object_name", rr.TextLog(object_name), static=True)
            rerun_enabled = True
            print(f"[Record] Saving rerun to {rerun_filename}")
        else:
            print("[Record] rerun package not available; only mp4 will be saved.")

        print(f"[Record] Saving video to {video_filename}")

    def stop_recording():
        nonlocal video_writer, rerun_enabled
        if video_writer is None:
            return

        video_writer.release()
        video_writer = None
        rerun_enabled = False
        print(f"[Record] Video saved: {video_filename}")
        if rerun_filename is not None and rr is not None:
            print(f"[Record] Rerun saved: {rerun_filename}")

    def toggle_recording():
        if video_writer is None:
            start_recording()
        else:
            stop_recording()

    def get_tactile_norm():
        return float(np.linalg.norm(tactile_data.flatten()))

    def move_to_base_pose_async():
        thread = base_pose_state["thread"]
        if thread is not None and thread.is_alive():
            base_pose_state["status"] = "already moving"
            return

        if gripper is None:
            base_pose_state["status"] = "gripper disabled"
            print("[Base Pose] Gripper/robot is not enabled. Run with --gripper or --teleop.")
            return

        def worker():
            try:
                base_pose_state["status"] = "opening gripper"
                print("[Base Pose] Opening gripper before moving.")

                if teleop is not None:
                    teleop.set_target_velocity([0, 0, 0, 0, 0, 0])
                    teleop.stop()

                gripper.cancel_event.set()
                gripper.open()
                time.sleep(1.5)

                base_pose_state["status"] = "moving"
                print("[Base Pose] Moving to base pose.")

                with gripper.lock:
                    arm = gripper.arm
                    if arm.warn_code != 0:
                        arm.clean_warn()
                    if arm.error_code != 0:
                        arm.clean_error()

                    arm.motion_enable(True)
                    arm.set_mode(0)
                    arm.set_state(0)
                    time.sleep(0.1)

                    code = arm.set_position(
                        x=BASE_POSE["x"],
                        y=BASE_POSE["y"],
                        z=BASE_POSE["z"],
                        roll=BASE_POSE["roll"],
                        pitch=BASE_POSE["pitch"],
                        yaw=BASE_POSE["yaw"],
                        speed=BASE_POSE_SPEED,
                        wait=True,
                    )

                if code == 0:
                    base_pose_state["status"] = "reached"
                    print("[Base Pose] Reached base pose.")
                else:
                    base_pose_state["status"] = f"error {code}"
                    print(f"[Base Pose] Failed to move to base pose: code={code}")

            except Exception as e:
                base_pose_state["status"] = "error"
                print("[Base Pose] Error:", e)

            finally:
                if teleop is not None and not teleop.running:
                    teleop.start()

        base_pose_state["thread"] = threading.Thread(target=worker, daemon=True)
        base_pose_state["thread"].start()

    try:
        zed.open()

        if not args.no_tactile:
            tactile = TactileSensor(port=args.port)
            tactile.start()

        if args.teleop:
            if XArmAPI is None:
                raise RuntimeError("xarm package is not installed. Install xArm-Python-SDK to use --teleop.")

            shared_arm = XArmAPI(args.robot_ip)
            time.sleep(0.2)
            with shared_arm_lock:
                if shared_arm.warn_code != 0:
                    shared_arm.clean_warn()
                if shared_arm.error_code != 0:
                    shared_arm.clean_error()
                shared_arm.motion_enable(True)

        if args.gripper:
            gripper = RobotiqGripper(
                robot_ip=args.robot_ip,
                arm=shared_arm,
                lock=shared_arm_lock if shared_arm is not None else None,
            )
            gripper.connect()

        if args.teleop:
            teleop = XArmXYZTeleop(
                arm=shared_arm,
                lock=shared_arm_lock,
                speed=args.teleop_speed,
            )
            teleop.start()

        running = True
        tactile_data = np.zeros(NUM_MAGS * 3, dtype=np.float32)

        while running:
            # -------------------------
            # Events
            # -------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and RECORD_BUTTON_RECT.collidepoint(event.pos):
                        toggle_recording()
                    elif event.button == 1 and BASE_POSE_BUTTON_RECT.collidepoint(event.pos):
                        move_to_base_pose_async()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_r:
                        toggle_recording()

                    elif event.key == pygame.K_b:
                        if tactile is not None:
                            tactile.reset_baseline()
                            close_command_sent = False
                            auto_baseline_reset_at = None
                            auto_release_armed = False
                            auto_release_triggered = False
                            hold_tactile_ref = None
                            hold_ref_capture_at = None
                            hold_ref_source_status = None

                    elif event.key == pygame.K_c:
                        if gripper is not None:
                            if args.grasp:
                                gripper.grasp_async(get_tactile_norm, args.extra_grasp_pos)
                            else:
                                gripper.close_async()
                            close_command_sent = True
                            auto_release_armed = False
                            auto_release_triggered = False
                            hold_tactile_ref = None
                            hold_ref_capture_at = None
                            hold_ref_source_status = None
                    elif event.key == pygame.K_v:
                        if gripper is not None:
                            gripper.open_async()
                            auto_release_armed = False
                            auto_release_triggered = True
                            hold_tactile_ref = None
                            hold_ref_capture_at = None
                            hold_ref_source_status = None
                            if tactile is not None and close_command_sent:
                                auto_baseline_reset_at = time.time() + AUTO_BASELINE_RESET_DELAY_S
                                close_command_sent = False

            if teleop is not None:
                teleop.set_target_velocity(
                    compute_xyz_velocity(pygame.key.get_pressed(), args.teleop_speed)
                )

            if auto_baseline_reset_at is not None and time.time() >= auto_baseline_reset_at:
                if tactile is not None:
                    tactile.reset_baseline()
                    tactile_data = np.zeros(NUM_MAGS * 3, dtype=np.float32)
                    print("[Tactile] Baseline reset automatically after opening gripper.")
                auto_baseline_reset_at = None

            # -------------------------
            # Read sensors
            # -------------------------
            frame_rgb = zed.get_rgb_frame()
            if frame_rgb is None:
                continue

            if tactile is not None:
                tactile_data = tactile.read()

            tactile_total_norm = float(np.linalg.norm(tactile_data.flatten()))
            gripper_status = gripper.last_status if gripper is not None else None

            if gripper_status in GRASP_ACTIVE_STATUSES and (
                auto_release_armed
                or auto_release_triggered
                or hold_ref_capture_at is not None
                or hold_tactile_ref is not None
            ):
                auto_release_armed = False
                auto_release_triggered = False
                hold_tactile_ref = None
                hold_ref_capture_at = None
                hold_ref_source_status = None

            if (
                gripper_status in GRASP_DONE_STATUSES
                and not auto_release_armed
                and not auto_release_triggered
                and hold_ref_capture_at is None
            ):
                hold_ref_capture_at = time.time() + AUTO_RELEASE_REF_DELAY_S
                hold_ref_source_status = gripper_status
                print(
                    "[Auto Release] Grasp complete. "
                    f"Capturing tactile reference in {AUTO_RELEASE_REF_DELAY_S:.1f}s "
                    f"(status={gripper_status})."
                )

            if hold_ref_capture_at is not None and time.time() >= hold_ref_capture_at:
                hold_tactile_ref = tactile_total_norm
                auto_release_armed = True
                hold_ref_capture_at = None
                print(
                    "[Auto Release] Armed. "
                    f"ref={hold_tactile_ref:.2f}, source_status={hold_ref_source_status}, "
                    f"delta_threshold={AUTO_RELEASE_TACTILE_DELTA_TH}."
                )

            if auto_release_armed and not auto_release_triggered and hold_tactile_ref is not None:
                release_delta = abs(tactile_total_norm - hold_tactile_ref)
                if gripper is not None and release_delta > AUTO_RELEASE_TACTILE_DELTA_TH:
                    gripper.open_async()
                    auto_release_armed = False
                    auto_release_triggered = True
                    close_command_sent = False
                    print(
                        "[Auto Release] Triggered. "
                        f"norm={tactile_total_norm:.2f}, ref={hold_tactile_ref:.2f}, "
                        f"delta={release_delta:.2f} > {AUTO_RELEASE_TACTILE_DELTA_TH}."
                    )

            # -------------------------
            # Draw
            # -------------------------
            window.fill((255, 255, 255))

            draw_tactile_panel(window, tactile_data, scaling=args.scaling, teleop_enabled=args.teleop)
            draw_gripper_status(window, gripper, grasp_enabled=args.grasp)
            draw_auto_release_status(window, auto_release_armed, auto_release_triggered, hold_tactile_ref)
            draw_recording_status(window, video_writer is not None, object_name)
            draw_base_pose_button(window, base_pose_state["status"], gripper is not None)
            draw_teleop_status(window, teleop, args.teleop)

            cam_surface = frame_to_pygame_surface(frame_rgb)
            window.blit(cam_surface, (TACTILE_W, 0))

            pygame.display.flip()

            if video_writer is not None:
                visualizer_bgr = pygame_surface_to_bgr_frame(window)
                video_writer.write(visualizer_bgr)

            if rerun_enabled:
                elapsed_s = time.time() - recording_start_time
                log_rerun_frame(
                    frame_rgb=frame_rgb,
                    visualizer_rgb=cv2.cvtColor(visualizer_bgr, cv2.COLOR_BGR2RGB),
                    tactile_data=tactile_data,
                    gripper=gripper,
                    teleop=teleop,
                    object_name=object_name,
                    frame_idx=recording_frame_idx,
                    elapsed_s=elapsed_s,
                )
                recording_frame_idx += 1

            clock.tick(FPS)

    finally:
        if video_writer is not None:
            stop_recording()
        if tactile is not None:
            tactile.close()
        if gripper is not None:
            gripper.cancel_event.set()
        if teleop is not None:
            teleop.stop()
        if gripper is not None:
            gripper.close()
        if shared_arm is not None:
            shared_arm.disconnect()
            print("[xArm] Shared arm disconnected.")
        zed.close()
        pygame.quit()


if __name__ == "__main__":
    main()

# python 0517_visualize_cam_tactile_robot_control_gripper_open.py --gripper --grasp --teleop --extra_grasp_pos 10
