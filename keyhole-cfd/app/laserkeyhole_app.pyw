#!/usr/bin/env python3
"""
laserkeyhole_app.pyw - Standalone Windows GUI for laser keyhole OpenFOAM simulations.

Run from Windows by double-clicking or with:
    pythonw keyhole-cfd/app/laserkeyhole_app.pyw
    python  keyhole-cfd/app/laserkeyhole_app.pyw

Requirements:
    - Windows 10/11 with WSL2 and an OpenFOAM 2306 environment.
    - The OpenFOAM case is expected under /home/<user>/welding-cases/keyhole-cfd.
    - The solver laserKeyholeVoF is installed in the WSL OpenFOAM user tree.
"""

from __future__ import annotations

import re
import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from tkinter import messagebox, scrolledtext, ttk
from typing import Optional


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
KEYHOLE_CFD_DIR = APP_DIR.parent
OPENFOAM_DIR = KEYHOLE_CFD_DIR / "openfoam"
SYSTEM_DIR = OPENFOAM_DIR / "system"
CONSTANT_DIR = OPENFOAM_DIR / "constant"
CONFIG_DIR = KEYHOLE_CFD_DIR / "config"
MATERIALS_DIR = KEYHOLE_CFD_DIR / "materials"
SCRIPTS_DIR = KEYHOLE_CFD_DIR / "scripts"
SOLVER_DIR = KEYHOLE_CFD_DIR / "solver" / "laserKeyholeVoF"


def wsl_path(win_path: Path) -> str:
    """Convert a Windows path to a /mnt/... WSL path."""
    p = win_path.resolve()
    drive = p.drive.rstrip(":").lower()
    return f"/mnt/{drive}{p.as_posix()[2:]}"


# -----------------------------------------------------------------------------
# Minimal YAML editor (preserves comments and ordering)
# -----------------------------------------------------------------------------
def get_yaml_value(text: str, section: str, key: str) -> Optional[str]:
    """Return the value string for ``key`` inside ``section``."""
    lines = text.splitlines()
    in_section = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f"{section}:"):
            in_section = True
            continue
        if in_section:
            # Section ends when a non-indented (or less-indented) line is found
            if line and not line.startswith("  "):
                break
            m = re.match(rf"\s*{re.escape(key)}:\s*(.*)", stripped)
            if m:
                return m.group(1).strip()
    return None


def set_yaml_value(text: str, section: str, key: str, value: str) -> str:
    """Replace the value for ``key`` inside ``section`` and return new text."""
    lines = text.splitlines()
    in_section = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"{section}:"):
            in_section = True
            continue
        if in_section:
            if line and not line.startswith("  "):
                break
            if re.match(rf"{re.escape(key)}:\s*", stripped):
                lines[i] = re.sub(rf"(\s*{re.escape(key)}:\s*).*", rf"\1{value}", line)
                break
    # Preserve trailing newline
    out = "\n".join(lines)
    if text.endswith("\n"):
        out += "\n"
    return out


# -----------------------------------------------------------------------------
# Runner command builder
# -----------------------------------------------------------------------------
@dataclass
class AppConfig:
    wsl_distro: str
    wsl_case: str
    wsl_solver_root: str

    def wsl_bash(self, command: str) -> list[str]:
        return ["wsl", "-d", self.wsl_distro, "-e", "bash", "-lc", command]


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
class LaserKeyholeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Laser Keyhole Welding CFD - OpenFOAM Runner")
        root.geometry("900x700")
        root.minsize(700, 500)

        self.wsl_distro_var = tk.StringVar(value="Ubuntu-24.04")
        self.wsl_case_var = tk.StringVar(value="/home/nmcal/welding-cases/keyhole-cfd")
        self.wsl_solver_root_var = tk.StringVar(
            value="/home/nmcal/OpenFOAM/nmcal-v2306/applications/solvers"
        )

        self.values = {
            "laser_power": tk.StringVar(value="3000.0"),
            "laser_speed": tk.StringVar(value="0.015"),
            "laser_w0": tk.StringVar(value="150.0e-6"),
            "time_end": tk.StringVar(value="5e-4"),
            "time_max_delta_t": tk.StringVar(value="5e-6"),
            "time_max_co": tk.StringVar(value="0.4"),
            "time_max_alpha_co": tk.StringVar(value="0.4"),
            "time_write_interval": tk.StringVar(value="5e-5"),
            "parallel_cores": tk.StringVar(value="8"),
        }

        self.process: Optional[subprocess.Popen] = None
        self.queue: Queue = Queue()
        self.stop_event = threading.Event()
        self.run_state = tk.StringVar(value="idle")

        self.load_config_from_yaml()
        self.build_ui()
        self.poll_queue()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def build_ui(self):
        pad = {"padx": 6, "pady": 6}

        # Top settings
        settings = ttk.LabelFrame(self.root, text="WSL / Paths", padding=10)
        settings.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(settings, text="WSL distro:").grid(row=0, column=0, sticky=tk.W, **pad)
        ttk.Entry(settings, textvariable=self.wsl_distro_var, width=20).grid(
            row=0, column=1, sticky=tk.W, **pad
        )

        ttk.Label(settings, text="WSL case dir:").grid(row=0, column=2, sticky=tk.W, **pad)
        ttk.Entry(settings, textvariable=self.wsl_case_var, width=40).grid(
            row=0, column=3, sticky=tk.EW, **pad
        )

        ttk.Label(settings, text="Solver root:").grid(row=1, column=0, sticky=tk.W, **pad)
        ttk.Entry(settings, textvariable=self.wsl_solver_root_var, width=60).grid(
            row=1, column=1, columnspan=3, sticky=tk.EW, **pad
        )

        settings.columnconfigure(3, weight=1)

        # Parameters
        params = ttk.LabelFrame(self.root, text="Simulation parameters", padding=10)
        params.pack(fill=tk.X, padx=10, pady=5)

        row = 0
        for label, key, width in [
            ("Laser power [W]", "laser_power", 12),
            ("Travel speed [m/s]", "laser_speed", 12),
            ("Focus radius w0 [m]", "laser_w0", 12),
            ("End time [s]", "time_end", 12),
            ("Max deltaT [s]", "time_max_delta_t", 12),
            ("Max Courant", "time_max_co", 8),
            ("Max alpha Courant", "time_max_alpha_co", 8),
            ("Write interval [s]", "time_write_interval", 12),
            ("MPI cores", "parallel_cores", 8),
        ]:
            ttk.Label(params, text=label).grid(row=row, column=0, sticky=tk.W, **pad)
            ttk.Entry(params, textvariable=self.values[key], width=width).grid(
                row=row, column=1, sticky=tk.W, **pad
            )
            row += 1

        # Buttons
        btns = ttk.Frame(self.root, padding=10)
        btns.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btns, text="Save config", command=self.on_save_config).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(btns, text="Rebuild solver", command=self.on_rebuild_solver).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(btns, text="Configure + Decompose", command=self.on_configure_decompose).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(btns, text="Run simulation", command=self.on_run).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(btns, text="Stop", command=self.on_stop).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Reconstruct latest", command=self.on_reconstruct).pack(
            side=tk.LEFT, padx=3
        )

        # Status
        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding=4)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_bar, textvariable=self.run_state).pack(side=tk.LEFT)

        # Log
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=20, state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def log(self, message: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def error(self, message: str):
        self.log(f"ERROR: {message}")
        messagebox.showerror("Error", message)

    def info(self, message: str):
        self.log(f"INFO: {message}")

    # ------------------------------------------------------------------
    # Config loading / saving
    # ------------------------------------------------------------------
    def master_yaml_path(self) -> Path:
        return CONFIG_DIR / "simulation_master.yaml"

    def load_config_from_yaml(self):
        path = self.master_yaml_path()
        if not path.exists():
            self.error(f"Master config not found: {path}")
            return
        text = path.read_text(encoding="utf-8")
        try:
            power = get_yaml_value(text, "laser", "power")
            speed = get_yaml_value(text, "laser", "travel_speed")
            w0 = get_yaml_value(text, "laser", "focus_radius_w0")
            end = get_yaml_value(text, "time", "end")
            max_dt = get_yaml_value(text, "time", "max_delta_t")
            max_co = get_yaml_value(text, "time", "max_courant")
            max_alpha_co = get_yaml_value(text, "time", "max_alpha_courant")
            write = get_yaml_value(text, "time", "write_interval")
            cores = get_yaml_value(text, "parallel", "num_processors")

            if power is not None:
                self.values["laser_power"].set(power)
            if speed is not None:
                self.values["laser_speed"].set(speed)
            if w0 is not None:
                self.values["laser_w0"].set(w0)
            if end is not None:
                self.values["time_end"].set(end)
            if max_dt is not None:
                self.values["time_max_delta_t"].set(max_dt)
            if max_co is not None:
                self.values["time_max_co"].set(max_co)
            if max_alpha_co is not None:
                self.values["time_max_alpha_co"].set(max_alpha_co)
            if write is not None:
                self.values["time_write_interval"].set(write)
            if cores is not None:
                self.values["parallel_cores"].set(cores)
        except Exception as exc:
            self.error(f"Failed to parse YAML: {exc}")

    def save_config_to_yaml(self) -> bool:
        path = self.master_yaml_path()
        if not path.exists():
            self.error(f"Master config not found: {path}")
            return False
        try:
            text = path.read_text(encoding="utf-8")
            text = set_yaml_value(text, "laser", "power", self.values["laser_power"].get())
            text = set_yaml_value(text, "laser", "travel_speed", self.values["laser_speed"].get())
            text = set_yaml_value(text, "laser", "focus_radius_w0", self.values["laser_w0"].get())
            text = set_yaml_value(text, "time", "end", self.values["time_end"].get())
            text = set_yaml_value(text, "time", "max_delta_t", self.values["time_max_delta_t"].get())
            text = set_yaml_value(text, "time", "max_courant", self.values["time_max_co"].get())
            text = set_yaml_value(text, "time", "max_alpha_courant", self.values["time_max_alpha_co"].get())
            text = set_yaml_value(text, "time", "write_interval", self.values["time_write_interval"].get())
            text = set_yaml_value(text, "parallel", "num_processors", self.values["parallel_cores"].get())
            path.write_text(text, encoding="utf-8")
            self.info(f"Saved {path}")
            return True
        except Exception as exc:
            self.error(f"Failed to save config: {exc}")
            return False

    # ------------------------------------------------------------------
    # WSL command execution
    # ------------------------------------------------------------------
    def run_command(self, command: str, *, state_text: str = "running"):
        if self.process is not None and self.process.poll() is None:
            self.error("Another command is already running.")
            return

        cfg = AppConfig(
            wsl_distro=self.wsl_distro_var.get(),
            wsl_case=self.wsl_case_var.get(),
            wsl_solver_root=self.wsl_solver_root_var.get(),
        )

        self.stop_event.clear()
        self.run_state.set(state_text)
        self.log(f"$ {command[:200]}...")

        def target():
            try:
                proc = subprocess.Popen(
                    cfg.wsl_bash(command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                self.process = proc
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    self.queue.put(("line", line.rstrip()))
                    if self.stop_event.is_set():
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                code = proc.wait()
                self.queue.put(("done", code))
            except Exception as exc:
                self.queue.put(("error", str(exc)))
            finally:
                self.process = None

        threading.Thread(target=target, daemon=True).start()

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "line":
                    self.log(payload)
                elif kind == "done":
                    self.run_state.set("idle" if payload == 0 else f"failed (code {payload})")
                    self.log(f"Command finished with exit code {payload}")
                elif kind == "error":
                    self.run_state.set("error")
                    self.error(payload)
        except Empty:
            pass
        self.root.after(100, self.poll_queue)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def on_save_config(self):
        self.save_config_to_yaml()

    def _configure_command(self, cfg: AppConfig) -> str:
        """Return bash snippet that copies OpenFOAM files into the WSL case and runs configure_case.py."""
        wsl_case = cfg.wsl_case
        wsl_system = f"{wsl_case}/system"
        wsl_constant = f"{wsl_case}/constant"
        win_system = wsl_path(SYSTEM_DIR)
        win_constant = wsl_path(CONSTANT_DIR)
        wsl_cfg = wsl_path(CONFIG_DIR)
        wsl_mat = wsl_path(MATERIALS_DIR)
        wsl_script = wsl_path(SCRIPTS_DIR / "configure_case.py")
        of_env = "source /usr/lib/openfoam/openfoam2306/etc/bashrc"
        return (
            f"{of_env} && "
            f"mkdir -p {wsl_system} {wsl_constant} {wsl_case}/scripts && "
            f"cp -r {wsl_cfg} {wsl_case}/ && "
            f"cp -r {wsl_mat} {wsl_case}/ && "
            f"cp {wsl_script} {wsl_case}/scripts/ && "
            f"cp {win_system}/* {wsl_system}/ && "
            f"cp {win_constant}/* {wsl_constant}/ && "
            f"python3 {wsl_case}/scripts/configure_case.py "
            f"--config {wsl_case}/config/simulation_master.yaml "
            f"--case-dir {wsl_case}"
        )

    def on_rebuild_solver(self):
        cfg = AppConfig(
            wsl_distro=self.wsl_distro_var.get(),
            wsl_case=self.wsl_case_var.get(),
            wsl_solver_root=self.wsl_solver_root_var.get(),
        )
        wsl_src = wsl_path(SOLVER_DIR)
        dest = f"{cfg.wsl_solver_root}/laserKeyholeVoF"
        of_env = "source /usr/lib/openfoam/openfoam2306/etc/bashrc"
        cmd = (
            f"{of_env} && "
            f"rm -rf {dest} && "
            f"cp -r {wsl_src} {dest} && "
            f"cd {dest} && wmake"
        )
        self.run_command(cmd, state_text="rebuilding solver")

    def on_configure_decompose(self):
        if not self.save_config_to_yaml():
            return
        try:
            _ = int(self.values["parallel_cores"].get())
        except ValueError:
            self.error("MPI cores must be an integer")
            return

        cfg = AppConfig(
            wsl_distro=self.wsl_distro_var.get(),
            wsl_case=self.wsl_case_var.get(),
            wsl_solver_root=self.wsl_solver_root_var.get(),
        )
        command = self._configure_command(cfg) + f" && cd {cfg.wsl_case} && decomposePar -force"
        self.run_command(command, state_text="configuring + decomposing")

    def on_run(self):
        if not self.save_config_to_yaml():
            return
        try:
            nprocs = int(self.values["parallel_cores"].get())
        except ValueError:
            self.error("MPI cores must be an integer")
            return

        cfg = AppConfig(
            wsl_distro=self.wsl_distro_var.get(),
            wsl_case=self.wsl_case_var.get(),
            wsl_solver_root=self.wsl_solver_root_var.get(),
        )

        solver_cmd = f"mpirun --oversubscribe -np {nprocs} laserKeyholeVoF -parallel"
        command = (
            f"{self._configure_command(cfg)} && "
            f"cd {cfg.wsl_case} && "
            f"decomposePar -force && "
            f"{solver_cmd}"
        )
        self.run_command(command, state_text="running simulation")

    def on_stop(self):
        self.stop_event.set()
        if self.process is not None:
            self.process.terminate()
            self.log("Sent terminate. Use 'Run' again if it does not stop.")
        self.run_state.set("stopping")

    def on_reconstruct(self):
        cfg = AppConfig(
            wsl_distro=self.wsl_distro_var.get(),
            wsl_case=self.wsl_case_var.get(),
            wsl_solver_root=self.wsl_solver_root_var.get(),
        )
        of_env = "source /usr/lib/openfoam/openfoam2306/etc/bashrc"
        command = f"{of_env} && cd {cfg.wsl_case} && reconstructPar -latestTime"
        self.run_command(command, state_text="reconstructing")


def main():
    root = tk.Tk()
    app = LaserKeyholeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
