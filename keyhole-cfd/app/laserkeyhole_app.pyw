#!/usr/bin/env python3
"""
laserkeyhole_app.pyw - Polished Windows GUI for laser keyhole OpenFOAM simulations.

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
from tkinter import font, messagebox, scrolledtext, ttk
from typing import Optional

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
KEYHOLE_CFD_DIR = APP_DIR.parent
OPENFOAM_DIR = KEYHOLE_CFD_DIR / "openfoam"
SYSTEM_DIR = OPENFOAM_DIR / "system"
CONSTANT_DIR = OPENFOAM_DIR / "constant"
ZERO_DIR = OPENFOAM_DIR / "0"
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
    lines = text.splitlines()
    in_section = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f"{section}:"):
            in_section = True
            continue
        if in_section:
            if line and not line.startswith("  "):
                break
            m = re.match(rf"\s*{re.escape(key)}:\s*(.*)", stripped)
            if m:
                return m.group(1).strip()
    return None


def set_yaml_value(text: str, section: str, key: str, value: str) -> str:
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
                lines[i] = re.sub(rf"(\s*{re.escape(key)}:\s*).*", rf"\g<1>{value}", line)
                break
    out = "\n".join(lines)
    if text.endswith("\n"):
        out += "\n"
    return out


# -----------------------------------------------------------------------------
# Tooltip helper
# -----------------------------------------------------------------------------
class Tooltip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._enter)
        widget.bind("<Leave>", self._leave)

    def _enter(self, _=None):
        x, y, _, _ = self.widget.bbox("insert") if isinstance(self.widget, tk.Text) else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 25
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            background="#ffffe0",
            font=("Segoe UI", 9),
            padx=4,
            pady=2,
        )
        label.pack()

    def _leave(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


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
        root.title("Laser Keyhole Welding CFD")
        root.geometry("1000x780")
        root.minsize(850, 650)

        # Variables
        self.wsl_distro_var = tk.StringVar(value="Ubuntu-24.04")
        self.wsl_case_var = tk.StringVar(value="/home/nmcal/welding-cases/keyhole-cfd")
        self.wsl_solver_root_var = tk.StringVar(
            value="/home/nmcal/OpenFOAM/nmcal-v2306/applications/solvers"
        )
        self.run_setfields_var = tk.BooleanVar(value=True)

        self.values = {
            "laser_power": tk.StringVar(value="3000.0"),
            "laser_speed": tk.StringVar(value="0.015"),
            "laser_w0": tk.StringVar(value="150.0e-6"),
            "time_end": tk.StringVar(value="5.0e-4"),
            "time_initial_delta_t": tk.StringVar(value="1.0e-7"),
            "time_max_delta_t": tk.StringVar(value="5.0e-6"),
            "time_max_co": tk.StringVar(value="0.4"),
            "time_max_alpha_co": tk.StringVar(value="0.4"),
            "time_write_interval": tk.StringVar(value="5.0e-5"),
            "parallel_cores": tk.StringVar(value="8"),
        }

        self.process: Optional[subprocess.Popen] = None
        self.queue: Queue = Queue()
        self.stop_event = threading.Event()
        self.run_state = tk.StringVar(value="Ready")

        self._setup_styles()
        self._build_menu()
        self._build_header()
        self._build_main_ui()

        self.load_config_from_yaml()
        self.poll_queue()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Segoe UI", size=10)
        self.header_font = font.Font(family="Segoe UI", size=18, weight="bold")
        self.subheader_font = font.Font(family="Segoe UI", size=10, weight="normal")
        style.configure("Header.TLabel", font=self.header_font)
        style.configure("Subheader.TLabel", font=self.subheader_font, foreground="#555")
        style.configure("Group.TLabelframe", font=font.Font(family="Segoe UI", size=10, weight="bold"))
        style.configure("Run.TButton", font=font.Font(family="Segoe UI", size=10, weight="bold"))

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save configuration", command=self.on_save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        sim_menu = tk.Menu(menubar, tearoff=0)
        sim_menu.add_command(label="Configure + Decompose", command=self.on_configure_decompose)
        sim_menu.add_command(label="Run simulation", command=self.on_run)
        sim_menu.add_command(label="Stop", command=self.on_stop)
        sim_menu.add_separator()
        sim_menu.add_command(label="Reconstruct latest", command=self.on_reconstruct)
        sim_menu.add_command(label="Rebuild solver", command=self.on_rebuild_solver)
        menubar.add_cascade(label="Simulation", menu=sim_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Laser Keyhole Welding CFD Runner\n\n"
            "A standalone Windows GUI for configuring and running the custom "
            "OpenFOAM 2306 solver laserKeyholeVoF inside WSL2.",
        )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _build_header(self):
        header = ttk.Frame(self.root, padding=(15, 10))
        header.pack(fill=tk.X)

        ttk.Label(header, text="Laser Keyhole Welding CFD", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Configure, decompose, and run OpenFOAM 2306 laser keyhole simulations",
            style="Subheader.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

    # ------------------------------------------------------------------
    # Main UI
    # ------------------------------------------------------------------
    def _build_main_ui(self):
        notebook = ttk.Notebook(self.root, padding=5)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # --- Setup tab ---
        setup_outer = ttk.Frame(notebook)
        setup_outer.rowconfigure(0, weight=1)
        setup_outer.columnconfigure(0, weight=1)

        setup_canvas = tk.Canvas(setup_outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(setup_outer, orient=tk.VERTICAL, command=setup_canvas.yview)
        self.setup_frame = ttk.Frame(setup_canvas, padding=10)
        self.setup_frame.bind(
            "<Configure>",
            lambda e: setup_canvas.configure(scrollregion=setup_canvas.bbox("all")),
        )
        setup_canvas.create_window((0, 0), window=self.setup_frame, anchor=tk.NW, width=940)
        setup_canvas.configure(yscrollcommand=scrollbar.set)
        setup_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        notebook.add(setup_outer, text="Setup", padding=2)
        notebook.add(ttk.Frame(notebook, name="log"), text="Log")

        # Keep reference to log frame
        self.log_parent = notebook.nametowidget("log")

        # Build setup sections
        self._build_wsl_section(self.setup_frame)
        self._build_laser_section(self.setup_frame)
        self._build_time_section(self.setup_frame)
        self._build_parallel_section(self.setup_frame)
        self._build_actions_section(self.setup_frame)

        # Build log
        self._build_log_section(self.log_parent)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------
    def _entry_with_label(
        self,
        parent,
        row,
        label,
        var,
        tooltip="",
        width=15,
    ):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=4)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=4)
        if tooltip:
            Tooltip(entry, tooltip)
        return entry

    def _build_wsl_section(self, parent):
        frame = ttk.LabelFrame(parent, text="WSL Environment", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="WSL distro:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Entry(frame, textvariable=self.wsl_distro_var, width=20).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=4
        )

        ttk.Label(frame, text="WSL case dir:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Entry(frame, textvariable=self.wsl_case_var, width=55).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=4
        )
        Tooltip(
            frame,
            "Directory in WSL that contains the OpenFOAM case (0/, constant/, system/)",
        )

        ttk.Label(frame, text="Solver root:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Entry(frame, textvariable=self.wsl_solver_root_var, width=55).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=4
        )

    def _build_laser_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Laser Parameters", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        self._entry_with_label(
            frame, 0, "Power [W]", self.values["laser_power"], "Laser power in watts"
        )
        self._entry_with_label(
            frame,
            1,
            "Travel speed [m/s]",
            self.values["laser_speed"],
            "Laser travel speed along the weld direction",
        )
        self._entry_with_label(
            frame,
            2,
            "Focus radius w0 [m]",
            self.values["laser_w0"],
            "1/e² beam radius at focus",
        )

    def _build_time_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Time Control", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        row = 0
        self._entry_with_label(
            frame, row, "End time [s]", self.values["time_end"], "Total simulation end time"
        )
        row += 1
        self._entry_with_label(
            frame,
            row,
            "Initial deltaT [s]",
            self.values["time_initial_delta_t"],
            "First time step",
        )
        row += 1
        self._entry_with_label(
            frame, row, "Max deltaT [s]", self.values["time_max_delta_t"], "Largest allowed time step"
        )
        row += 1
        self._entry_with_label(
            frame, row, "Max Courant", self.values["time_max_co"], "Max velocity Courant number"
        )
        row += 1
        self._entry_with_label(
            frame, row, "Max alpha Courant", self.values["time_max_alpha_co"], "Max VOF interface Courant number"
        )
        row += 1
        self._entry_with_label(
            frame,
            row,
            "Write interval [s]",
            self.values["time_write_interval"],
            "How often results are written to disk",
        )

    def _build_parallel_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Parallel Execution", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="MPI cores:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Entry(frame, textvariable=self.values["parallel_cores"], width=8).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=4
        )
        ttk.Label(
            frame,
            text="Use 8 cores for the current 600k-cell mesh; fewer cores run slower.",
            foreground="#666",
        ).grid(row=0, column=2, sticky=tk.W, padx=10)

    def _build_actions_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        self.save_button = ttk.Button(frame, text="Save configuration", command=self.on_save_config)
        self.save_button.grid(row=0, column=0, padx=5, pady=5)

        self.rebuild_button = ttk.Button(frame, text="Rebuild solver", command=self.on_rebuild_solver)
        self.rebuild_button.grid(row=0, column=1, padx=5, pady=5)

        self.config_button = ttk.Button(
            frame, text="Configure + Decompose", command=self.on_configure_decompose
        )
        self.config_button.grid(row=0, column=2, padx=5, pady=5)

        self.run_button = ttk.Button(frame, text="Run simulation", command=self.on_run, style="Run.TButton")
        self.run_button.grid(row=0, column=3, padx=5, pady=5)

        self.stop_button = ttk.Button(frame, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=4, padx=5, pady=5)

        self.recon_button = ttk.Button(frame, text="Reconstruct latest", command=self.on_reconstruct)
        self.recon_button.grid(row=0, column=5, padx=5, pady=5)

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
            row=1, column=0, columnspan=6, sticky=tk.EW, pady=(8, 4)
        )

        self.setfields_check = ttk.Checkbutton(
            frame,
            text="Run setFields before decompose (required for a fresh metal/gas split)",
            variable=self.run_setfields_var,
        )
        self.setfields_check.grid(row=2, column=0, columnspan=6, sticky=tk.W, padx=5)

    def _build_log_section(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(parent, text="Solver Output", padding=5)
        log_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=20,
            state=tk.DISABLED,
            font=("Consolas", 9),
        )
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)

        # Status / progress
        status_frame = ttk.Frame(parent, relief=tk.SUNKEN, padding=4)
        status_frame.grid(row=1, column=0, sticky=tk.EW, padx=5, pady=(0, 5))
        status_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(status_frame, textvariable=self.run_state)
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=120)
        self.progress.grid(row=0, column=1, sticky=tk.E, padx=(10, 0))

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
        self.run_state.set("Error")
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
            mapping = {
                ("laser", "power"): "laser_power",
                ("laser", "travel_speed"): "laser_speed",
                ("laser", "focus_radius_w0"): "laser_w0",
                ("time", "end"): "time_end",
                ("time", "initial_delta_t"): "time_initial_delta_t",
                ("time", "max_delta_t"): "time_max_delta_t",
                ("time", "max_courant"): "time_max_co",
                ("time", "max_alpha_courant"): "time_max_alpha_co",
                ("time", "write_interval"): "time_write_interval",
                ("parallel", "num_processors"): "parallel_cores",
            }
            for (section, key), var_key in mapping.items():
                value = get_yaml_value(text, section, key)
                if value is not None:
                    self.values[var_key].set(value)
        except Exception as exc:
            self.error(f"Failed to parse YAML: {exc}")

    def validate_numeric(self) -> bool:
        bad = []
        for key, var in self.values.items():
            try:
                float(var.get().strip())
            except ValueError:
                bad.append(key.replace("_", " "))
        if bad:
            self.error(f"These fields must be numeric: {', '.join(bad)}")
            return False
        try:
            cores = int(self.values["parallel_cores"].get())
            if cores < 1:
                raise ValueError
        except ValueError:
            self.error("MPI cores must be a positive integer")
            return False
        return True

    def save_config_to_yaml(self) -> bool:
        if not self.validate_numeric():
            return False
        path = self.master_yaml_path()
        if not path.exists():
            self.error(f"Master config not found: {path}")
            return False
        try:
            text = path.read_text(encoding="utf-8")
            updates = [
                ("laser", "power", self.values["laser_power"].get().strip()),
                ("laser", "travel_speed", self.values["laser_speed"].get().strip()),
                ("laser", "focus_radius_w0", self.values["laser_w0"].get().strip()),
                ("time", "end", self.values["time_end"].get().strip()),
                ("time", "initial_delta_t", self.values["time_initial_delta_t"].get().strip()),
                ("time", "max_delta_t", self.values["time_max_delta_t"].get().strip()),
                ("time", "max_courant", self.values["time_max_co"].get().strip()),
                ("time", "max_alpha_courant", self.values["time_max_alpha_co"].get().strip()),
                ("time", "write_interval", self.values["time_write_interval"].get().strip()),
                ("parallel", "num_processors", self.values["parallel_cores"].get().strip()),
            ]
            for section, key, value in updates:
                text = set_yaml_value(text, section, key, value)
            path.write_text(text, encoding="utf-8")
            self.info(f"Saved configuration to {path}")
            return True
        except Exception as exc:
            self.error(f"Failed to save config: {exc}")
            return False

    # ------------------------------------------------------------------
    # WSL command execution
    # ------------------------------------------------------------------
    def set_running(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        for btn in (self.save_button, self.config_button, self.run_button, self.rebuild_button, self.recon_button):
            btn.configure(state=state)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        if running:
            self.run_state.set("Running ...")
            self.progress.start(10)
        else:
            self.progress.stop()

    def run_command(self, command: str, *, state_text: str = "Running"):
        if self.process is not None and self.process.poll() is None:
            self.error("Another command is already running.")
            return

        cfg = AppConfig(
            wsl_distro=self.wsl_distro_var.get(),
            wsl_case=self.wsl_case_var.get(),
            wsl_solver_root=self.wsl_solver_root_var.get(),
        )

        self.stop_event.clear()
        self.log(f"$ {command[:200]}...")
        self.set_running(True)

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
                            proc.wait(timeout=3)
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
                    self.set_running(False)
                    self.run_state.set("Ready" if payload == 0 else f"Failed (code {payload})")
                    self.log(f"Command finished with exit code {payload}")
                elif kind == "error":
                    self.set_running(False)
                    self.run_state.set("Error")
                    self.error(payload)
        except Empty:
            pass
        self.root.after(100, self.poll_queue)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def on_save_config(self):
        self.save_config_to_yaml()

    def _prepare_command(self, cfg: AppConfig) -> str:
        """Return bash snippet that copies OpenFOAM files into the WSL case, configures, and (optionally) sets fields."""
        wsl_case = cfg.wsl_case
        wsl_system = f"{wsl_case}/system"
        wsl_constant = f"{wsl_case}/constant"
        wsl_zero = f"{wsl_case}/0"
        win_system = wsl_path(SYSTEM_DIR)
        win_constant = wsl_path(CONSTANT_DIR)
        win_zero = wsl_path(ZERO_DIR)
        wsl_cfg = wsl_path(CONFIG_DIR)
        wsl_mat = wsl_path(MATERIALS_DIR)
        wsl_script = wsl_path(SCRIPTS_DIR / "configure_case.py")
        of_env = "source /usr/lib/openfoam/openfoam2306/etc/bashrc"

        cmd = (
            f"{of_env} && "
            f"mkdir -p {wsl_system} {wsl_constant} {wsl_zero} {wsl_case}/scripts && "
            f"cp -r {wsl_cfg} {wsl_case}/ && "
            f"cp -r {wsl_mat} {wsl_case}/ && "
            f"cp {wsl_script} {wsl_case}/scripts/ && "
            f"cp {win_system}/* {wsl_system}/ && "
            f"cp {win_constant}/* {wsl_constant}/ && "
            f"cp {win_zero}/* {wsl_zero}/ && "
            f"python3 {wsl_case}/scripts/configure_case.py "
            f"--config {wsl_case}/config/simulation_master.yaml "
            f"--case-dir {wsl_case}"
        )
        if self.run_setfields_var.get():
            cmd += f" && cd {wsl_case} && setFields"
        return cmd

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
        self.run_command(cmd, state_text="Rebuilding solver")

    def on_configure_decompose(self):
        if not self.save_config_to_yaml():
            return
        cfg = AppConfig(
            wsl_distro=self.wsl_distro_var.get(),
            wsl_case=self.wsl_case_var.get(),
            wsl_solver_root=self.wsl_solver_root_var.get(),
        )
        command = self._prepare_command(cfg) + f" && cd {cfg.wsl_case} && decomposePar -force"
        self.run_command(command, state_text="Configuring + decomposing")

    def on_run(self):
        if not self.save_config_to_yaml():
            return
        nprocs = int(self.values["parallel_cores"].get())
        cfg = AppConfig(
            wsl_distro=self.wsl_distro_var.get(),
            wsl_case=self.wsl_case_var.get(),
            wsl_solver_root=self.wsl_solver_root_var.get(),
        )

        solver_cmd = f"mpirun --oversubscribe -np {nprocs} laserKeyholeVoF -parallel"
        command = (
            f"{self._prepare_command(cfg)} && "
            f"cd {cfg.wsl_case} && "
            f"decomposePar -force && "
            f"{solver_cmd}"
        )
        self.run_command(command, state_text="Running simulation")

    def on_stop(self):
        self.stop_event.set()
        if self.process is not None:
            self.process.terminate()
            self.log("Sent terminate signal. Use 'Run' again after it stops.")
        self.run_state.set("Stopping ...")

    def on_reconstruct(self):
        cfg = AppConfig(
            wsl_distro=self.wsl_distro_var.get(),
            wsl_case=self.wsl_case_var.get(),
            wsl_solver_root=self.wsl_solver_root_var.get(),
        )
        of_env = "source /usr/lib/openfoam/openfoam2306/etc/bashrc"
        command = f"{of_env} && cd {cfg.wsl_case} && reconstructPar -latestTime"
        self.run_command(command, state_text="Reconstructing latest time")


def main():
    root = tk.Tk()
    app = LaserKeyholeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
