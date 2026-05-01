from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class UIResult:
    value: Optional[str]
    cancelled: bool = False


def _root() -> tk.Tk:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def info(msg: str, title: str = "Info") -> None:
    root = _root()
    messagebox.showinfo(title, msg, parent=root)
    root.destroy()


def warn(msg: str, title: str = "Cancelled") -> None:
    root = _root()
    messagebox.showwarning(title, msg, parent=root)
    root.destroy()


def pick_file(title: str, initial_dir: Path, filetypes: Iterable[tuple[str, str]]):
    root = _root()
    path = filedialog.askopenfilename(
        title=title, initialdir=str(initial_dir), filetypes=list(filetypes)
    )
    root.destroy()
    if not path:
        return UIResult(None, cancelled=True)
    return UIResult(path)


def pick_dir(title: str, initial_dir: Path):
    root = _root()
    path = filedialog.askdirectory(title=title, initialdir=str(initial_dir))
    root.destroy()
    if not path:
        return UIResult(None, cancelled=True)
    return UIResult(path)


def save_file(title: str, initial_dir: Path, defaultextension: str, filetypes):
    root = _root()
    path = filedialog.asksaveasfilename(
        title=title,
        initialdir=str(initial_dir),
        defaultextension=defaultextension,
        filetypes=list(filetypes),
    )
    root.destroy()
    if not path:
        return UIResult(None, cancelled=True)
    return UIResult(path)


def ask_str(prompt: str, title: str, initial: str = "") -> UIResult:
    root = _root()
    value = simpledialog.askstring(title, prompt, initialvalue=initial, parent=root)
    root.destroy()
    if value is None:
        return UIResult(None, cancelled=True)
    return UIResult(value)


def ask_int(prompt: str, title: str, initial: int, min_value: int | None = None):
    root = _root()
    value = simpledialog.askinteger(
        title, prompt, initialvalue=initial, minvalue=min_value, parent=root
    )
    root.destroy()
    if value is None:
        return UIResult(None, cancelled=True)
    return UIResult(str(value))


def ask_float(
    prompt: str, title: str, initial: float, min_value: float | None = None
):
    root = _root()
    value = simpledialog.askfloat(
        title, prompt, initialvalue=initial, minvalue=min_value, parent=root
    )
    root.destroy()
    if value is None:
        return UIResult(None, cancelled=True)
    return UIResult(str(value))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
