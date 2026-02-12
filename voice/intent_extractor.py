import difflib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyautogui
import pygetwindow as gw


@dataclass
class VoiceCommand:
    raw: str
    action: str
    args: Tuple[str, ...] = ()


class VoiceCommandProcessor:
    """
    Parse and execute voice commands. Designed to be forgiving and pragmatic.
    """

    def __init__(self):
        self.pending_confirmation: Optional[str] = None
        self.last_opened_folder: Path = Path.home()
        self.app_index = AppIndex()

    def split_commands(self, text: str) -> List[str]:
        text = text.lower().strip()
        text = text.replace(";", " ")
        parts = re.split(r"\b(?:and then|then|and|after that|,)\b", text)
        return [p.strip() for p in parts if p.strip()]

    def parse_command(self, text: str) -> VoiceCommand:
        t = text.lower().strip()

        if t in {"confirm", "yes", "confirm shutdown", "confirm sleep"}:
            return VoiceCommand(raw=text, action="confirm")
        if t in {"cancel", "abort", "no", "cancel shutdown"}:
            return VoiceCommand(raw=text, action="cancel")

        # System controls
        if "shutdown" in t:
            return VoiceCommand(raw=text, action="shutdown")
        if "sleep" in t:
            return VoiceCommand(raw=text, action="sleep")
        if "switch tab" in t or "next tab" in t:
            return VoiceCommand(raw=text, action="tab_next")
        if "previous tab" in t or "back tab" in t:
            return VoiceCommand(raw=text, action="tab_prev")
        if "minimize all" in t:
            return VoiceCommand(raw=text, action="minimize_all")
        if "minimize" in t:
            return VoiceCommand(raw=text, action="minimize")
        if "maximize" in t:
            return VoiceCommand(raw=text, action="maximize")
        if "close all" in t:
            return VoiceCommand(raw=text, action="close_all")
        if "close" in t:
            return VoiceCommand(raw=text, action="close")

        # Editing and typing
        if t.startswith("type "):
            return VoiceCommand(raw=text, action="type_text", args=(text[5:].strip(),))
        if "bold" in t:
            return VoiceCommand(raw=text, action="bold")
        if "italic" in t:
            return VoiceCommand(raw=text, action="italic")
        if "underline" in t:
            return VoiceCommand(raw=text, action="underline")
        if "align left" in t:
            return VoiceCommand(raw=text, action="align_left")
        if "align center" in t or "centre" in t:
            return VoiceCommand(raw=text, action="align_center")
        if "align right" in t:
            return VoiceCommand(raw=text, action="align_right")
        if "justify" in t:
            return VoiceCommand(raw=text, action="align_justify")
        if "increase font size" in t or "font size up" in t:
            return VoiceCommand(raw=text, action="font_up")
        if "decrease font size" in t or "font size down" in t:
            return VoiceCommand(raw=text, action="font_down")
        m = re.search(r"font size (\d+)", t)
        if m:
            return VoiceCommand(raw=text, action="font_size", args=(m.group(1),))
        m = re.search(r"font (?:change|set) to (.+)", t)
        if m:
            return VoiceCommand(raw=text, action="font_name", args=(m.group(1).strip(),))

        if "copy" in t:
            return VoiceCommand(raw=text, action="copy")
        if "cut" in t:
            return VoiceCommand(raw=text, action="cut")
        if "paste" in t:
            return VoiceCommand(raw=text, action="paste")
        if "select all" in t:
            return VoiceCommand(raw=text, action="select_all")

        # File / folder operations
        if t.startswith("open folder"):
            path = self._extract_path(text, prefix="open folder")
            return VoiceCommand(raw=text, action="open_folder", args=(path,))
        if t.startswith("open file"):
            path = self._extract_path(text, prefix="open file")
            return VoiceCommand(raw=text, action="open_file", args=(path,))
        if t.startswith("open "):
            target = text[5:].strip()
            match = self.app_index.match(target)
            if match:
                return VoiceCommand(raw=text, action="open_app", args=(match.path, match.display))
            return VoiceCommand(raw=text, action="open_app", args=(target,))
        if t.startswith("create folder"):
            name, path = self._extract_name_and_path(text, prefix="create folder")
            return VoiceCommand(raw=text, action="create_folder", args=(name, path))
        if t.startswith("create file"):
            name, path = self._extract_name_and_path(text, prefix="create file")
            return VoiceCommand(raw=text, action="create_file", args=(name, path))
        if t.startswith("delete file"):
            path = self._extract_path(text, prefix="delete file")
            return VoiceCommand(raw=text, action="delete_file", args=(path,))
        if t.startswith("delete folder"):
            path = self._extract_path(text, prefix="delete folder")
            return VoiceCommand(raw=text, action="delete_folder", args=(path,))
        if t.startswith("rename file"):
            old_name, new_name = self._extract_rename(text, prefix="rename file")
            return VoiceCommand(raw=text, action="rename_file", args=(old_name, new_name))
        if t.startswith("rename folder"):
            old_name, new_name = self._extract_rename(text, prefix="rename folder")
            return VoiceCommand(raw=text, action="rename_folder", args=(old_name, new_name))
        if "backward" in t or "go back" in t:
            return VoiceCommand(raw=text, action="backward")
        if "scroll up" in t:
            return VoiceCommand(raw=text, action="scroll_up")
        if "scroll down" in t:
            return VoiceCommand(raw=text, action="scroll_down")
        m = re.search(r"multi\s*select row (\d+)\s*(?:column|col)\s*(\d+)", t)
        if m:
            return VoiceCommand(raw=text, action="multiselect_row_col", args=(m.group(1), m.group(2)))
        m = re.search(r"select row (\d+)\s*(?:column|col)\s*(\d+)", t)
        if m:
            return VoiceCommand(raw=text, action="select_row_col", args=(m.group(1), m.group(2)))

        return VoiceCommand(raw=text, action="unknown")

    def execute(self, command: VoiceCommand) -> str:
        if command.action == "confirm":
            return self._execute_confirm()
        if command.action == "cancel":
            return self._execute_cancel()

        if command.action == "shutdown":
            self.pending_confirmation = "shutdown"
            return "Pending confirmation: say 'confirm' to shutdown."
        if command.action == "sleep":
            self.pending_confirmation = "sleep"
            return "Pending confirmation: say 'confirm' to sleep."

        # System actions
        if command.action == "tab_next":
            pyautogui.hotkey("ctrl", "tab")
            return "Switched to next tab."
        if command.action == "tab_prev":
            pyautogui.hotkey("ctrl", "shift", "tab")
            return "Switched to previous tab."
        if command.action == "minimize_all":
            pyautogui.hotkey("win", "m")
            return "Minimized all windows."
        if command.action == "minimize":
            pyautogui.hotkey("alt", "space")
            pyautogui.press("n")
            return "Minimized current window."
        if command.action == "maximize":
            pyautogui.hotkey("alt", "space")
            pyautogui.press("x")
            return "Maximized current window."
        if command.action == "close":
            pyautogui.hotkey("alt", "f4")
            return "Closed current window."
        if command.action == "close_all":
            self._close_all_windows()
            return "Closed all visible windows."

        # Text editing
        if command.action == "type_text":
            pyautogui.typewrite(command.args[0])
            return "Typed text."
        if command.action == "bold":
            pyautogui.hotkey("ctrl", "b")
            return "Toggled bold."
        if command.action == "italic":
            pyautogui.hotkey("ctrl", "i")
            return "Toggled italic."
        if command.action == "underline":
            pyautogui.hotkey("ctrl", "u")
            return "Toggled underline."
        if command.action == "align_left":
            pyautogui.hotkey("ctrl", "l")
            return "Aligned left."
        if command.action == "align_center":
            pyautogui.hotkey("ctrl", "e")
            return "Aligned center."
        if command.action == "align_right":
            pyautogui.hotkey("ctrl", "r")
            return "Aligned right."
        if command.action == "align_justify":
            pyautogui.hotkey("ctrl", "j")
            return "Aligned justify."
        if command.action == "font_up":
            pyautogui.hotkey("ctrl", "shift", ">")
            return "Increased font size."
        if command.action == "font_down":
            pyautogui.hotkey("ctrl", "shift", "<")
            return "Decreased font size."
        if command.action == "font_size":
            size = command.args[0]
            # Word: Alt -> H -> F S
            pyautogui.hotkey("alt", "h")
            pyautogui.press("f")
            pyautogui.press("s")
            pyautogui.typewrite(size)
            pyautogui.press("enter")
            return f"Set font size to {size}."
        if command.action == "font_name":
            font = command.args[0]
            pyautogui.hotkey("alt", "h")
            pyautogui.press("f")
            pyautogui.press("f")
            pyautogui.typewrite(font)
            pyautogui.press("enter")
            return f"Set font to {font}."
        if command.action == "copy":
            pyautogui.hotkey("ctrl", "c")
            return "Copied."
        if command.action == "cut":
            pyautogui.hotkey("ctrl", "x")
            return "Cut."
        if command.action == "paste":
            pyautogui.hotkey("ctrl", "v")
            return "Pasted."
        if command.action == "select_all":
            pyautogui.hotkey("ctrl", "a")
            return "Selected all."

        # File and folder operations
        if command.action == "open_folder":
            path = self._resolve_path(command.args[0])
            os.startfile(path)
            self.last_opened_folder = Path(path)
            return f"Opened folder: {path}"
        if command.action == "open_file":
            path = self._resolve_path(command.args[0])
            os.startfile(path)
            return f"Opened file: {path}"
        if command.action == "open_app":
            target = command.args[0]
            if len(command.args) == 2 and Path(target).exists():
                os.startfile(target)
                return f"Opened app: {command.args[1]}"
            self._open_app_by_search(target)
            return f"Opening app: {target}"
        if command.action == "create_folder":
            name, base = command.args
            folder_path = Path(base) / name
            folder_path.mkdir(parents=True, exist_ok=True)
            return f"Created folder: {folder_path}"
        if command.action == "create_file":
            name, base = command.args
            file_path = Path(base) / name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch(exist_ok=True)
            return f"Created file: {file_path}"
        if command.action == "delete_file":
            path = self._resolve_path(command.args[0])
            Path(path).unlink(missing_ok=True)
            return f"Deleted file: {path}"
        if command.action == "delete_folder":
            path = self._resolve_path(command.args[0])
            try:
                Path(path).rmdir()
            except OSError:
                return f"Folder not empty: {path}"
            return f"Deleted folder: {path}"
        if command.action == "rename_file":
            old, new = command.args
            Path(old).rename(new)
            return f"Renamed file to: {new}"
        if command.action == "rename_folder":
            old, new = command.args
            Path(old).rename(new)
            return f"Renamed folder to: {new}"
        if command.action == "backward":
            pyautogui.hotkey("alt", "left")
            return "Navigated back."
        if command.action == "scroll_up":
            pyautogui.scroll(600)
            return "Scrolled up."
        if command.action == "scroll_down":
            pyautogui.scroll(-600)
            return "Scrolled down."
        if command.action == "select_row_col":
            row, col = int(command.args[0]), int(command.args[1])
            self._select_row_column(row, col)
            return f"Selected row {row} column {col}."
        if command.action == "multiselect_row_col":
            row, col = int(command.args[0]), int(command.args[1])
            self._select_row_column(row, col, multiselect=True)
            return f"Multi-selected row {row} column {col}."

        return f"Unrecognized command: {command.raw}"

    def _execute_confirm(self) -> str:
        if self.pending_confirmation == "shutdown":
            subprocess.Popen(["shutdown", "/s", "/t", "5"], shell=False)
            self.pending_confirmation = None
            return "Shutdown confirmed. System will shut down in 5 seconds."
        if self.pending_confirmation == "sleep":
            subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], shell=False)
            self.pending_confirmation = None
            return "Sleep confirmed."
        return "Nothing to confirm."

    def _execute_cancel(self) -> str:
        if self.pending_confirmation == "shutdown":
            subprocess.Popen(["shutdown", "/a"], shell=False)
            self.pending_confirmation = None
            return "Shutdown canceled."
        if self.pending_confirmation == "sleep":
            self.pending_confirmation = None
            return "Sleep canceled."
        return "Nothing to cancel."

    def _extract_path(self, text: str, prefix: str) -> str:
        rest = text[len(prefix):].strip()
        if rest.startswith("\"") and rest.endswith("\""):
            rest = rest[1:-1]
        if not rest:
            return str(self.last_opened_folder)
        return rest

    def _extract_name_and_path(self, text: str, prefix: str) -> Tuple[str, str]:
        rest = text[len(prefix):].strip()
        match = re.search(r"named (.+?)(?: in (.+))?$", rest)
        if match:
            name = match.group(1).strip()
            base = match.group(2).strip() if match.group(2) else str(self.last_opened_folder)
            return name, base
        # Fallback: use whole rest as name
        return rest, str(self.last_opened_folder)

    def _extract_rename(self, text: str, prefix: str) -> Tuple[str, str]:
        rest = text[len(prefix):].strip()
        match = re.search(r"(.+?) to (.+)", rest)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return rest, rest

    def _resolve_path(self, raw: str) -> str:
        if raw.startswith("~"):
            return str(Path(raw).expanduser())
        if raw in {".", "current", "here"}:
            return str(self.last_opened_folder)
        return raw

    def _open_app_by_search(self, name: str) -> None:
        pyautogui.press("win")
        pyautogui.typewrite(name)
        pyautogui.press("enter")

    def _close_all_windows(self) -> None:
        try:
            windows = [w for w in gw.getAllWindows() if w.isVisible and w.title]
            for win in windows:
                try:
                    win.close()
                except Exception:
                    continue
        except Exception:
            pyautogui.hotkey("alt", "f4")

    def _select_row_column(self, row: int, col: int, multiselect: bool = False) -> None:
        # Approximate selection in Explorer Details view:
        # Home -> move down (row-1), then right (col-1), then select.
        pyautogui.press("home")
        for _ in range(max(0, row - 1)):
            pyautogui.press("down")
        for _ in range(max(0, col - 1)):
            pyautogui.press("right")
        if multiselect:
            pyautogui.keyDown("ctrl")
            pyautogui.press("space")
            pyautogui.keyUp("ctrl")
        else:
            pyautogui.press("space")

    def build_grammar_phrases(self, max_apps: int = 500) -> List[str]:
        phrases = {
            "confirm",
            "cancel",
            "shutdown",
            "sleep",
            "switch tab",
            "next tab",
            "previous tab",
            "back tab",
            "minimize",
            "minimize all",
            "maximize",
            "close",
            "close all",
            "copy",
            "cut",
            "paste",
            "select all",
            "scroll up",
            "scroll down",
        }
        app_phrases = self.app_index.build_open_phrases(max_apps=max_apps)
        phrases.update(app_phrases)
        return sorted(phrases)


@dataclass(frozen=True)
class AppMatch:
    display: str
    path: str


class AppIndex:
    def __init__(self) -> None:
        self.apps = self._build_index()

    def _build_index(self) -> Dict[str, AppMatch]:
        apps: Dict[str, AppMatch] = {}
        for shortcut in self._iter_start_menu_shortcuts():
            display = shortcut.stem
            key = self._normalize(display)
            if not key:
                continue
            if key not in apps:
                apps[key] = AppMatch(display=display, path=str(shortcut))
        for alias, target in self._alias_map().items():
            norm_alias = self._normalize(alias)
            norm_target = self._normalize(target)
            if norm_target in apps and norm_alias not in apps:
                apps[norm_alias] = apps[norm_target]
        return apps

    def match(self, query: str) -> Optional[AppMatch]:
        q = self._normalize(query)
        if not q:
            return None
        if q in self.apps:
            return self.apps[q]
        candidates = difflib.get_close_matches(q, self.apps.keys(), n=1, cutoff=0.65)
        if candidates:
            return self.apps[candidates[0]]
        return None

    def build_open_phrases(self, max_apps: int = 500) -> List[str]:
        names = list(self.apps.values())[:max_apps]
        phrases = [f"open {app.display.lower()}" for app in names]
        return phrases

    def _iter_start_menu_shortcuts(self) -> List[Path]:
        roots = [
            Path(os.environ.get("ProgramData", r"C:\ProgramData"))
            / r"Microsoft\Windows\Start Menu\Programs",
            Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming")))
            / r"Microsoft\Windows\Start Menu\Programs",
        ]
        shortcuts: List[Path] = []
        for root in roots:
            if root.exists():
                shortcuts.extend(root.rglob("*.lnk"))
        return shortcuts

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text).strip()
        return text

    def _alias_map(self) -> Dict[str, str]:
        return {
            "chrome": "Google Chrome",
            "google": "Google Chrome",
            "edge": "Microsoft Edge",
            "word": "Word",
            "excel": "Excel",
            "powerpoint": "PowerPoint",
            "notepad": "Notepad",
            "calculator": "Calculator",
            "paint": "Paint",
            "cmd": "Command Prompt",
            "command prompt": "Command Prompt",
            "powershell": "Windows PowerShell",
            "vs code": "Visual Studio Code",
        }
