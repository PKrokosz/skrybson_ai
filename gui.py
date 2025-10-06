"""Simple Tkinter GUI for running the transcription pipeline."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class TranscriptionWorker(threading.Thread):
    """Background thread that executes the transcription script."""

    def __init__(self, env: Dict[str, str], output_queue: "queue.Queue[str]") -> None:
        super().__init__(daemon=True)
        self._env = env
        self._output_queue = output_queue
        self._process: Optional[subprocess.Popen[str]] = None

    def run(self) -> None:  # pragma: no cover - Tkinter runner
        command = [sys.executable, "-u", "transcribe.py"]
        try:
            self._output_queue.put("[info] Startuję transkrypcję...\n")
            self._process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=self._env,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._output_queue.put(line)
            return_code = self._process.wait()
            self._output_queue.put(f"[info] Zakończono z kodem {return_code}.\n")
        except FileNotFoundError as exc:
            self._output_queue.put(f"[error] Nie znaleziono transcribe.py: {exc}\n")
        except Exception as exc:  # pragma: no cover - defensive
            self._output_queue.put(f"[error] Błąd podczas uruchomienia: {exc}\n")

    def terminate(self) -> None:  # pragma: no cover - Tkinter runner
        if self._process and self._process.poll() is None:
            self._process.terminate()


class TranscriptionGUI(tk.Tk):  # pragma: no cover - Tkinter runner
    """Main application window."""

    POLL_INTERVAL_MS = 150

    def __init__(self) -> None:
        super().__init__()
        self.title("Skrybson AI — Transkrypcja")
        self.geometry("760x540")
        self._worker: Optional[TranscriptionWorker] = None
        self._queue: "queue.Queue[str]" = queue.Queue()

        self._recordings_var = tk.StringVar(value=str(Path("recordings").resolve()))
        self._output_var = tk.StringVar(value=str(Path("out").resolve()))
        self._session_var = tk.StringVar(value="")
        self._device_var = tk.StringVar(value="cuda")
        self._vad_var = tk.BooleanVar(value=True)
        self._sanitize_var = tk.BooleanVar(value=False)
        self._align_var = tk.BooleanVar(value=False)

        self._build_layout()
        self.after(self.POLL_INTERVAL_MS, self._poll_output)

    def _build_layout(self) -> None:
        padding = {"padx": 10, "pady": 5}

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        paths_frame = ttk.LabelFrame(main_frame, text="Ścieżki")
        paths_frame.pack(fill=tk.X, **padding)

        self._add_path_row(paths_frame, "Nagrania", self._recordings_var, self._browse_recordings)
        self._add_path_row(paths_frame, "Wyjściowy", self._output_var, self._browse_output)

        session_frame = ttk.Frame(paths_frame)
        session_frame.pack(fill=tk.X, **padding)
        ttk.Label(session_frame, text="Sesja (opcjonalnie)").pack(side=tk.LEFT)
        session_entry = ttk.Entry(session_frame, textvariable=self._session_var)
        session_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Button(session_frame, text="Wybierz", command=self._browse_session).pack(side=tk.LEFT)

        options_frame = ttk.LabelFrame(main_frame, text="Opcje")
        options_frame.pack(fill=tk.X, **padding)

        device_label = ttk.Label(options_frame, text="Urządzenie")
        device_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        device_combo = ttk.Combobox(
            options_frame,
            textvariable=self._device_var,
            values=("cuda", "cpu"),
            state="readonly",
            width=10,
        )
        device_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Checkbutton(options_frame, text="Filtr VAD", variable=self._vad_var).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Checkbutton(options_frame, text="Usuwaj szumy", variable=self._sanitize_var).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5
        )
        ttk.Checkbutton(options_frame, text="Align słów", variable=self._align_var).grid(
            row=1, column=2, sticky=tk.W, padx=5, pady=5
        )

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, **padding)
        self._start_button = ttk.Button(action_frame, text="Start", command=self._start_transcription)
        self._start_button.pack(side=tk.LEFT)
        self._stop_button = ttk.Button(action_frame, text="Stop", command=self._stop_transcription, state=tk.DISABLED)
        self._stop_button.pack(side=tk.LEFT, padx=10)

        text_frame = ttk.LabelFrame(main_frame, text="Log")
        text_frame.pack(fill=tk.BOTH, expand=True, **padding)
        self._text = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED)
        self._text.pack(fill=tk.BOTH, expand=True)

    def _add_path_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, command) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Button(row, text="Wybierz", command=command).pack(side=tk.LEFT)

    def _browse_recordings(self) -> None:
        selected = filedialog.askdirectory(title="Wybierz katalog nagrań")
        if selected:
            self._recordings_var.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(title="Wybierz katalog wyjściowy")
        if selected:
            self._output_var.set(selected)

    def _browse_session(self) -> None:
        base = Path(self._recordings_var.get())
        initialdir = base if base.exists() else Path.cwd()
        selected = filedialog.askdirectory(title="Wybierz katalog sesji", initialdir=initialdir)
        if selected:
            self._session_var.set(selected)

    def _validate_paths(self) -> bool:
        recordings = Path(self._recordings_var.get())
        output = Path(self._output_var.get())
        if not recordings.exists():
            messagebox.showerror("Błąd", "Katalog nagrań nie istnieje.")
            return False
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Błąd", f"Nie mogę utworzyć katalogu wyjściowego: {exc}")
            return False
        session = self._session_var.get().strip()
        if session:
            session_path = Path(session)
            if not session_path.exists():
                messagebox.showwarning("Ostrzeżenie", "Wybrana sesja nie istnieje — zostanie zignorowana.")
        return True

    def _environment(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "RECORDINGS_DIR": self._recordings_var.get(),
                "OUTPUT_DIR": self._output_var.get(),
                "WHISPER_DEVICE": self._device_var.get(),
                "WHISPER_VAD": "1" if self._vad_var.get() else "0",
                "SANITIZE_LOWER_NOISE": "1" if self._sanitize_var.get() else "0",
                "WHISPER_ALIGN": "1" if self._align_var.get() else "0",
            }
        )
        session = self._session_var.get().strip()
        if session:
            env["SESSION_DIR"] = session
        else:
            env.pop("SESSION_DIR", None)
        return env

    def _start_transcription(self) -> None:
        if self._worker is not None:
            messagebox.showinfo("Informacja", "Transkrypcja już trwa.")
            return
        if not self._validate_paths():
            return
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.configure(state=tk.DISABLED)
        env = self._environment()
        self._worker = TranscriptionWorker(env, self._queue)
        self._worker.start()
        self._start_button.configure(state=tk.DISABLED)
        self._stop_button.configure(state=tk.NORMAL)

    def _stop_transcription(self) -> None:
        if self._worker is None:
            return
        self._worker.terminate()
        self._queue.put("[info] Wysłano sygnał zatrzymania.\n")

    def _poll_output(self) -> None:
        try:
            while True:
                line = self._queue.get_nowait()
                self._append_text(line)
        except queue.Empty:
            pass
        finally:
            if self._worker and not self._worker.is_alive():
                self._worker = None
                self._start_button.configure(state=tk.NORMAL)
                self._stop_button.configure(state=tk.DISABLED)
            self.after(self.POLL_INTERVAL_MS, self._poll_output)

    def _append_text(self, text: str) -> None:
        self._text.configure(state=tk.NORMAL)
        self._text.insert(tk.END, text)
        self._text.see(tk.END)
        self._text.configure(state=tk.DISABLED)


def main() -> None:  # pragma: no cover - GUI launcher
    app = TranscriptionGUI()
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
