# Sanity check raport modułów

## Podsumowanie kontroli

| Moduł | Zakres | Automatyczne kontrole | Status | Najważniejsze obserwacje |
| --- | --- | --- | --- | --- |
| Transkrypcja (`transcribe.py`) | CLI do przetwarzania nagrań i orkiestracji pipeline'u | `ruff`, `pytest` | ✅ | Stabilne, pokryte testami jednostkowymi narzędzia pomocnicze. |
| Alignment (`align.py`) | Integracja z WhisperX i diarization | `ruff`, analiza kodu | ✅ | Kod defensywny, brak testów automatycznych. |
| Klasyczny GUI (`gui.py`) | Lekka nakładka Tkinter uruchamiająca CLI | `ruff`, analiza kodu | ✅ | Kod działający, ale brak testów oraz typów dla interfejsu. |
| Modularny UI (`ui/…`) | Aplikacja ttkbootstrap z widokami i serwisami | `ruff`, `pytest`, `mypy` | ⚠️ | Testy przechodzą, jednak `mypy` wskazuje 66 błędów (gł. brak anotacji i błędne typy). |
| Benchmark (`bench.py`, `bench/`) | Benchmark modeli `faster-whisper` | `ruff`, analiza kodu | ✅ | Kompletny pipeline z obsługą manifestów i metryk. |

## Szczegóły modułów

### Transkrypcja (`transcribe.py`)
* Bogaty logger narracyjny zapewnia spójne logi wykonywania i pomiar czasu etapów. 【F:transcribe.py†L45-L124】
* Konfiguracja `TranscribeConfig` i presety profili opisują urządzenia, modele i parametry dekodera. 【F:transcribe.py†L132-L198】
* Testy jednostkowe `tests/test_transcribe_utils.py` pokrywają kluczowe funkcje pomocnicze (np. sanitizację szumu, wybór profili); cały pakiet testowy przechodzi. 【3f07fb†L1-L10】
* `ruff check .` nie zgłasza problemów. 【bc39bf†L1-L2】
* `mypy .` nie raportuje błędów specyficznych dla tego modułu, ale globalny wynik obniża pakiet UI.

### Alignment (`align.py`)
* Klasa `WhisperWordAligner` zapewnia leniwą inicjalizację modeli dopasowania oraz opcjonalny diarizer sterowany tokenem `PYANNOTE_AUTH_TOKEN`. 【F:align.py†L31-L121】
* Funkcja CLI `main` umożliwia uruchomienie alignera z linii poleceń oraz zapis wyników do JSON. 【F:align.py†L133-L176】
* Brak dedykowanych testów — warto rozważyć dodanie fixture z mockiem `whisperx` oraz sprawdzenie obsługi błędów `AlignmentError`.

### Klasyczny GUI (`gui.py`)
* Wątek `TranscriptionWorker` odpala `transcribe.py` jako proces podrzędny i strumieniuje logi do kolejki. 【F:gui.py†L15-L45】
* Główne okno `TranscriptionGUI` konfiguruje ścieżki wejścia/wyjścia, opcje pipeline'u oraz podgląd logów. 【F:gui.py†L39-L132】
* Brak testów ani typowania dla komponentów Tkinter; sanity check ograniczył się do analizy kodu i potwierdzenia, że lint nie wykrywa błędów.

### Modularny UI (`ui/…`)
* `SkrybsonApp` bazuje na `ttkbootstrap.Window`, dynamicznie ładuje widoki i udostępnia skróty klawiszowe. 【F:ui/app.py†L21-L103】
* Serwis `discover_sessions` agreguje metadane manifestu i nagrań, budując `SessionSummary` z ikonami statusu. 【F:ui/services/sessions.py†L18-L121】
* `AppState` zarządza konfiguracją, profilami oraz kolejką logów z zachowaniem wątkowości. 【F:ui/state.py†L14-L179】
* Testy UI (`tests/ui/test_sessions.py`, `tests/ui/test_tasks.py`) przechodzą, co potwierdza działanie bazowych scenariuszy. 【3f07fb†L1-L10】
* `mypy` wskazuje 66 błędów — dominują błędne anotacje `Mapping` (zwracanie `Any`), konflikty `__slots__` oraz brak definicji typów widgetów (`ttk.Widget`). 【3843dd†L1-L44】【42075b†L1-L4】【c706f3†L1-L30】
* Rekomendacje: uzupełnić deklaracje typów widgetów (np. przez import `from tkinter import ttk` lub aliasy w `typing.TYPE_CHECKING`), poprawić anotacje w `SessionSummary` i `AppState`, usunąć zbędne `type: ignore` oraz zainicjować interfejsy w widokach (`View.refresh`, `View.on_show`).

### Benchmark (`bench.py`, `bench/`)
* Parser argumentów obsługuje manifesty próbek, dane precomputed oraz wybór modeli/urządzeń. 【F:bench.py†L32-L77】
* Ładowanie manifestów i wyników mapuje dane na `Sample` oraz strukturę modeli z walidacją wejścia. 【F:bench.py†L80-L112】
* Moduł liczy metryki WER i char diff wraz z normalizacją tekstu. 【F:bench.py†L114-L156】
* Funkcja `run_model_on_samples` integruje `faster-whisper`, śledzi czas oraz VRAM (jeśli dostępne CUDA). 【F:bench.py†L163-L200】
* Brak dedykowanych testów; sanity check ogranicza się do przeglądu kodu i lintingu.

