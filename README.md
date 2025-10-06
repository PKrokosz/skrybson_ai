# Skrybson AI — Runbook

Transkrybent nagrań Discorda skupiony na jakości polskiej mowy. Repozytorium udostępnia
skrypt CLI, który obrabia nagrania per użytkownik i generuje zestaw artefaktów
(JSON, SRT, VTT oraz oś czasu całej rozmowy).

## 🚀 Szybki start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # opcjonalnie: pip install -r requirements-align.txt
cp .env.example .env
python transcribe.py --profile quality@cuda
```

Domyślna konfiguracja zakłada katalogi `./recordings` (wejście) i `./out`
(wyjście). Wartości można nadpisać w `.env` lub flagami CLI.

### Profile uruchomieniowe

| Profil          | Urządzenie | Model      | Compute        | Zastosowanie                           |
| --------------- | ---------- | ---------- | -------------- | -------------------------------------- |
| `quality@cuda`  | CUDA       | large-v3   | int8_float16   | Maksymalna jakość na GPU ≥6 GiB VRAM   |
| `cpu-fallback`  | CPU        | medium     | int8           | Tryb „po spotkaniu” bez GPU            |
| `ci-mock`       | CPU        | mock/tiny  | int8 (mock)    | Błyskawiczne testy bez modeli ASR      |

Profil można ustawić zmienną `WHISPER_PROFILE` lub flagą `--profile`. Wciąż
można nadpisywać poszczególne parametry (`WHISPER_MODEL`, `WHISPER_DEVICE`,
`WHISPER_COMPUTE`, `--beam-size`, `--language`, ...).

### Kluczowe zmienne środowiskowe

| Zmienna                | Opis                                                                 | Domyślna wartość |
| ---------------------- | -------------------------------------------------------------------- | ---------------- |
| `RECORDINGS_DIR`       | Główny katalog nagrań (manifest + katalogi sesji)                     | `./recordings`   |
| `OUTPUT_DIR`           | Katalog wynikowy                                                     | `./out`          |
| `SESSION_DIR`          | Wymuszenie konkretnej sesji (względnie względem `RECORDINGS_DIR`)    | ostatnia sesja   |
| `WHISPER_PROFILE`      | Jeden z profili (`quality@cuda`, `cpu-fallback`, `ci-mock`)          | `quality@cuda`   |
| `WHISPER_MODEL`        | Nazwa modelu whisper (np. `large-v3`, `medium`)                      | wg profilu       |
| `WHISPER_DEVICE`       | `cuda` lub `cpu`                                                      | wg profilu       |
| `WHISPER_COMPUTE`      | Tryb obliczeń (`int8_float16`, `int8`, `float16`)                    | wg profilu       |
| `WHISPER_SEGMENT_BEAM` | Rozmiar wiązki segmentów                                              | `5`              |
| `WHISPER_LANG`         | Język dekodera                                                        | `pl`             |
| `WHISPER_VAD`          | Włączenie filtra VAD                                                  | `true`           |
| `SANITIZE_LOWER_NOISE` | Redukcja drobnych wtrąceń („uhm”, „eee”)                              | `false`          |
| `WHISPER_ALIGN`        | Alignment słów (wymaga pakietów z `requirements-align.txt`)          | `false`          |
| `WHISPER_MOCK`         | Wymuszenie mocka nawet poza profilem `ci-mock`                        | `false`          |

Pełna lista dostępnych opcji: `python transcribe.py --help`.

## 📦 Artefakty wyjściowe

Dla każdej ścieżki użytkownika powstają pliki `*.json`, `*.srt`, `*.vtt`, a w
`transcripts/all_in_one.srt` otrzymasz wspólną oś czasu rozmowy. Segmenty są
„soft-merge’owane”, dzięki czemu krótkie wtrącenia łączą się z sąsiadami, a
znaczniki czasu pozostają spójne. W `conversation.json` znajdziesz oś czasu z
względnymi odchyleniami od początku sesji oraz mapowanie na pliki źródłowe.

W przypadku obecności `manifest.json` sesji, skrypt aktualizuje go o ścieżki do
nowo wygenerowanych transkryptów.

## 🧠 Strategie przełączania modeli

1. **Wykrywanie środowiska.** Jeśli nie ma CUDA, aplikacja przełącza się na
   profil CPU (`medium @ int8`).
2. **Obsługa OOM.** Próba inicjalizacji na GPU wykonuje sekwencję wariantów:
   `large-v3 @ int8_float16` → `large-v3 @ int8` → `medium @ int8_float16` →
   `medium @ int8` → fallback CPU. Wszystkie kroki są logowane.
3. **Profil `ci-mock`.** Tworzony jest lekki model mockujący odpowiedzi (bez
   pobierania modeli ASR); przydaje się w CI oraz przy szybkim smoke-teście
   pipeline’u.

## 🛠️ Workflow developera

```bash
pip install -r requirements-dev.txt
ruff check .
mypy .
pytest
```

Zestaw powyżej jest identyczny z GitHub Actions (`.github/workflows/ci.yml`).
Testy operują na danych generowanych w locie i nie wymagają realnych modeli.

### Przydatne profile i runbooki

- `python transcribe.py --profile quality@cuda` — produkcyjna jakość na GPU.
- `python transcribe.py --profile cpu-fallback` — tryb offline na CPU.
- `python transcribe.py --profile ci-mock` — szybkie sprawdzenie pipeline’u bez
  zależności ASR.
- Dokument WSL: [`docs/runbooks/wsl.md`](docs/runbooks/wsl.md)
- Benchmark: [`docs/bench.md`](docs/bench.md)

## 🔒 Bezpieczeństwo

Plik `.env` nie jest wersjonowany. Uzupełnij tylko `.env.example`, a szczegóły
trzymania sekretów opisuje [`SECURITY.md`](SECURITY.md).

## 📑 Licencja

Repozytorium korzysta z tej samej licencji co `faster-whisper`.
