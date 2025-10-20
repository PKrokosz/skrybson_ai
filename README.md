# Skrybson AI

> **Polski transkrybent nagrań Discorda** – lokalne narzędzie, które porządkuje surowe nagrania sesji na artefakty gotowe do analizy.

## Spis treści

1. [Opis projektu](#opis-projektu)
2. [Najważniejsze funkcje](#najważniejsze-funkcje)
3. [Architektura i komponenty](#architektura-i-komponenty)
4. [Wymagania](#wymagania)
5. [Szybki start (TL;DR)](#szybki-start-tldr)
6. [Konfiguracja i układ danych](#konfiguracja-i-układ-danych)
7. [Korzystanie z CLI](#korzystanie-z-cli)
8. [Artefakty wyjściowe](#artefakty-wyjściowe)
9. [GUI oparte o Tkinter](#gui-oparte-o-tkinter)
10. [Dodatkowe narzędzia](#dodatkowe-narzędzia)
11. [Workflow developera](#workflow-developera)
12. [FAQ i rozwiązywanie problemów](#faq-i-rozwiązywanie-problemów)
13. [Bezpieczeństwo](#bezpieczeństwo)
14. [Licencja](#licencja)

---

## Opis projektu

Skrybson AI to zestaw narzędzi do lokalnej transkrypcji nagrań Discorda. Projekt stawia na wysoką
jakość mowy polskiej i pełną kontrolę nad danymi (bez wysyłania materiałów do chmury). Rdzeniem
systemu jest skrypt CLI, który dla każdej sesji Discorda tworzy transkrypcje per użytkownik oraz
wspólną oś czasu rozmowy.

## Najważniejsze funkcje

- 🎙️ **Transkrypcja Whisper/faster-whisper** z gotowymi profilami GPU/CPU i inteligentnym
  przełączaniem modeli.
- 🧹 **Porządkowanie wypowiedzi** – łączenie krótkich wtrąceń, usuwanie szumu typu „uhm/eee",
  normalizacja znaków specjalnych.
- 🗂️ **Artefakty wieloformatowe** – JSON, SRT, VTT oraz globalna oś czasu (`conversation.json`).
- 🔀 **Obsługa manifestu** (`manifest.json`) – automatyczna aktualizacja odnośników do plików z
  transkryptami.
- 🧪 **Tryb mock** (`ci-mock`) – natychmiastowe testy bez pobierania modeli ASR.
- 🧭 **Dodatkowe narzędzia** – wyrównywanie słów z WhisperX, benchmark GPU/CPU, eksperymentalne GUI.

## Architektura i komponenty

```
├── transcribe.py      # Główne CLI i logika transkrypcji
├── align.py           # Opcjonalne wyrównywanie słów (WhisperX + diarization)
├── ui/                # Aplikacja Tkinter (ttkbootstrap)
├── bench.py / bench/  # Narzędzia pomiarowe i sanity checki modeli
├── docs/              # Runbooki (WSL, benchmarki, sanity report)
└── tests/             # Testy jednostkowe logiki pomocniczej
```

Centralna logika transkrypcji mieszka w `transcribe.py`. Skrypt:

1. Wykrywa środowisko (CUDA/CPU) i dobiera profil (`quality@cuda`, `cpu-fallback`, `ci-mock`).
2. Przygotowuje konfigurację (katalog nagrań, katalog wyjściowy, sesja docelowa, beam search,
   język, filtry VAD, redukcję szumu, alignment słów).
3. Iteruje po nagraniach użytkowników, generuje transkrypcje i zapisuje artefakty per użytkownik.
4. Buduje globalną oś czasu (`all_in_one.srt`, `conversation.json`) i aktualizuje `manifest.json`.

## Wymagania

### Oprogramowanie

- Python **3.10+** (konfiguracja narzędzi linters/testów zakłada 3.12).
- System operacyjny: Linux/macOS/WSL; Windows natywnie wymaga środowiska zgodnego z Pythonem 3.10+.
- `ffmpeg` (zalecane dla bezproblemowego odczytu różnych formatów audio).
- (Opcjonalnie) NVIDIA CUDA 11+ z min. **6 GiB VRAM** dla profilu `quality@cuda`.

### Zależności pip

- Podstawowe: `pip install -r requirements.txt` (m.in. `faster-whisper`).
- Align/dokładne znaczniki słów: `pip install -r requirements-align.txt` (WhisperX,
  `pyannote.audio`).
- Narzędzia developerskie: `pip install -r requirements-dev.txt` (`pytest`, `ruff`, `mypy`).

## Szybki start (TL;DR)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # opcjonalnie: -r requirements-align.txt
cp .env.example .env
python transcribe.py --profile quality@cuda
```

Domyślna konfiguracja korzysta z `./recordings` (źródło) i `./out` (wyniki). Możesz je nadpisać
zmiennymi środowiskowymi lub flagami CLI.

## Konfiguracja i układ danych

### Zmienne środowiskowe

| Nazwa                 | Opis                                                                 | Domyślna wartość |
| --------------------- | -------------------------------------------------------------------- | ---------------- |
| `RECORDINGS_DIR`      | Katalog nagrań (zawiera katalogi sesji lub `manifest.json`)           | `./recordings`   |
| `OUTPUT_DIR`          | Katalog wynikowy                                                     | `./out`          |
| `SESSION_DIR`         | Wymuszona ścieżka sesji (względna względem `RECORDINGS_DIR`)          | ostatnia sesja   |
| `WHISPER_PROFILE`     | Profil (`quality@cuda`, `cpu-fallback`, `ci-mock`)                   | `quality@cuda`   |
| `WHISPER_MODEL`       | Rozmiar modelu whisper (np. `large-v3`, `medium`)                    | wg profilu       |
| `WHISPER_DEVICE`      | `cuda` lub `cpu`                                                      | wg profilu       |
| `WHISPER_COMPUTE`     | Tryb obliczeń (`int8_float16`, `int8`, `float16`)                    | wg profilu       |
| `WHISPER_SEGMENT_BEAM`| Rozmiar wiązki segmentów                                             | `5`              |
| `WHISPER_LANG`        | Wymuszony język modelu                                               | `pl`             |
| `WHISPER_VAD`         | Filtr ciszy Voice Activity Detection                                 | `true`           |
| `SANITIZE_LOWER_NOISE`| Redukcja wtrąceń („uhm”, „eee”)                                      | `false`          |
| `WHISPER_ALIGN`       | Generowanie znaczników słów (wymaga `requirements-align.txt`)        | `false`          |
| `WHISPER_MOCK`        | Wymuszenie mocka niezależnie od profilu                              | `false`          |

Pełną listę flag CLI uzyskasz poleceniem `python transcribe.py --help`.

### Struktura katalogów

```
recordings/
└── 2024-05-01-sprint-demo/        # katalog sesji (ostatnia modyfikacja = ostatnia sesja)
    ├── manifest.json              # opcjonalny opis spotkania (aktualizowany automatycznie)
    ├── members/                   # nagrania per użytkownik
    │   ├── 1234567890.flac
    │   └── ...
    └── stage/                     # nagrania wspólne/scalane (opcjonalne)
```

- Jeżeli `SESSION_DIR` nie jest ustawione, wybierana jest **najświeższa** sesja w `RECORDINGS_DIR`.
- Manifest może zawierać pola `startISO`, `title`, `participants` itd.; skrypt dopisze sekcję
  `transcripts` ze ścieżkami do wygenerowanych plików.

## Korzystanie z CLI

```bash
python transcribe.py \
  --recordings ./recordings \
  --output ./out \
  --profile quality@cuda \
  --align-words \
  --sanitize-lower-noise
```

Najważniejsze flagi:

- `--profile` – szybki wybór zestawu parametrów. Dostępne: `quality@cuda`, `cpu-fallback`,
  `ci-mock`.
- `--device`, `--model`, `--compute-type` – ręczne nadpisanie wyborów z profilu.
- `--session` – wskazanie konkretnej sesji (ścieżka absolutna lub względna względem `RECORDINGS_DIR`).
- `--align-words` – włącza word-level timestamps (wymaga dodatkowych zależności i GPU dla diarization).
- `--sanitize-lower-noise` – usuwa drobne wtrącenia („uhm”, „eee”).
- `--vad/--no-vad` – włącza/wyłącza filtr ciszy (VAD).

### Strategie przełączania modeli

1. **Wykrywanie środowiska:** brak CUDA → automatyczny profil CPU (`medium @ int8`).
2. **Obsługa OOM:** dla GPU wykonywana jest sekwencja prób: `large-v3 @ int8_float16` →
   `large-v3 @ int8` → `medium @ int8_float16` → `medium @ int8` → fallback na CPU.
3. **Profil `ci-mock`:** generuje deterministyczny tekst „mockowy” bez pobierania modeli – przydatne
   w CI oraz smoke testach.

## Artefakty wyjściowe

Dla każdego użytkownika powstają pliki `*.json`, `*.srt`, `*.vtt` z informacjami o segmentach (czas
startu, końca, tekst, lista źródłowych plików). W katalogu sesji pojawiają się także:

- `transcripts/all_in_one.srt` – wspólna oś czasu rozmowy.
- `transcripts/conversation.json` – scalona lista segmentów z normalizowanymi timestampami i mapą do
  źródeł audio.
- `transcripts/<user_id>/...` – pliki per użytkownik (JSON + napisy).
- Aktualizowany `manifest.json` ze ścieżkami do nowych transkryptów.

Segmenty są „soft-merge’owane”, dzięki czemu krótkie wtrącenia tego samego użytkownika łączą się w
spójniejsze wypowiedzi. Dodatkowe metadane obejmują listę źródłowych plików oraz – przy włączonym
alignmencie – słowa wraz ze znacznikami czasowymi.

## GUI oparte o Tkinter

Eksperymentalną aplikację okienkową uruchomisz poleceniem:

```bash
python -m ui.app
```

Skróty klawiszowe:

- `Ctrl+O` – wybór katalogu nagrań (`RECORDINGS_DIR`).
- `Ctrl+R` – odśwież listę sesji.
- `Ctrl+T` – widok transkrypcji.
- `Ctrl+E` – widok wyników i eksportu.
- `Ctrl+L` – logi procesów.

## Dodatkowe narzędzia

- `align.py` – CLI do niezależnego wyrównywania słów (WhisperX) i opcjonalnej diarization
  (`--diarize`, wymaga `PYANNOTE_AUTH_TOKEN`).
- `bench.py` / `bench/` – zestaw skryptów do benchmarków i sanity checków modeli Whisper.
- `docs/runbooks/wsl.md` – instrukcje uruchomienia w środowisku WSL.
- `docs/bench.md` – wyniki benchmarków i wskazówki dot. wydajności.

### Rejestrowanie sesji Discorda (bot Node.js)

Repo zawiera minimalistycznego bota nagrywającego kanały głosowe Discorda (`index.js`). Bot
nasłuchuje komendy slash `/record start|stop` i zapisuje strumienie użytkowników do WAV-ów w
`<RECORDINGS_DIR>/<SESSION_PREFIX>-<timestamp>-<channel_id>/raw`. Manifest sesji (`manifest.json`)
otrzymuje metadane (`startISO`, `stopISO`, id kanału, itp.), dzięki czemu `transcribe.py` może
bezpośrednio przetwarzać nagrania.

Szybki start bota:

```bash
npm install
cp .env.example .env  # ustaw DISCORD_TOKEN
node index.js         # lub: npm start
```

Najważniejsze zmienne środowiskowe:

| Nazwa              | Opis                                                                 | Domyślna wartość |
| ------------------ | -------------------------------------------------------------------- | ---------------- |
| `DISCORD_TOKEN`    | Token bota Discorda (wymagany).                                       | —                |
| `RECORDINGS_DIR`   | Katalog, w którym bot zapisuje sesje.                                 | `./recordings`   |
| `SESSION_PREFIX`   | Prefiks katalogów sesji tworzonych przez bota.                        | `session`        |

Po zalogowaniu bot rejestruje slash command `record`. Komenda `/record start` rozpoczyna
nagrywanie aktualnego kanału głosowego użytkownika, `/record stop` kończy sesję i domyka wpis w
`manifest.json`.

## Workflow developera

```bash
pip install -r requirements-dev.txt
ruff check .
mypy .
pytest
```

- Testy (`pytest`) generują dane tymczasowe – nie wymagają pobierania modeli.
- `ruff` i `mypy` mają identyczną konfigurację jak GitHub Actions (`.github/workflows/ci.yml`).
- Repo zawiera `docker-compose.yml` i `docker-compose.gpu.yml` do uruchomień kontenerowych.

## FAQ i rozwiązywanie problemów

<details>
<summary>Brak CUDA / problemy z GPU</summary>

Skrypt automatycznie spadnie na profil CPU (`medium @ int8`). Możesz wymusić tryb CPU flagą
`--profile cpu-fallback` lub zmienną `WHISPER_DEVICE=cpu`.

</details>

<details>
<summary>Modele zajmują za dużo miejsca</summary>

Użyj profilu `ci-mock`, aby przetestować pipeline bez pobierania modeli ASR. W produkcji możesz
ustawić `WHISPER_MODEL=small` lub `medium`.

</details>

<details>
<summary>Chcę zredukować wtrącenia typu „uhm”</summary>

Ustaw `SANITIZE_LOWER_NOISE=true` lub dodaj flagę `--sanitize-lower-noise`. Dla pełnej kontroli
wyłącz ją flagą `--keep-noise`.

</details>

<details>
<summary>Potrzebuję word-level timestamps</summary>

Zainstaluj `pip install -r requirements-align.txt`, ustaw `WHISPER_ALIGN=true` lub dodaj flagę
`--align-words`. W razie potrzeby podaj `PYANNOTE_AUTH_TOKEN` dla diarization.

</details>

## Bezpieczeństwo

Plik `.env` nie jest wersjonowany. Uzupełnij `.env.example` i trzymaj sekrety zgodnie ze wskazówkami
w [`SECURITY.md`](SECURITY.md).

## Licencja

Repozytorium dziedziczy licencję projektu [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper).
