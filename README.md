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

## CODEX CHECK

# skrybson-ai
Lokalny zestaw narzędzi do nagrywania kanałów głosowych Discord i przetwarzania ich na uporządkowane transkrypcje oraz artefakty analityczne.​:codex-file-citation[codex-file-citation]{line_range_start=1 line_range_end=20 path=pyproject.toml git_url="https://github.com/PKrokosz/skrybson_ai/blob/main/pyproject.toml#L1-L20"}​

## Spis treści
- [Architektura (mermaid)](#architektura-mermaid)
- [Możliwości i funkcje (Capabilities)](#możliwości-i-funkcje-capabilities)
- [Publiczne API / Punkty wejścia](#publiczne-api--punkty-wejścia)
- [Konfiguracja i środowisko](#konfiguracja-i-środowisko)
- [Instalacja i szybki start](#instalacja-i-szybki-start)
- [Przepływy pracy (Sposoby pracy)](#przepływy-pracy-sposoby-pracy)
- [Testy i jakość](#testy-i-jakość)
- [Integracje i zewnętrzne usługi](#integracje-i-zewnętrzne-usługi)
- [Ograniczenia i znane problemy](#ograniczenia-i-znane-problemy)
- [Roadmap / TODO](#roadmap--todo)
- [Załączniki](#załączniki)
- [Luki informacyjne](#luki-informacyjne)

## Architektura (mermaid)
```mermaid
flowchart TD
    DiscordBot[index.js<br/>Discord recorder] -->|WAV| Recordings[(recordings/)]
    Recordings --> Transcriber[transcribe.py<br/>CLI]
    Transcriber -->|JSON/SRT/VTT| Output[(out/<session>/transcripts)]
    Transcriber -->|manifest update| Manifest[manifest.json]
    Output --> AlignCLI[align.py<br/>WhisperX]
    Output --> TkUI[ui.app<br/>SkrybsonApp]
    Output --> Bench[bench.py<br/>metrics]
    Docker[docker-compose] --> DiscordBot
    Docker --> Transcriber
Source: 

Możliwości i funkcje (Capabilities)
Nagrywaj kanały głosowe Discorda komendą /record start, zapisując surowe fragmenty WAV per użytkownik i aktualizując manifest sesji.

Generuj transkrypcje per użytkownika i globalne (JSON, SRT, VTT), tworząc indeks i odświeżając manifest sesji.

Oczyszczaj tekst (redukcja powtórzeń, filtracja wypełniaczy) i miękko scalaj krótkie segmenty konwersacji.

Wyrównuj słowa i (opcjonalnie) diarizuj nagrania przy użyciu WhisperX z poziomu CLI lub zautomatyzowanego worker’a GUI.

Zarządzaj sesjami, transkrypcją, alignmentem, logami i eksportem w wielowidokowym interfejsie SkrybsonApp.

Benchmarkuj modele faster-whisper na zestawie próbek, korzystając z gotowych wyników lub świeżych uruchomień.

Publiczne API / Punkty wejścia
CLI Python
Transkrypcja
python transcribe.py --recordings ./recordings --output ./out --profile quality@cuda
Parametry nadpisują zmienne środowiskowe (katalogi, profil, urządzenie, model, beam, język, VAD, redukcję szumów, alignment). Wyjściem są katalogi transcripts/ z JSON/SRT/VTT i zaktualizowany manifest sesji. Zwraca kod błędu, gdy brakuje nagrań lub sesji.

Alignment słów
python align.py recordings/session/raw.wav out/transcripts/user.json --output out/transcripts/user.aligned.json --device cuda --language pl --diarize
Wymaga JSON-a z listą segmentów; opcjonalnie generuje diarization przy tokenie PYANNOTE_AUTH_TOKEN. Wynik zawiera listę słów z czasami oraz (gdy diarization aktywne) etykiety mówców.

Benchmark
python bench.py --run-models --models small medium --device cuda --compute-type int8_float16
Bez --run-models korzysta z bench/results/precomputed.json. Raport zapisuje do bench/results/latest_metrics.json i wypisuje tabelę WER/VRAM/czas.

Interfejsy graficzne
python -m ui.app uruchamia aplikację SkrybsonApp (wymaga środowiska z Tk/ttkbootstrap). Widoki pozwalają wskazać katalogi, odpalać transkrypcję (dry-run / real), monitorować logi i alignment.

python gui.py oferuje uproszczone okno Tkinter z formularzem ścieżek i przyciskami Start/Stop; przekazuje parametry do transcribe.py w wątku roboczym.

Bot Discord
npm start lub node index.js uruchamia bota, który rejestruje globalną komendę /record z podkomendami start i stop. Wymaga DISCORD_TOKEN i (opcjonalnie) RECORDINGS_DIR, SESSION_PREFIX. Surowe WAV-y i manifest są zapisywane w strukturze recordings/session-<timestamp>-<channelId>/raw. Przy zatrzymaniu bot dopisuje stopISO do manifestu.

Source: 

Konfiguracja i środowisko
Wersje narzędzi: Python ≥3.10 (docelowo 3.12 w narzędziach mypy/ruff), opcjonalne extras align (whisperx + pyannote). Node 22 w obrazie produkcyjnym. Docker Compose udostępnia wariant CPU (python:3.11-slim) i GPU (faster-whisper CUDA).

Zależności runtime: faster-whisper (podstawa), ttkbootstrap, Pillow (UI ikony), biblioteki Discord (discord.js, @discordjs/voice, prism-media, wav, fs-extra, @discordjs/opus, libsodium-wrappers, @snazzah/davey).

Zmienne środowiskowe:

DISCORD_TOKEN, RECORDINGS_DIR, SESSION_PREFIX (bot).

RECORDINGS_DIR, OUTPUT_DIR, SESSION_DIR, WHISPER_PROFILE, WHISPER_DEVICE, WHISPER_MODEL, WHISPER_COMPUTE, WHISPER_SEGMENT_BEAM, WHISPER_LANG, WHISPER_VAD, SANITIZE_LOWER_NOISE, WHISPER_ALIGN, WHISPER_MOCK (transkrypcja).

PYANNOTE_AUTH_TOKEN (alignment diarization).

Kontenery Compose wstrzykują powyższe oraz mapują katalogi hosta (recordings, out).

Konfiguracja UI: profile (quality@cuda, cpu-fallback, custom) i ścieżki zapisywane w ~/.skrybson/config.json. Status bar prezentuje aktywny profil i status widoku.

Source: 

Instalacja i szybki start
Wariant Python
bash
Skopiuj kod
# co robi: tworzy i aktywuje środowisko wirtualne
python -m venv .venv && source .venv/bin/activate
# co robi: instaluje zależności bazowe oraz (opcjonalnie) alignment
pip install -U pip && pip install -r requirements.txt && pip install '.[align]'  # extras gdy potrzebny whisperx
# co robi: uruchamia transkrypcję z domyślnym profilem
python transcribe.py --recordings ./recordings --output ./out
Source: 

Wariant Node
bash
Skopiuj kod
# co robi: instaluje zależności bota
npm install
# co robi: uruchamia nagrywanie (slash commands /record start|stop)
npm start
Source: 

Przepływy pracy (Sposoby pracy)
Nagrywanie ➜ Transkrypcja ręczna: Uruchom bota (npm start), nagraj /record start, po zakończeniu /record stop, a następnie python transcribe.py wskazując katalog nagrań recordings/…. Pliki wynikowe trafiają do out/<sesja>/transcripts.

Automatyzacja w Docker Compose: docker compose up discord-recorder (bot) oraz jednorazowy docker compose run transcriber generują wyniki w wolumenach hosta (./recordings, ./out). Wersja GPU korzysta z obrazu faster-whisper i rezerwuje urządzenie NVIDIA.

Transkrypcja z UI: W widoku „Transkrypcja” ustaw katalogi i zmienne (WHISPER_*, SANITIZE_LOWER_NOISE, WHISPER_ALIGN), następnie wybierz „Dry-Run (Mock)” lub „Start”. UI steruje transcribe.py przez TaskManager, zbierając logi w czasie rzeczywistym.

Alignment: Widok „Alignment” skanuje katalog wynikowy (discover_alignment_candidates) i pozwala uruchomić równolegle align.py dla wybranych JSON-ów, raportując sukcesy/ostrzeżenia w logu.

Eksploracja wyników i eksport: Widok „Sesje” prezentuje manifesty, „Wyniki” agreguje SRT/VTT/JSON i pozwala kopiować pliki do dowolnego katalogu, a „Logs” zbiera logi z transkrypcji.

Benchmarki: python bench.py --run-models uruchamia modele na próbkach z bench/manifest.json; bez flagi korzysta z wyników bench/results/precomputed.json. Widok „Bench” (UI) obecnie pokazuje statyczne wiersze i zapisuje raport do docs/bench.md.

Source: 

Testy i jakość
Jednostkowe & integracyjne: pytest (skonfigurowany z -ra, Python path = .). Testy pokrywają parser CLI, sanitizację tekstu, łączenie segmentów, fallbacki modelu, walidację manifestu, manager zadań oraz konstrukcję UI (warunkowo).

Linting/format: ruff (E,F,B,I; limit 100 znaków) i mypy (Python 3.12, ostrzeżenia warn_return_any, ignorowanie brakujących importów; UI oraz bench/align/guy zignorowane).

Przykładowe komendy:

pytest

ruff check .

mypy transcribe.py tests

Source: 

Integracje i zewnętrzne usługi
Discord API (slash commands, Voice Gateway) via discord.js, @discordjs/voice, @discordjs/opus, prism-media, libsodium-wrappers. Wymaga ważnego tokena bota w .env.

WhisperX / pyannote do alignmentu i diarization (opcjonalne, aktywowane przez extras align i zmienną PYANNOTE_AUTH_TOKEN).

Docker Compose do koordynacji usług (Node recorder + Python transcriber) oraz wariantu GPU (obraz ghcr.io/guillaumekln/faster-whisper).

Source: 

Ograniczenia i znane problemy
Transkrypcja kończy się z kodem błędu, jeśli brakuje katalogu nagrań (recordings_dir), katalogu sesji lub katalogu raw; dodatkowo ignoruje pliki WAV mniejsze niż 1 KB.

UI pozwala wpisać WHISPER_VAD_MIN_SILENCE_MS / WHISPER_VAD_SPEECH_PAD_MS, lecz load_config ich nie odczytuje – wartości VAD dobierane są automatycznie na podstawie wersji faster-whisper.

Alignment wykorzystuje pierwszy wpis raw_files z JSON-u; przy wielu nagraniach ostrzega i pomija pozostałe.

Widok „Bench” prezentuje dane przykładowe i nadpisuje docs/bench.md nowym raportem – brak integracji z rzeczywistymi metrykami UI.

Source: 

Roadmap / TODO
Brak oznaczonych TODO/FIXME w repozytorium (wyszukiwanie rg "TODO" nie zwróciło wyników).

Uzupełnić widok „Bench” o rzeczywiste dane z bench/results/latest_metrics.json zamiast statycznych wierszy i przenoszenia raportu do docs/bench.md.

Rozważyć wystawienie konfigurowalnych parametrów VAD w transcribe.py, by odpowiadały polom w UI (WHISPER_VAD_MIN_SILENCE_MS, WHISPER_VAD_SPEECH_PAD_MS).

Załączniki
Kluczowe pliki i role
Ścieżka	Rola
index.js	Bot Discord nagrywający kanały głosowe i utrzymujący manifest sesji.
transcribe.py	Główny pipeline transkrypcji: ładowanie konfiguracji, inferencja, sanitizacja, generowanie artefaktów.
align.py	CLI do wyrównania segmentów na poziomie słów oraz diarization WhisperX.
ui/app.py	Start SkrybsonApp z widokami Sessions/Transcribe/Align/Results/Bench/Logs/Settings.
bench.py	Benchmark modeli faster-whisper na zdefiniowanych próbkach i generacja raportu JSON/tabeli.

Słownik pojęć
Sesja – katalog w recordings/ z plikami raw/*.wav oraz manifest.json opisującym kanał, uczestników i artefakty transkrypcji.

Profil (profile) – zestaw domyślnych parametrów transkrypcji (WHISPER_*, beam, język, mock) wybierany w UI i CLI poprzez WHISPER_PROFILE.

Mock transcriber – lekka implementacja MockWhisperModel zwracająca placeholdery, używana w profilu ci-mock lub trybie „Dry-Run (Mock)”.

Luki informacyjne
Dokumentacja wdrożenia UI (opakowanie jako aplikacja desktopowa) – Nieustalone w repo.

Integracja wyników benchmarków z interfejsem użytkownika – Nieustalone w repo (obecnie statyczne wpisy w UI).

[ ] Czy wszystkie sekcje zawierają cytaty ze ścieżkami i zakresami linii?
[ ] Czy nie użyto nieistniejących komend/plików?
[ ] Czy każdy endpoint/komenda ma przykład użycia?
[ ] Czy diagram odzwierciedla realne pliki/konfiguracje?

Skopiuj kod




## Bezpieczeństwo

Plik `.env` nie jest wersjonowany. Uzupełnij `.env.example` i trzymaj sekrety zgodnie ze wskazówkami
w [`SECURITY.md`](SECURITY.md).

## Licencja

Repozytorium dziedziczy licencję projektu [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper).
