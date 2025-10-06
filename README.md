# Skrybson AI — Runbook

Repozytorium zawiera narzędzia do transkrypcji nagrań Discorda przy użyciu biblioteki [faster-whisper](https://github.com/guillaumekln/faster-whisper).

## Szybki start
- Przykładowa konfiguracja środowiska znajduje się w pliku `.env.example`.
- Przed uruchomieniem upewnij się, że katalogi `recordings/` oraz `out/` istnieją.

## Tryby uruchomienia

### 1. WSL (natywny Python)
Szczegółowe kroki opisuje [Runbook WSL](docs/runbooks/wsl.md). W skrócie:
```bash
python -m venv .venv
source .venv/bin/activate
pip install faster-whisper
python transcribe.py
```

Wywołanie `transcribe.py` obsługuje teraz flagi CLI, dzięki czemu nie musisz eksportować zmiennych środowiskowych:

```bash
python transcribe.py \
  --recordings recordings/2024-09-01 \
  --output out \
  --device cpu \
  --model medium \
  --beam-size 3
```

Pełną listę dostępnych opcji znajdziesz w `python transcribe.py --help`.
Jeżeli chcesz korzystać z GPU, doinstaluj odpowiednią wersję PyTorch z obsługą CUDA (patrz runbook).

### 1a. Proste GUI (Tkinter)

Dla szybkiego uruchomienia transkrypcji bezpośrednio z komputera możesz skorzystać z prostego interfejsu graficznego opartego o Tkinter.

```bash
python gui.py
```

W oknie aplikacji wskaż katalog z nagraniami oraz folder wyjściowy (domyślnie `recordings/` i `out/`). Opcjonalnie wybierz konkretną sesję, urządzenie (`cuda`/`cpu`) oraz dodatkowe filtry.

### 2. Docker z GPU
Dla środowisk z kartą NVIDIA dostępny jest plik `docker-compose.gpu.yml` oparty o obraz `ghcr.io/guillaumekln/faster-whisper:latest-cuda`.

Uruchomienie:
```bash
docker compose -f docker-compose.gpu.yml up
```

Montowane są katalogi:
- `./recordings` → `/app/recordings`
- `./out` → `/app/out`
- całe repozytorium → `/app`

Po starcie kontenera w logach powinno pojawić się potwierdzenie wykrycia GPU (np. wyjście `nvidia-smi` lub informacja `cuda` z biblioteki CTranslate2).

## Polityka modeli i strojenie
- Domyślnie (CUDA) używany jest model `large-v3` z trybem `int8_float16`.
- Jeżeli GPU nie jest dostępne, aplikacja automatycznie przełącza się na CPU z modelem `medium` i trybem `int8`.
- Parametr wiązki segmentów można sterować przez `WHISPER_SEGMENT_BEAM` (domyślnie `5`).
- Filtr VAD włącza się/wyłącza zmienną `WHISPER_VAD` (`on` domyślnie).
- Precyzyjne znaczniki słów (WhisperX) aktywujesz przez `WHISPER_ALIGN=1` (wymaga dodatkowych modeli align/pyannote).

## Rozwiązywanie problemów
- Brak wykrytego GPU: sprawdź `nvidia-smi` zarówno w systemie gospodarza, jak i wewnątrz kontenera.
- Wolna transkrypcja: ustaw `WHISPER_MODEL` na mniejszy (np. `medium` lub `small`) albo wymuś `WHISPER_DEVICE=cpu`.

## Licencja
Projekt korzysta z tych samych zasad licencyjnych co `faster-whisper`.
