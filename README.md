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
Jeżeli chcesz korzystać z GPU, doinstaluj odpowiednią wersję PyTorch z obsługą CUDA (patrz runbook).

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

## Rozwiązywanie problemów
- Brak wykrytego GPU: sprawdź `nvidia-smi` zarówno w systemie gospodarza, jak i wewnątrz kontenera.
- Wolna transkrypcja: zmień `WHISPER_MODEL` na mniejszy (np. `medium` lub `small`) i dostosuj `WHISPER_COMPUTE`.

## Licencja
Projekt korzysta z tych samych zasad licencyjnych co `faster-whisper`.
