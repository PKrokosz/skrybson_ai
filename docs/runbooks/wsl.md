# Runbook: WSL

Ten runbook opisuje, jak uruchomić transkrypcję w środowisku Windows Subsystem for Linux (WSL).

## Wymagania wstępne
- Zainstalowany WSL2 z dystrybucją Ubuntu (lub kompatybilną).
- Zainstalowany Python 3.10+ wraz z `python3-venv`.
- Karta graficzna NVIDIA (opcjonalnie) oraz sterowniki/NVIDIA CUDA Toolkit jeżeli planujesz wykorzystać GPU.

## Kroki instalacji

1. **Sklonuj repozytorium i przejdź do katalogu projektu:**
   ```bash
   git clone <URL_DO_REPO>
   cd skrybson_ai
   ```

2. **Utwórz i aktywuj środowisko wirtualne:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Zainstaluj zależności Python:**
   - Domyślna instalacja CPU:
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - Instalacja z akceleracją GPU (CUDA):
     ```bash
     pip install --upgrade pip
     pip install faster-whisper
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
     > Dostosuj wersję CUDA (`cu118`) do posiadanej karty oraz zainstalowanych sterowników.

4. **Przygotuj katalogi na nagrania oraz wyniki:**
   ```bash
   mkdir -p recordings out
   ```

5. **Uruchom transkrypcję:**
   ```bash
   python transcribe.py
   ```

## Walidacja działania GPU
- Jeżeli GPU jest poprawnie wykryte, skrypt wypisze informację o używaniu urządzenia `cuda`.
- Możesz również uruchomić `nvidia-smi`, aby upewnić się, że urządzenie jest widoczne w WSL.

## Dalsze kroki
- Konfigurację sterowaną zmiennymi środowiskowymi znajdziesz w pliku `.env.example`.
- Aby wyjść z wirtualnego środowiska, wpisz `deactivate`.
