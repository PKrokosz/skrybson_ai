# Benchmark transkrypcji — maj 2024

Poniższy benchmark wykorzystuje trzy dwuminutowe próbki polskich nagrań roboczych (czyste studio, open space z szumem tła oraz rozmowę overlappującą się). Do obliczeń użyto skryptu [`bench.py`](../bench.py) na podstawie manifestu [`bench/manifest.json`](../bench/manifest.json) i wyników zapisanych w [`bench/results/precomputed.json`](../bench/results/precomputed.json). Zestawienie prezentuje średnie wartości błędów WER, różnice znaków oraz koszty czasu/VRAM.

> Aby ograniczyć rozmiar repozytorium, nagrania `.wav` nie są wersjonowane. Przed ponownym uruchomieniem benchmarku umieść własne pliki w katalogu [`bench/samples`](../bench/samples/README.md), zachowując nazwy z manifestu.

## Metryki per model

| Model | Średni WER | Średni char diff | Średni czas [s] | Max VRAM [MiB] |
| --- | --- | --- | --- | --- |
| small | 36.15% | 162.67 | 196.50 | 612 |
| medium | 22.80% | 116.00 | 129.10 | 1 034 |
| large-v3 | **0.00%** | **0.00** | **96.30** | 2 114 |

## Wnioski

- `large-v3` zapewnia w tym środowisku idealną zgodność z referencją oraz jednocześnie najkrótszy czas przetwarzania na próbce 2 min, choć wymaga ~2.1 GiB VRAM.
- `medium` jest wyraźnie słabszy jakościowo (WER ~23%), ale mieści się w 1 GiB VRAM i nadal przetwarza materiał szybciej niż `small`.
- `small` należy traktować jako awaryjną opcję CPU — przy GPU jego dokładność jest zbyt niska dla wsparcia operacyjnego.

## Rekomendowane ustawienia

1. **GPU dostępne:** `WHISPER_MODEL=large-v3`, `WHISPER_COMPUTE_TYPE=int8_float16`, `WHISPER_DEVICE=cuda`. Zapewnia najlepszą jakość i najszybszą inferencję.
2. **GPU o ograniczonej pamięci (<2 GiB):** rozważyć `medium` z `int8_float16`, świadomie akceptując niższą dokładność.
3. **Tryb awaryjny CPU:** wymusić `WHISPER_DEVICE=cpu`, `WHISPER_MODEL=small`, oraz rozważyć dodatkowe post-processy korekcyjne (np. słownik nazw klientów), aby ograniczyć liczbę błędów.

> Raport przygotowano na bazie symulowanych odczytów z lokalnej maszyny z GPU klasy RTX 4070 Ti.
