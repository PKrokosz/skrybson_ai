# GitHub Actions CI — analiza 5x Why

## Run: commit 5bb90d2 (`docs: document discord recorder bot`)

### Obserwowany symptom
- Workflow `CI / lint-test` zatrzymuje się na kroku `Ruff (lint)`.
- Błędy zgłaszane przez `ruff check .`:
  - `B009` dla wywołań `getattr` z literałami w `ui/bootstrap.py`.
  - `I001` (nieposortowane importy) w modułach `ui/services/align.py` oraz widokach `ui/views/*`.
- Po przejściu kroku lintingu `mypy --config-file pyproject.toml` kończy się błędem `Unused "type: ignore" comment` w `transcribe.py`.

```
$ ruff check .
B009 Do not call `getattr` with a constant attribute value. [...]
I001 Import block is un-sorted or un-formatted [...]

$ mypy --config-file pyproject.toml
transcribe.py:244: error: Unused "type: ignore" comment
```

### 5x Why — ruff (B009)
1. **Dlaczego** workflow zakończył się niepowodzeniem? → `ruff check .` zwrócił kod wyjścia ≠ 0.
2. **Dlaczego** `ruff` zgłosił `B009`? → `ui/bootstrap.py` używa `getattr(module, "ToolTip")` dla stałej nazwy atrybutu.
3. **Dlaczego** zastosowano `getattr` dla stałej nazwy? → Wzorzec miał zabezpieczać opcjonalny import `ttkbootstrap`, kopiując dynamiczny styl dostępu.
4. **Dlaczego** fallback potrzebował "dynamicznego" dostępu? → Autor chciał uniknąć `AttributeError`, zakładając, że biblioteka może nie mieć `ToolTip`.
5. **Dlaczego** nie wykryto sprzeczności z zasadą B009 przed push? → `ruff` nie został uruchomiony lokalnie ani jako hook pre-commit przed wysłaniem commitu.

**Działania korygujące**: zastąpić `getattr` bezpośrednim dostępem do atrybutu (`tooltip_mod.ToolTip`), dołączyć hook pre-commit/CI w lokalnym workflow.

### 5x Why — ruff (I001)
1. **Dlaczego** CI dalej raportuje błędy? → `ruff` sygnalizuje nieposortowane importy (`I001`).
2. **Dlaczego** importy są nieposortowane? → Sekcje `from __future__ import annotations` i standardowe importy zostały ręcznie przemieszczone (np. podczas refaktoryzacji widoków UI), łamiąc konwencję isort.
3. **Dlaczego** refaktoryzacja naruszyła kolejność? → Edycja odbywała się bez użycia automatycznych narzędzi (`ruff --fix`, `isort`).
4. **Dlaczego** narzędzia nie zostały uruchomione? → Brak wpisu w dokumentacji "Jak rozwijać" wymuszającego uruchomienie `ruff --fix` po zmianach importów.
5. **Dlaczego** dokumentacja tego nie wspomina? → Do tej pory importy były w większości utrzymywane przez IDE, więc potrzeba nie została odnotowana.

**Działania korygujące**: uruchamiać `ruff check --select I --fix`, dodać wzmiankę w README/CONTRIBUTING o obowiązkowym sortowaniu importów.

### 5x Why — mypy (Unused "type: ignore")
1. **Dlaczego** krok `Mypy (type check)` kończy się błędem? → `mypy` zwraca kod ≠ 0 z powodu nieużytego komentarza `# type: ignore` w `transcribe.py`.
2. **Dlaczego** komentarz jest nieużywany? → Aktualna wersja `mypy` posiada stuba dla `faster_whisper.vad`, więc import `VadOptions` nie generuje błędu typu.
3. **Dlaczego** komentarz pozostał w kodzie? → Był potrzebny w starszych wersjach (bez stuba) i nie został usunięty przy aktualizacji dependencji.
4. **Dlaczego** aktualizacja dependencji nie obejmowała sprzątania komentarzy? → Brak checklisty regresyjnej obejmującej usunięcie zbędnych `type: ignore` po upgrade'ach.
5. **Dlaczego** nie istnieje taka checklist? → Zespół polegał na tym, że `mypy` zignoruje redundantny komentarz; nie uwzględniono, że `--warn-unused-ignores` jest włączone.

**Działania korygujące**: usunąć `# type: ignore` lub ograniczyć import do bloku `typing.TYPE_CHECKING`, zachować checklistę po aktualizacji zależności.

### Podsumowanie działań zapobiegawczych
- Uruchamiać lokalnie pełny pipeline (`ruff`, `mypy`, `pytest`) przed push.
- Rozważyć dodanie hooków pre-commit wymuszających lint i sortowanie importów.
- Uzupełnić dokumentację developerską o sekcję "Checklist przed PR" z powyższymi punktami.
