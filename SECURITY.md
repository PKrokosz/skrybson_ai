# Security Guidelines

- **Sekrety trzymamy lokalnie.** W repozytorium wersjonowany jest tylko plik
  `.env.example`. Własny `.env` przechowuj poza Gitem i nigdy nie commituj
  tokenów Discorda.
- **Dostępy ograniczone do minimum.** Token bota Discorda generuj tylko z
  uprawnieniami niezbędnymi do pobierania nagrań.
- **Repozytorium jest lokalne.** Materiały audio i transkrypcje zawierające dane
  wrażliwe trzymaj na zasobach kontrolowanych przez zespół. Narzędzie nie wysyła
  żadnych danych na zewnątrz.
- **Audyt środowiska przed uruchomieniem.** Zweryfikuj, że katalogi
  `recordings/` oraz `out/` są dostępne tylko dla uprawnionych użytkowników.
- **CI bez sekretów.** Workflow w `.github/workflows/ci.yml` działa wyłącznie na
  mockach i nie wymaga żadnych kluczy dostępowych.
