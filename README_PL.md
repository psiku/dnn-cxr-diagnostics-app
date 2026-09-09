# Załącznik 1. Uruchomienie aplikacji

Przyjęto model działania aplikacji wymagający lokalnego uruchomienia środowiska, zgodnie z poniższym opisem.

Do uruchomienia aplikacji wymagane jest posiadanie zainstalowanego narzędzia Docker Desktop wraz z Docker Compose w wersji 2, a także plików z wagami modeli, które ze względu na rozmiar nie są przechowywane w repozytorium i są do pobrania za pomocą linku. Uruchomienie w wariancie bez konteneryzacji wymaga dodatkowo środowiska Python w wersji 3.12 oraz Node.js w wersji 22 wraz z menedżerem pakietów npm.

Przygotowanie aplikacji do uruchomienia przebiega w sposób opisany poniżej:

1. Pobranie kodu źródłowego z repozytorium przy użyciu polecenia:
   `git clone https://github.com/psiku/dnn-cxr-diagnostics-app.git`
2. Inicjalizacja środowiska konteneryzacji poprzez uruchomienie oprogramowania Docker Desktop.
3. Pobranie plików z wagami modeli modeli:
   - Model segmentacyjny: https://drive.google.com/file/d/1yNCaLjKooiDQ2dDEfdK2K88s9_E8hpJS/view?usp=sharing. Następnie wagi modelu ***segmentation_model_bestMSE.pt*** należy przerzucić do folderu ***backend/models/***
   - Wagi modeli klasyfikacyjnych: https://drive.google.com/file/d/1Y3vn89wdX1VUaYGRxlniGHi4ab7fSFWv/view?usp=sharing
4. Umieszczenie wszystkich plików konfiguracyjnych w odpowiednich miesjcach. Pliki konfiguracyjne modeli oraz pliki z progami znajdują się w folderze **example_config_files**.

Pliki wymagane do uruchomienia serwera:

* `backend/models/app/model.ckpt` – wagi sieci klasyfikacyjnej DenseNet-161 rozpoznającej 14 klas patologii.
* `backend/models/segmentation_model_bestMSE.pt` – wagi modelu segmentacji HybridGNet wyznaczającego obszar płuc i tchawicy.
* `backend/config/thresholds.json` – progi decyzyjne wyznaczone dla poszczególnych klas patologii.
* `backend/config/model_config.yml` – plik konfiguracyjny opisujący architekturę sieci oraz ścieżki do powyższych plików.

Po wykonaniu poprzednich kroków, celem uruchomienia aplikacji należy wykonać poniższe kroki wewnątrz folderu głównego repozytorium:

1. Wykonanie komendy `docker compose up --build`
2. Odczekanie na zakończenie procesu budowania obrazów, które przy pierwszym uruchomieniu może trwać kilkanaście minut ze względu na instalację biblioteki PyTorch
3. Przejście w przeglądarce internetowej pod adres `http://localhost:8080`

Uruchomione środowisko udostępnia interfejs użytkownika pod adresem `http://localhost:8080`, interfejs programistyczny serwera pod adresem `http://localhost:8080/api/` oraz interaktywną dokumentację API w standardzie OpenAPI pod adresem `http://localhost:8080/docs`. Korzystanie z aplikacji z innego urządzenia znajdującego się w tej samej sieci lokalnej jest możliwe po zastąpieniu nazwy `localhost` adresem IP komputera, na którym uruchomiono środowisko. Zatrzymanie aplikacji następuje po wykonaniu komendy `docker compose down`.

Alternatywnie możliwe jest uruchomienie aplikacji bez użycia konteneryzacji. W tym celu należy wykonać poniższe kroki wewnątrz folderu głównego repozytorium:

1. Wykonanie komendy `cd backend`
2. Wykonanie komendy `python -m venv .venv` w celu utworzenia wirtualnego środowiska Pythona
3. Wykonanie komendy `.venv\Scripts\activate` w systemie Windows lub `source .venv/bin/activate` w systemach uniksowych
4. Wykonanie komendy `pip install -r requirements.txt`
5. Wykonanie komendy `python main.py`, co powoduje uruchomienie serwera pod adresem `http://127.0.0.1:8000`
6. Wykonanie w oddzielnym oknie terminala komendy `cd frontend`
7. Wykonanie komendy `npm install`
8. Wykonanie komendy `npm run dev`, co powoduje uruchomienie aplikacji klienckiej pod adresem `http://localhost:5173`

Podczas uruchamiania aplikacji w środowisku lokalnym należy zastąpić ścieżki względne w plikach konfiguracyjnych ścieżkami bezwzględnymi.

Wyniki pracy aplikacji, obejmujące opisy badań, wygenerowane raporty w formacie PDF oraz obrazy z naniesionymi etykietami, zapisywane są w katalogu `backend/data/annotations`. W wariancie wykorzystującym konteneryzację katalog ten jest współdzielony z systemem plików komputera, dzięki czemu zgromadzone dane pozostają dostępne również po zatrzymaniu kontenerów.

Poprawność działania warstwy serwerowej można zweryfikować poprzez uruchomienie testów jednostkowych i integracyjnych komendą `pytest` wykonaną wewnątrz katalogu `backend` z aktywnym wirtualnym środowiskiem Pythona.
