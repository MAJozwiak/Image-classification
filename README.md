# Image Classification API

Projekt do trenowania modelu klasyfikacji obrazów oraz udostępniania go przez API.



## Instrukcja uruchomienia

### 1. Utwórz i aktywuj wirtualne środowisko

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\activate

### 2. Zainstaluj wymagane pakiety
pip install -r requirements.txt

### 3. Uruchom serwer API
uvicorn app:app --reload

### 4. Korzystanie z API
Otwórz w przeglądarce:
http://127.0.0.1:8000/docs

Tam możesz testować endpointy:

POST /train – trenowanie modelu

POST /test – testowanie modelu, zwraca accuracy

POST /predict – klasyfikacja pojedynczego zdjęcia
