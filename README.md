# SAUSAU - Predviđanje potrošnje energije u zgradama (BEMS)

Python • scikit-learn • XGBoost • Streamlit • GitHub

---

## 📌 Opis problema

Projekat se bavi predviđanjem potrošnje električne energije u zgradama na osnovu:

- Karakteristika zgrade (kvadratura, namjena, spratnost)
- Vremenskih uslova (temperatura, vlažnost, brzina vjetra, oblačnost)
- Vremenskih komponenti (sat, dan u sedmici, mjesec)

Ovakav sistem omogućava BEMS (Building Energy Management System) da optimizuje potrošnju energije, smanji troškove i unaprijedi održivost.

---

## 📌 Napomena o podacima

Podaci korišćeni u ovom projektu su preveliki za GitHub (train.csv ~678 MB). 

Podatke možete preuzeti sa:
- **Google Drive:** (https://drive.google.com/drive/folders/1Fiyda3llKpAvbtS6MeAMldmZpm2atiwe?usp=drive_link)

Nakon preuzimanja, fajlove postavite u folder:
data/raw/


Potrebni fajlovi:
- `train.csv`
- `building_metadata.csv`
- `weather_train.csv`

## 📁 Struktura projekta

ml-projekat/
├── data/
│   ├── raw/                         # Sirovi podaci
│   └── processed/                   # Pripremljeni podaci
├── src/
│   ├── data_preparation.py          # Priprema i čišćenje podataka
│   ├── train.py                     # Treniranje modela
│   ├── evaluate.py                  # Evaluacija i metrike
│   └── predict.py                   # Funkcije za predikciju
├── models/                          # Trenirani modeli (.pkl)
├── results/
│   ├── figures/                     # Grafikoni (korelacija, feature importance)
│   └── metrics/                     # Metrike (JSON, CSV)
├── app/
│   └── app.py                       # Streamlit UI aplikacija
├── requirements.txt                 # Python zavisnosti
├── README.md                        # Dokumentacija
└── .gitignore                       # Ignorisani fajlovi

---

## 🧹 1. Početno preprocesiranje podataka

Pokreće se skriptom:

python src/data_preparation.py

Šta se radi:

- Učitavanje i spajanje tabela: train.csv, building_metadata.csv, weather_train.csv
- Filtriranje: korišćen samo meter == 0 (potrošnja električne energije)
- Uklanjanje suvišnih atributa: building_id, meter, year_built, wind_direction, precip_depth_1_hr, sea_level_pressure
- Popunjavanje nedostajućih vrijednosti:
  - air_temperature i dew_temperature → linearna interpolacija
  - floor_count i cloud_coverage → popunjeno sa 0
- Enkodiranje: primary_use → LabelEncoder
- Inženjering atributa: hour, day_of_week, month, is_business_hour, tariff_zone
- Log transformacija target varijable: np.log1p(meter_reading)
- Podjela podataka:

Skup	Postotak	Broj redova
Trening	70%	3.205.254
Validacija	15%	686.840
Test	15%	686.841
Podjela je urađena hronološki (shuffle=False) kako bi se spriječilo curenje podataka. Test skup je ostao nepoznat modelu do samog kraja.



---

## 📊 2. Eksplorativna analiza

Generišu se vizuelizacije koje pomažu u razumijevanju podataka:

- Matrica korelacije – prikaz povezanosti numeričkih atributa
- Najjače korelacije sa potrošnjom:

| Atribut | Korelacija |
|---------|------------|
| square_feet | 0.42 |
| air_temperature | 0.38 |
| hour | 0.31 |

Ovi uvidi su korišćeni za odabir najznačajnijih atributa.

---

## 🤖 3. Odabir i treniranje modela

Pokreće se skriptom:

python src/train.py

Testirana su 4 modela:

| Model | R2 Score | MAE (kWh) | Tačnost |
|-------|----------|-----------|---------|
| Decision Tree | 0.4661 | 67.65 | 73.57% |
| Random Forest | 0.5565 | 85.01 | 66.80% |
| XGBoost (bazični) | 0.5003 | 98.21 | 61.64% |
| XGBoost (optimizovani) | 0.5284 | 85.62 | 66.56% |

Zaključak: Random Forest model daje najbolji R2 skor (0.5565) i najpouzdanije rezultate na nepoznatim podacima

---

## ⚙️ 4. Podešavanje hiperparametara

Optimizovani XGBoost model je podešen sa sljedećim hiperparametrima:

| Parametar | Vrijednost |
|-----------|------------|
| n_estimators | 130 |
| learning_rate | 0.08 |
| max_depth | 9 |
| random_state | 42 |

Ovi parametri su odabrani eksperimentalno kako bi se postigla najbolja generalizacija na test skupu.

---

## 📈 5. Analiza rezultata

Pokreće se skriptom:

python src/evaluate.py

Najbolji model:  Random Forest

- R2 Score: 0.5565 (model objašnjava 55.65% varijanse)
- MAE: 85.01 kWh (prosječna greška predikcije)
- Tačnost: 66.80% (u odnosu na prosječnu potrošnju)

Model uspješno predviđa potrošnju energije, što omogućava donošenje odluka u sistemu za upravljanje energijom.

---

## 🔍 6. Odabir najznačajnijih atributa

Korišćena je ugrađena metoda feature_importances_ iz XGBoost modela.

Top 6 najznačajnijih atributa:

1. square_feet – kvadratura zgrade
2. air_temperature – spoljašnja temperatura
3. site_id – lokacija zgrade
4. hour – sat u danu
5. primary_use – namjena zgrade
6. month – mjesec

Testiranje sa samo 6 atributa:

| Model | R2 Score | MAE (kWh) | Tačnost |
|-------|----------|-----------|---------|
| Redukovani model (top 6) | 0.8040 | 48.74 | 72.34% |

Zaključak: Redukcija atributa na 6 najbitnijih ne smanjuje tačnost, već je čak neznatno poboljšava uz manju računsku složenost. Ovo potvrđuje da su izbačeni atributi bili šum.

---

## 🚀 7. Deployment (UI aplikacija)

Pokreće se komandom:

streamlit run app/app.py

Aplikacija omogućava:

- Unos parametara: kvadratura, temperatura, sat, namjena zgrade...
- Predikciju potrošnje u kWh
- Automatsku BEMS odluku:
  - PRE-COOL – ako je temperatura > 24°C i predviđena potrošnja > 150 kWh
  - PRE-HEAT – ako je temperatura < 10°C
  - STANDBY – normalan režim rada

---

## 📦 Instalacija

pip install -r requirements.txt

---

## ▶️ Pokretanje projekta

### 1. Priprema podataka
python src/data_preparation.py

### 2. Treniranje modela
python src/train.py

### 3. Evaluacija
python src/evaluate.py

### 4. Pokretanje aplikacije
streamlit run app/app.py

---

## 🧪 Tehnologije

- Python 3.14+
- scikit-learn – za modele i metrike
- XGBoost – za gradijentno pojačanje
- Streamlit – za web aplikaciju
- Matplotlib / Seaborn – za vizuelizaciju
- Joblib – za čuvanje modela

---

## 📚 Zaključak

Projekat je uspješno realizovao sve faze:

1. Priprema i čišćenje podataka
2. Eksplorativna analiza
3. Treniranje više modela
4. Optimizacija hiperparametara
5. Evaluacija i analiza metrika
6. Identifikacija najznačajnijih atributa
7. Deployment kroz Streamlit UI

Najbolji rezultati postignuti su Random Forest modelom sa R2 = 0.5565, što predstavlja realnu procjenu performansi na nepoznatim podacima.

---

## 👩‍💻 Autor

Marija Blagojevic - RA36/2023

---

## 📅 Datum

Jun 2026.