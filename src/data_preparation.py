import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_prepare_data():
    """
    Učitava i priprema sve podatke sa VALIDACIONIM skupom
    """
    print("=" * 60)
    print("  PRIPREMA PODATAKA SA VALIDACIONIM SKUPOM")
    print("=" * 60)
    
    print("\n1. Korak: Učitavanje sirovih podataka...")
    df_zgrade = pd.read_csv('data/raw/building_metadata_popravljeno.csv')
    df_vreme = pd.read_csv('data/raw/weather_train.csv')
    df_potrosnja = pd.read_csv('data/raw/train.csv')
    
    print("2. Korak: Spajanje tabela...")
    df_potrosnja = df_potrosnja[df_potrosnja['meter'] == 0]
    projekat_df = pd.merge(df_potrosnja, df_zgrade, on='building_id', how='inner')
    projekat_df = pd.merge(projekat_df, df_vreme, on=['site_id', 'timestamp'], how='inner')
    
    print("3. Korak: Uklanjanje suvišnih atributa...")
    kolone_za_izbacivanje = ['building_id', 'meter', 'year_built', 'wind_direction', 
                             'precip_depth_1_hr', 'sea_level_pressure']
    projekat_df = projekat_df.drop(columns=kolone_za_izbacivanje, errors='ignore')
    
    print("4. Korak: Popunjavanje nedostajućih vrijednosti...")
    projekat_df['air_temperature'] = projekat_df['air_temperature'].interpolate(method='linear')
    projekat_df['dew_temperature'] = projekat_df['dew_temperature'].interpolate(method='linear')
    projekat_df['floor_count'] = projekat_df['floor_count'].fillna(0)
    projekat_df['cloud_coverage'] = projekat_df['cloud_coverage'].fillna(0)
    
    print("5. Korak: Ekstrakcija vremenskih komponenti...")
    projekat_df['timestamp'] = pd.to_datetime(projekat_df['timestamp'])
    projekat_df['hour'] = projekat_df['timestamp'].dt.hour
    projekat_df['day_of_week'] = projekat_df['timestamp'].dt.dayofweek
    projekat_df['month'] = projekat_df['timestamp'].dt.month
    
    projekat_df['is_business_hour'] = np.where(
        (projekat_df['day_of_week'] < 5) & 
        (projekat_df['hour'] >= 7) & 
        (projekat_df['hour'] <= 19), 1, 0
    )
    projekat_df['tariff_zone'] = np.where(
        (projekat_df['hour'] >= 8) & 
        (projekat_df['hour'] <= 22), 1, 0
    )
    
    print("6. Korak: Label Encoding...")
    le_clean = LabelEncoder()
    projekat_df['primary_use'] = le_clean.fit_transform(projekat_df['primary_use'].astype(str))
    
    print("7. Korak: Priprema feature-a i target-a...")
    ulazne_kolone = [
        'site_id', 'primary_use', 'square_feet', 'floor_count', 
        'air_temperature', 'cloud_coverage', 'dew_temperature', 'wind_speed',
        'hour', 'day_of_week', 'month', 'is_business_hour', 'tariff_zone'
    ]
    
    X = projekat_df[ulazne_kolone]
    y = np.log1p(projekat_df['meter_reading'])
    
    print("8. Korak: PODJELA NA TRENING, VALIDACIJU I TEST (70/15/15)...")
    
    # Hronološka podjela (shuffle=False)
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.85)  # 70% + 15%
    
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:val_size]
    X_test = X.iloc[val_size:]
    
    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:val_size]
    y_test = y.iloc[val_size:]
    
    print(f"   ✅ Trening skup: {X_train.shape[0]} redova (70%)")
    print(f"   ✅ Validacioni skup: {X_val.shape[0]} redova (15%)")
    print(f"   ✅ Test skup: {X_test.shape[0]} redova (15%)")
    
    print("9. Korak: Skaliranje...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("10. Korak: Čuvanje podataka...")
    os.makedirs('data/processed', exist_ok=True)
    
    # Sačuvaj sve skupove
    joblib.dump((X_train_scaled, X_val_scaled, X_test_scaled, 
                 y_train, y_val, y_test), 
                'data/processed/prepared_data.pkl')
    joblib.dump(ulazne_kolone, 'data/processed/feature_columns.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Sačuvaj i kao CSV za pregled
    pd.DataFrame(X_train_scaled, columns=ulazne_kolone).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_val_scaled, columns=ulazne_kolone).to_csv('data/processed/X_val.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=ulazne_kolone).to_csv('data/processed/X_test.csv', index=False)
    pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_val).to_csv('data/processed/y_val.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/y_test.csv', index=False)
    
    print("   ✅ CSV fajlovi sačuvani u data/processed/")
    
    print("\n" + "=" * 60)
    print("  ✅ PRIPREMA PODATAKA ZAVRŠENA!")
    print("=" * 60)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, ulazne_kolone

if __name__ == "__main__":
    load_and_prepare_data()