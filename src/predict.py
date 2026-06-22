import numpy as np
import pandas as pd
import joblib

def load_model(model_name='rf_model'):
    """
    Učitava model.
    
    Parametri:
    - model_name: 'rf_model' (Random Forest), 'xgb_hiper', 'xgb_bazicni', 'dt_model'
    
    Returns:
    - Učitani model
    """
    return joblib.load(f'models/{model_name}.pkl')

def load_scaler():
    """Učitava StandardScaler koji je korišten za skaliranje podataka."""
    return joblib.load('models/scaler.pkl')

def load_feature_columns():
    """Učitava nazive feature kolona koje model očekuje."""
    return joblib.load('data/processed/feature_columns.pkl')

def predict_single(features, model_name='rf_model'):
    """
    Predikcija za JEDAN primjer.
    
    Parametri:
    - features: dict sa vrijednostima za sve feature-e
    - model_name: koji model koristiti (default: 'rf_model')
    
    Returns:
    - Predviđena potrošnja u kWh
    """
    # Učitaj model, scaler i feature kolone
    model = load_model(model_name)
    scaler = load_scaler()
    columns = load_feature_columns()
    
    # Pretvori dict u DataFrame sa ispravnim redoslijedom kolona
    df = pd.DataFrame([features])[columns]
    
    # Skaliraj podatke
    df_scaled = scaler.transform(df)
    
    # Predikcija
    pred_log = model.predict(df_scaled)[0]
    
    # Inverzna log transformacija
    pred = np.expm1(pred_log)
    
    return pred

if __name__ == "__main__":
    print("=" * 60)
    print("  🔮 PREDIKCIJA - TEST PRIMJER")
    print("=" * 60)
    
    # Test primjer sa fiktivnim podacima
    test_features = {
        'site_id': 0,
        'primary_use': 0,
        'square_feet': 10000,
        'floor_count': 5,
        'air_temperature': 22.0,
        'cloud_coverage': 0.3,
        'dew_temperature': 12.0,
        'wind_speed': 4.0,
        'hour': 14,
        'day_of_week': 2,
        'month': 7,
        'is_business_hour': 1,
        'tariff_zone': 1
    }
    
    print("\n🏢 Test unos:")
    for key, value in test_features.items():
        print(f"   {key}: {value}")
    
    # Predikcija
    prediction = predict_single(test_features, 'rf_model')
    
    print(f"\n✅ Predviđena potrošnja: {prediction:.2f} kWh")
    print("=" * 60)