import pandas as pd
import numpy as np
import joblib
import time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os

def train_models():
    """
    Trenira sve modele koristeći TREINING + VALIDACIJU
    Test se koristi SAMO na kraju za evaluaciju
    """
    # Učitaj podatke (SVA TRI SKUPA)
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load('data/processed/prepared_data.pkl')
    
    print("=" * 60)
    print("  TRENIRANJE MODELA SA VALIDACIONIM SKUPOM")
    print("=" * 60)
    
    # ----- BAZIČNI XGBoost -----
    print("\n1. Treniranje BAZIČNOG XGBoost modela...")
    xgb_bazicni = XGBRegressor(random_state=42, n_jobs=-1)
    start = time.time()
    xgb_bazicni.fit(X_train, y_train)
    print(f"   ✅ Bazični XGBoost treniran za {time.time() - start:.2f} sekundi!")
    
    # ----- OPTIMIZOVANI XGBoost (SA VALIDACIJOM) -----
    print("\n2. Treniranje OPTIMIZOVANOG XGBoost modela...")
    xgb_hiper = XGBRegressor(
        n_estimators=130,
        learning_rate=0.08,
        max_depth=9,
        random_state=42,
        n_jobs=-1
    )
    start = time.time()
    xgb_hiper.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],  # ← VALIDACIJA, NE TEST!
        verbose=15
    )
    print(f"   ✅ Optimizovani XGBoost treniran za {time.time() - start:.2f} sekundi!")
    
    # ----- RANDOM FOREST -----
    print("\n3. Treniranje RANDOM FOREST modela...")
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    start = time.time()
    rf_model.fit(X_train, y_train)
    print(f"   ✅ Random Forest treniran za {time.time() - start:.2f} sekundi!")
    
    # ----- DECISION TREE -----
    print("\n4. Treniranje DECISION TREE modela...")
    dt_model = DecisionTreeRegressor(random_state=42)
    start = time.time()
    dt_model.fit(X_train, y_train)
    print(f"   ✅ Decision Tree treniran za {time.time() - start:.2f} sekundi!")
    
    # Sačuvaj modele
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_bazicni, 'models/xgb_bazicni.pkl')
    joblib.dump(xgb_hiper, 'models/xgb_hiper.pkl')
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(dt_model, 'models/dt_model.pkl')
    
    # ----- EVALUACIJA NA TEST SKUPU (TEK NA KRAJU) -----
    print("\n📊 KONAČNA EVALUACIJA NA TEST SKUPU (NEPOZNAT PODACI)...")
    
    y_pred_bazicni = xgb_bazicni.predict(X_test)
    y_pred_hiper = xgb_hiper.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_dt = dt_model.predict(X_test)
    
    # Inverzna transformacija
    if y_test.max() < 20:
        y_test_stvarni = np.expm1(y_test)
        y_pred_bazicni = np.expm1(y_pred_bazicni)
        y_pred_hiper = np.expm1(y_pred_hiper)
        y_pred_rf = np.expm1(y_pred_rf)
        y_pred_dt = np.expm1(y_pred_dt)
    else:
        y_test_stvarni = y_test
    
    results = []
    for name, pred in [('Bazični XGBoost', y_pred_bazicni),
                       ('Optimizovani XGBoost', y_pred_hiper),
                       ('Random Forest', y_pred_rf),
                       ('Decision Tree', y_pred_dt)]:
        r2 = r2_score(y_test_stvarni, pred)
        mae = mean_absolute_error(y_test_stvarni, pred)
        accuracy = (1 - (mae / y_test_stvarni.mean())) * 100
        results.append({
            'Model': name,
            'R2 Score': r2,
            'MAE (kWh)': mae,
            'Tačnost (%)': accuracy
        })
    
    # Prikaži rezultate
    print("\n" + "=" * 70)
    print("  IZVEŠTAJ - UPOREDNA ANALIZA MODELA (TEST SKUP - NEPOZNAT)")
    print("=" * 70)
    for r in results:
        print(f"  {r['Model']}:")
        print(f"    R2 Score: {r['R2 Score']:.4f}")
        print(f"    MAE: {r['MAE (kWh)']:.2f} kWh")
        print(f"    Tačnost: {r['Tačnost (%)']:.2f}%")
        print()
    print("=" * 70)
    
    print("\n✅ Svi modeli sačuvani u 'models/' folderu!")
    print("✅ Test skup je ostao NEPOZNAT modelu do samog kraja!")
    
    return xgb_bazicni, xgb_hiper, rf_model, dt_model

if __name__ == "__main__":
    train_models()