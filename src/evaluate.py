import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
import os
import json

def evaluate_models():
    """
    Evaluacija na TEST skupu (NEPOZNAT podaci)
    """
    # Učitaj podatke (SVA TRI SKUPA)
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load('data/processed/prepared_data.pkl')
    
    # Učitaj modele
    xgb_bazicni = joblib.load('models/xgb_bazicni.pkl')
    xgb_hiper = joblib.load('models/xgb_hiper.pkl')
    rf_model = joblib.load('models/rf_model.pkl')
    dt_model = joblib.load('models/dt_model.pkl')
    
    print("=" * 60)
    print("  KONAČNA EVALUACIJA NA TEST SKUPU (NEPOZNAT PODACI)")
    print("=" * 60)
    
    # Predikcije na TEST skupu
    print("\n1. Generisanje predikcija na test skupu...")
    y_pred_bazicni = xgb_bazicni.predict(X_test)
    y_pred_hiper = xgb_hiper.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_dt = dt_model.predict(X_test)
    
    # Inverzna transformacija
    if y_test.max() < 20:
        print("   Detektovana logaritamska skala - inverzna transformacija...")
        y_test_stvarni = np.expm1(y_test)
        y_pred_bazicni = np.expm1(y_pred_bazicni)
        y_pred_hiper = np.expm1(y_pred_hiper)
        y_pred_rf = np.expm1(y_pred_rf)
        y_pred_dt = np.expm1(y_pred_dt)
    else:
        y_test_stvarni = y_test
    
    # Metrike
    print("\n2. Računanje metrika na test skupu...")
    
    models = {
        'Decision Tree': y_pred_dt,
        'Random Forest': y_pred_rf,
        'XGBoost Bazični': y_pred_bazicni,
        'XGBoost Optimizovani': y_pred_hiper
    }
    
    results = []
    for name, pred in models.items():
        r2 = r2_score(y_test_stvarni, pred)
        mae = mean_absolute_error(y_test_stvarni, pred)
        accuracy_pct = (1 - (mae / y_test_stvarni.mean())) * 100
        results.append({
            'Model': name,
            'R2 Score': r2,
            'MAE (kWh)': mae,
            'Tačnost (%)': accuracy_pct
        })
    
    df_results = pd.DataFrame(results)
    
    # Sačuvaj metrike
    os.makedirs('results/metrics', exist_ok=True)
    df_results.to_csv('results/metrics/model_comparison.csv', index=False)
    
    print("\n" + "=" * 70)
    print("  IZVEŠTAJ - UPOREDNA ANALIZA MODELA (TEST SKUP - NEPOZNAT)")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)
    
    with open('results/metrics/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✅ Metrike sačuvane u 'results/metrics/'")
    
    # ----- GRAFIKONI -----
    print("\n3. Generisanje grafikona...")
    os.makedirs('results/figures', exist_ok=True)
    
    # 1. Matrica korelacije
    print("   - Matrica korelacije...")
    sample_size = min(100000, len(X_test))
    feature_cols = joblib.load('data/processed/feature_columns.pkl')
    sample_df = pd.concat([
        pd.DataFrame(X_test[:sample_size], columns=feature_cols),
        pd.Series(np.expm1(y_test[:sample_size]), name='meter_reading')
    ], axis=1)
    
    numeric_cols = ['meter_reading', 'square_feet', 'floor_count', 
                    'air_temperature', 'cloud_coverage', 'dew_temperature', 'wind_speed']
    existing_cols = [col for col in numeric_cols if col in sample_df.columns]
    
    if len(existing_cols) > 1:
        corr_matrix = sample_df[existing_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Matrica korelacije atributa')
        plt.tight_layout()
        plt.savefig('results/figures/correlation_matrix.png')
        plt.close()
    
    # 2. Feature importance
    print("   - Feature importance...")
    feature_imp = pd.DataFrame({
        'Atribut': feature_cols,
        'Značajnost': xgb_hiper.feature_importances_
    }).sort_values('Značajnost', ascending=False)
    
    plt.figure(figsize=(12, 6.5))
    sns.barplot(data=feature_imp, x='Značajnost', y='Atribut', 
                hue='Atribut', palette='magma', legend=False)
    plt.title('Rang lista najznačajnijih atributa')
    plt.xlabel('Nivo značajnosti')
    plt.ylabel('Atribut')
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance.png')
    plt.close()
    
    # 3. Stvarno vs Predviđeno za SVE modele
    print("   - Stvarno vs Predviđeno za sve modele...")
    
    models_for_plot = {
        'Decision Tree': y_pred_dt,
        'Random Forest': y_pred_rf,
        'XGBoost Bazični': y_pred_bazicni,
        'XGBoost Optimizovani': y_pred_hiper
    }
    
    for name, pred in models_for_plot.items():
        plt.figure(figsize=(8, 8))
        
        # Uzmi uzorak za grafik (5000 tačaka)
        sample_idx = np.random.choice(len(y_test_stvarni), min(5000, len(y_test_stvarni)), replace=False)
        
        plt.scatter(y_test_stvarni.iloc[sample_idx], pred[sample_idx], alpha=0.3, s=10)
        plt.plot([0, y_test_stvarni.max()], [0, y_test_stvarni.max()], 'r--', linewidth=2, label='Savršena predikcija')
        
        plt.xlabel('Stvarna potrošnja (kWh)')
        plt.ylabel('Predviđena potrošnja (kWh)')
        plt.title(f'Stvarno vs Predviđeno - {name}')
        plt.legend()
        plt.tight_layout()
        
        # Sačuvaj svaki grafikon posebno
        filename = f'actual_vs_predicted_{name.replace(" ", "_")}.png'
        plt.savefig(f'results/figures/{filename}')
        plt.close()
        
        print(f"      ✅ {filename} sačuvan")
    
    print("\n✅ Grafikoni sačuvani u 'results/figures/'")
    print("=" * 60)
    print("  EVALUACIJA ZAVRŠENA!")
    print("  ✅ Test skup je ostao NEPOZNAT modelu!")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    evaluate_models()