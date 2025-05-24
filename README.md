# AI-End-to-End-Machine-learning-project

# 🚗 Gebrauchtwagenpreis-Vorhersage

## 📄 Projektbeschreibung  
Ziel dieses Projekts ist es, den Preis von Gebrauchtwagen vorherzusagen. Dazu werden sowohl fahrzeugspezifische Merkmale (wie Kilometerstand, Marke, Modell) als auch landesspezifische Wirtschaftsdaten (z. B. BIP, Einkommen) genutzt.

---

## 📈 Ergebnisse  
Das Modell der linearen Regression zeigte ein gutes Fit (R² = 0.96), allerdings ist der RMSE (≈28'375 CHF) relativ hoch.  
Der Random Forest Regressor zeigte ein deutlich höheres Overfitting-Verhalten mit R² = 0.52 bei RMSE = 99'482 CHF.  
Dies deutet darauf hin, dass das Modell stark durch einige dominante Features beeinflusst wird oder zusätzliche Feature Engineering nötig ist.

---

## 🔗 URLs

| Name         | Link     |
|--------------|----------|
| Huggingface  | *folgt*  |
| GitHub       | *folgt*  |

---

## 📦 Datenquellen und verwendete Features

| Datenquelle       | Features                                                                 |
|-------------------|--------------------------------------------------------------------------|
| `used_cars.csv`   | `brand`, `model`, `milage`, `fuel_type`, `transmission`, `price`, `country` |
| `country_data.csv`| `income`, `gdpp`                                                         |

---

## 🛠 Generierte Features

| Feature               | Beschreibung                                                      |
|------------------------|-------------------------------------------------------------------|
| `milage`              | Kilometerstand bereinigt (numerisch)                              |
| `relative_price`      | Verhältnis Preis zu mittlerem Einkommen des Landes (`price/income`) |
| `price_vs_gdpp`       | Verhältnis Preis zu BIP pro Kopf (`price/gdpp`)                   |
| One-Hot Encoding       | Für `model`, `brand`, `fuel_type`, `transmission`, `country`     |

---

## 🧪 Modellierung

**Train/Test-Split**: 80/20 Split mit `random_state=42`  
**Imputation**: Fehlende Werte wurden mit Mittelwert ersetzt (`SimpleImputer`)

### Trainierte Modelle
- `LinearRegression` (Scikit-learn)
- `RandomForestRegressor` (Scikit-learn, 100 Trees, Default)

### Evaluation

| Modell            | RMSE         | R²     |
|-------------------|--------------|--------|
| Linear Regression | 28'375.74 CHF| 0.96   |
| Random Forest     | 99'481.67 CHF| 0.52   |

---

## 📊 Wichtigste Features (Random Forest)

Die wichtigsten Merkmale laut Feature Importance Visualisierung:

1. `milage`  
2. `gdpp`  
3. `income`  
4. Modell-spezifische Dummy-Variablen (`model_<X>`)




