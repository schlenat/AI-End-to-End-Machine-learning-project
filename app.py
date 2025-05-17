import gradio as gr
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer

# Modell und Feature-Namen laden
with open("model.pkl", "rb") as f:
    model, feature_names = pickle.load(f)

# Imputer vorbereiten (gleich wie beim Training)
imputer = SimpleImputer(strategy="mean")

# Beispiel-Features definieren (vereinfachte Eingabe)
def predict_price(brand, model_name, year, milage, fuel_type, transmission, country, income, gdpp):
    # Manuelle Feature-Vorbereitung
    input_dict = {
        "model_year": int(year),
        "milage": float(milage),
        "income": float(income),
        "gdpp": float(gdpp),
        "price_vs_gdpp": 0,  # wird vom Modell ignoriert
    }

    # Dummy-Kodierungen
    columns = list(feature_names)
    for col in columns:
        if col.startswith("brand_"):
            input_dict[col] = 1.0 if col == f"brand_{brand}" else 0.0
        elif col.startswith("fuel_type_"):
            input_dict[col] = 1.0 if col == f"fuel_type_{fuel_type}" else 0.0
        elif col.startswith("transmission_"):
            input_dict[col] = 1.0 if col == f"transmission_{transmission}" else 0.0
        elif col.startswith("country_"):
            input_dict[col] = 1.0 if col == f"country_{country}" else 0.0
        elif col.startswith("model_"):
            input_dict[col] = 1.0 if col == f"model_{model_name}" else 0.0
        elif col not in input_dict:
            input_dict[col] = 0.0

    # DataFrame vorbereiten
    X_input = pd.DataFrame([input_dict])[columns]
    X_input = imputer.fit(X_input).transform(X_input)
    
    prediction = model.predict(X_input)[0]
    return f"Geschätzter Preis: {prediction:,.0f} CHF"

# UI definieren
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(label="Marke", choices=["BMW", "Toyota", "Ford", "Volkswagen"]),
        gr.Textbox(label="Modellbezeichnung"),
        gr.Number(label="Baujahr", value=2018),
        gr.Number(label="Kilometerstand", value=50000),
        gr.Dropdown(label="Treibstoff", choices=["Petrol", "Diesel", "Hybrid", "Electric"]),
        gr.Dropdown(label="Getriebe", choices=["Automatic", "Manual"]),
        gr.Dropdown(label="Land", choices=["Germany", "Switzerland", "France", "Italy"]),
        gr.Number(label="Durchschnittseinkommen", value=40000),
        gr.Number(label="BIP pro Kopf", value=35000),
    ],
    outputs=gr.Textbox(label="Vorhersage"),
    title="Gebrauchtwagen-Preisvorhersage",
    description="Basierend auf einem trainierten Modell (Linear Regression)",
)

demo.launch()