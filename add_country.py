import pandas as pd
import random

# CSV-Datei laden
df = pd.read_csv("used_cars.csv")

# Liste realistischer Länder aus deinem country_data.csv
countries = [
    "Germany", "France", "Italy", "Spain", "Switzerland",
    "Netherlands", "Austria", "Belgium", "Norway", "Sweden"
]

# Zufällig Länder zuweisen
df["country"] = [random.choice(countries) for _ in range(len(df))]

# Neue Datei speichern
df.to_csv("used_cars_with_country.csv", index=False)

print("Neue Datei mit Länderspalte gespeichert als 'used_cars_with_country.csv'")
