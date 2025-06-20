import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cardiotocography_v2.csv")

print("Podstawowe statystyki:")
print(df.describe())

class_counts = df["CLASS"].value_counts().sort_index()
print("\nRozkład klas:")
print(class_counts)

plt.figure(figsize=(10, 5))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Rozkład klas (CLASS)")
plt.xlabel("Klasa")
plt.ylabel("Liczba próbek")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

missing_values = df.isnull().sum().sort_values(ascending=False)
print("\nBrakujące wartości:")
print(missing_values[missing_values > 0])

plt.figure(figsize=(10, 5))
missing_values[missing_values > 0].plot(kind='bar')
plt.title("Brakujące dane w kolumnach")
plt.ylabel("Liczba brakujących wartości")
plt.xlabel("Cecha")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 5. Korelacje między cechami
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Mapa korelacji między cechami")
plt.tight_layout()
plt.show()
