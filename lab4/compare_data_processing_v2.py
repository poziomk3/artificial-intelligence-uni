# === Importy i dane ===
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 1. Wczytanie i imputacja
df = pd.read_csv("cardiotocography_v2.csv")
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 2. Podział na zbiór treningowy i walidacyjny
X = data_imputed.drop('CLASS', axis=1)
y = data_imputed['CLASS']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Różne przetwarzania
scaler = StandardScaler()
normalizer = Normalizer()
minmax = MinMaxScaler()
selector = SelectKBest(score_func=f_classif, k=10)

X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)

X_train_norm = normalizer.fit_transform(X_train)
X_val_norm = normalizer.transform(X_val)

X_train_minmax = minmax.fit_transform(X_train)
X_val_minmax = minmax.transform(X_val)

X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel = selector.transform(X_val)

# 4. Klasyfikacja
datasets = {
    "Raw": (X_train, X_val),
    "Standardized": (X_train_std, X_val_std),
    "Normalized": (X_train_norm, X_val_norm),
    "MinMax": (X_train_minmax, X_val_minmax),
    "Feature-Selected": (X_train_sel, X_val_sel),
}

results = []
for name, (X_tr, X_vl) in datasets.items():
    clf = GaussianNB()
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_vl)
    results.append({
        "Processing": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_val, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_val, y_pred, average='weighted', zero_division=0)
    })

results_df = pd.DataFrame(results)

# 5. Wykres
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
plt.figure(figsize=(14, 6))

for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 4, i)
    plt.bar(results_df["Processing"], results_df[metric])
    plt.title(metric)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.ylabel("Score")

plt.suptitle("GaussianNB – Porównanie metryk dla różnych metod przetwarzania danych")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
