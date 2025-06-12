import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cardiotocography_v2.csv")

imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

X = data_imputed.drop('CLASS', axis=1)
y = data_imputed['CLASS']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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

def plot_transforms(data_list, titles):
    fig, axes = plt.subplots(1, len(data_list), figsize=(18, 4))
    for ax, data, title in zip(axes, data_list, titles):
        sns.kdeplot(data[:, 0], ax=ax, fill=True)
        ax.set_title(title)
        ax.set_xlabel("Wartość cechy 0")
    plt.tight_layout()
    plt.show()

plot_transforms(
    [X_train.values, X_train_std, X_train_norm, X_train_minmax, X_train_sel],
    ["Raw", "Standardized", "Normalized", "MinMax", "Feature-Selected"]
)
