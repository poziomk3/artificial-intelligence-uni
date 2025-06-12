import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("cardiotocography_v2.csv")

imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

X = data_imputed.drop('CLASS', axis=1)
y = data_imputed['CLASS']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)

selector = SelectKBest(score_func=f_classif, k=10)
X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel = selector.transform(X_val)

data_variants = {
    "Standardized": (X_train_std, X_val_std),
    "Feature-Selected": (X_train_sel, X_val_sel)
}

def get_classifiers():
    nb1 = GaussianNB(var_smoothing=1e-9)
    nb2 = GaussianNB(var_smoothing=1e-4)
    nb3 = GaussianNB(var_smoothing=1e-1)

    dt1 = DecisionTreeClassifier(max_depth=3, min_samples_split=2, criterion='gini', random_state=42)
    dt2 = DecisionTreeClassifier(max_depth=10, min_samples_split=5, criterion='entropy', random_state=100)
    dt3 = DecisionTreeClassifier(max_depth=None, min_samples_split=20, min_samples_leaf=10, random_state=80)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42)

    return {
        'NB-smooth=1e-9': nb1,
        'NB-smooth=1e-4': nb2,
        'NB-smooth=1e-1': nb3,
        'DT-depth=3': dt1,
        'DT-depth=10-entropy': dt2,
        'DT-full': dt3,
        'RF-depth=8': rf,
    }

def evaluate_classifier(clf, X_train, X_val, y_train, y_val, description):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f"\nClassifier: {description}")
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    return {
        'Classifier': description,
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_val, y_pred, average='weighted', zero_division=0)
    }

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

classifiers = get_classifiers()
results_grouped = {variant: [] for variant in data_variants}

for variant, (Xtr, Xva) in data_variants.items():
    for desc, clf in classifiers.items():
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xva)
        metrics = evaluate_classifier(clf, Xtr, Xva, y_train, y_val, f"{desc}")
        results_grouped[variant].append(metrics)
        plot_confusion_matrix(y_val, y_pred, f"Confusion Matrix - {desc} ({variant})")

flat_results = []
for preprocessing, entries in results_grouped.items():
    for entry in entries:
        result = {
            "Classifier": entry.get("Classifier"),
            "Preprocessing": preprocessing,
            "Accuracy": round(entry.get("Accuracy", 0), 3),
            "Precision": round(entry.get("Precision", 0), 3),
            "Recall": round(entry.get("Recall", 0), 3),
            "F1": round(entry.get("F1", 0), 3)
        }
        flat_results.append(result)

df_results = pd.DataFrame(flat_results)
df_results = df_results.sort_values(by=["Preprocessing", "Classifier"]).reset_index(drop=True)
print(df_results)

def plot_metrics(results_grouped):
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    for variant, entries in results_grouped.items():
        df = pd.DataFrame(entries)
        plt.figure(figsize=(16, 4))
        for i, metric in enumerate(metrics):
            plt.subplot(1, 4, i + 1)
            sns.barplot(x="Classifier", y=metric, data=df)
            plt.title(f"{metric} ({variant})")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

plot_metrics(results_grouped)
