from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

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
