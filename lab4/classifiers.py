from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_classifiers():
    nb1 = GaussianNB(var_smoothing=1e-9)
    nb2 = GaussianNB(var_smoothing=1e-4)
    nb3 = GaussianNB(var_smoothing=1e-1)

    mb = MultinomialNB(alpha=1.0, fit_prior=True)

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
        'MB-alpha=1.0': mb,
        'DT-full': dt3,
        'RF-depth=8': rf,
    }
