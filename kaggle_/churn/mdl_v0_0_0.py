from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from matplotlib import pyplot as plt


def train_model(train_df, target_col="Churn_Binary", id_col="id", safra_col="safra"):

    X = train_df.drop(columns=[target_col, id_col, safra_col])
    y = train_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, min_weight_fraction_leaf=0.05, random_state=42)
    model.fit(X_train, y_train)

    plt.figure(figsize=(15, 7))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=["No Churn", "Churn"])
    plt.show()

    return model, X_train, X_test, y_train, y_test

