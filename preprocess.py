import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_monk_data(file_path):
    # Legge i dati separati da spazi
    data = pd.read_csv(file_path, sep='\s+', header=None)

    # Separa le feature (prime 6 colonne) e il target (ultima colonna)
    X_raw = data.iloc[:, 1:-1].values  # Salta 'Id' e 'class'
    y = data.iloc[:, 0].values  # 'class'

    # Codifica le feature categoriche
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X_raw)

    return X_encoded, y


def load_all_monks(base_path):
    datasets = {}
    for monk_name in ["monks-1", "monks-2", "monks-3"]:
        train_path = f"{base_path}/{monk_name}.train"
        test_path = f"{base_path}/{monk_name}.test"

        X_train, y_train = load_monk_data(train_path)
        X_test, y_test = load_monk_data(test_path)

        datasets[monk_name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }
    return datasets

