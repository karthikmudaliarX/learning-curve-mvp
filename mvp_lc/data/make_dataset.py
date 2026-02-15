# File: mvp_lc/data/make_dataset.py
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def make_demo_dataset(n_samples: int = 800, n_features: int = 20, random_state: int = 42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        random_state=random_state
    )
    return X, y

if __name__ == "__main__":
    X, y = make_demo_dataset()
    print("Dataset shape:", X.shape)
