import numpy as np
import matplotlib.pyplot as plt
import model
from tqdm import trange



def train_model(xs, ys, learning_rate, batch_size, iterations):
    n_features = xs.shape[1]
    theta = np.zeros(n_features + 1)
    m = xs.shape[0]
    j_history = []

    for it in trange(iterations, desc="Training Progress"):
        batch = np.random.randint(0, m, size=batch_size)

        j = model.J_logistic(theta, xs, ys, batch) / batch_size
        j_history.append(j)

        grad = model.gradient_J_logistic(theta, xs, ys, batch)
        theta = theta - learning_rate * (grad / batch_size)

    return theta, j_history


def accuracy(y_true, y_pred, threshold=0.5):
    """
    Calculates accuracy for binary classification.
    y_true: 1D Array (0 or 1)
    y_pred: 1D Array (Probabilities)
    threshold: threshold to convert y_pred to 0 or 1
    """
    n = y_true.shape[0]
    correct = 0
    for i in range(n):
        pred_label = 1 if y_pred[i] >= threshold else 0
        if pred_label == y_true[i]:
            correct += 1
    return correct / n

def f1_score(y_true, y_pred, threshold=0.5):
    """
    Calculates F1 score of a binary classification.
    y_true: 1D Array (0 or 1)
    y_pred: 1D Array (Probabilities))
    threshold: threshold to convert y_pred in 0 or 1
    """
    # Predicted labels
    pred_labels = (y_pred >= threshold).astype(int)

    # True/False Positives/Negatives
    tp = np.sum((pred_labels == 1) & (y_true == 1))
    fp = np.sum((pred_labels == 1) & (y_true == 0))
    fn = np.sum((pred_labels == 0) & (y_true == 1))

    # Precision & Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def f1_for_class(y_true, y_pred, cls, threshold=0.5):
    """
    Berechnet den F1 Score für eine einzelne Klasse (0 oder 1)
    """
    # Labels vorher binarisiert
    pred_labels = (y_pred >= threshold).astype(int)

    tp = np.sum((pred_labels == cls) & (y_true == cls))
    fp = np.sum((pred_labels == cls) & (y_true != cls))
    fn = np.sum((pred_labels != cls) & (y_true == cls))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def f1_binary_macro(y_true, y_pred, threshold=0.5):
    """
    Berechnet F1 Score für beide Klassen und den Macro-F1 Score
    Returns:
      f1_class0, f1_class1, macro_f1
    """
    f1_0 = f1_for_class(y_true, y_pred, cls=0, threshold=threshold)
    f1_1 = f1_for_class(y_true, y_pred, cls=1, threshold=threshold)
    macro = 0.5 * (f1_0 + f1_1)
    return f1_0, f1_1, macro



def plot_loss(j_history):
    plt.figure(figsize=(8,5))
    plt.plot(j_history, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (J)")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----- Manual split (NumPy) -----
def split_xy_manual(X, y, test_size=0.20, val_size=0.20, seed=42, shuffle=True):
    """
    Split X, y into train/val/test using NumPy only.
    - test_size: fraction of all samples reserved for test
    - val_size:  fraction of all samples reserved for validation
    Final proportions: train = 1 - test_size - val_size
    """
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_test = int(round(n * test_size))
    n_val  = int(round(n * val_size))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("Not enough samples for the requested split sizes.")

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val":   X[val_idx],   "y_val":   y[val_idx],
        "X_test":  X[test_idx],  "y_test":  y[test_idx],
        "idx": {"train": train_idx, "val": val_idx, "test": test_idx}
    }





if __name__ == "__main__":
    # ----- Load Data --------
    data = np.load("portfolio/dota_2/data/training_dataset.npz")
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.float64)

    # ----- Train/test split ------------
    spl_data = split_xy_manual(X, y, test_size=0.2, val_size=0, seed=42, shuffle=True)

    #------ Train model
    theta, j_history = train_model(spl_data["X_train"], spl_data["y_train"], 0.01, 50000, 1000)
    np.savez("portfolio/dota_2/data/theta_model.npz", theta=theta)
    print(theta)
    
    # ------ Calc accuracy --------------
    y_pred = []
    for x in spl_data["X_test"]:
        y_pred.append(model.predict_single_value(theta, x))
    y_pred = np.array(y_pred)
    acc = accuracy(spl_data["y_test"], y_pred, 0.5)
    f1  = f1_score(spl_data["y_test"], y_pred, 0.5)
    f1_0, f1_1, macro_f1 = f1_binary_macro(spl_data["y_test"], y_pred, threshold=0.5)

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("F1 Class 0:", f1_0)
    print("F1 Class 1:", f1_1)
    print("Macro F1:", macro_f1)

    plot_loss(j_history)