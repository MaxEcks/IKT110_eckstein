import numpy as np
import matplotlib.pyplot as plt
import model
from tqdm import trange


def train_model(xs, ys, learning_rate, batch_size, iterations):
    """
    Train logistic regression with mini-batch gradient descent.
    Balanced Mini-Batches (wenn beide Klassen vorhanden sind):
      - Jede Batch: ~50% Radiant (1), ~50% Dire (0).
    """
    n_features = xs.shape[1]
    theta = np.zeros(n_features + 1)
    m = xs.shape[0]
    j_history = []

    # Indizes nach Klassen trennen
    idx_pos = np.where(ys == 1)[0]
    idx_neg = np.where(ys == 0)[0]

    has_both_classes = (len(idx_pos) > 0) and (len(idx_neg) > 0)

    for it in trange(iterations, desc="Training Progress"):
        if has_both_classes:
            # Balanced Sampling: halbe/halbe
            half = batch_size // 2
            batch_pos = idx_pos[np.random.randint(0, len(idx_pos), size=half)]
            batch_neg = idx_neg[np.random.randint(0, len(idx_neg), size=batch_size - half)]
            batch = np.concatenate([batch_pos, batch_neg])
        else:
            # Fallback: altes Verhalten
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
    y_pred: 1D Array (Probabilities)
    threshold: threshold to convert y_pred in 0 or 1
    """
    pred_labels = (y_pred >= threshold).astype(int)

    tp = np.sum((pred_labels == 1) & (y_true == 1))
    fp = np.sum((pred_labels == 1) & (y_true == 0))
    fn = np.sum((pred_labels == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def f1_for_class(y_true, y_pred, cls, threshold=0.5):
    """
    Berechnet den F1 Score für eine einzelne Klasse (0 oder 1)
    """
    pred_labels = (y_pred >= threshold).astype(int)

    tp = np.sum((pred_labels == cls) & (y_true == cls))
    fp = np.sum((pred_labels == cls) & (y_true != cls))
    fn = np.sum((pred_labels != cls) & (y_true == cls))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

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


def find_best_threshold(y_true, y_scores, t_min=0.1, t_max=0.9, num=81):
    """
    Sucht den Threshold, der den höchsten Macro-F1 liefert.
    y_scores: Modell-Probabilities (0..1)
    """
    best_t = 0.5
    best_macro = -1.0

    thresholds = np.linspace(t_min, t_max, num)
    for t in thresholds:
        _, _, macro = f1_binary_macro(y_true, y_scores, threshold=t)
        if macro > best_macro:
            best_macro = macro
            best_t = t

    return best_t, best_macro


def plot_loss(j_history):
    plt.figure(figsize=(8, 5))
    plt.plot(j_history, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (J)")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


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

    # ------ Train model ----------------
    theta, j_history = train_model(
        spl_data["X_train"], spl_data["y_train"],
        learning_rate=0.01,
        batch_size=50000,
        iterations=1000
    )

    # ------ Probabilities auf Test-Set --------------
    y_scores = []
    for x in spl_data["X_test"]:
        y_scores.append(model.predict_single_value(theta, x))
    y_scores = np.array(y_scores)

    # ------ Metrics bei Threshold 0.5 ----------------
    acc_05 = accuracy(spl_data["y_test"], y_scores, threshold=0.5)
    f1_05  = f1_score(spl_data["y_test"], y_scores, threshold=0.5)
    f1_0_05, f1_1_05, macro_f1_05 = f1_binary_macro(
        spl_data["y_test"], y_scores, threshold=0.5
    )

    print("=== Metrics @ threshold = 0.5 ===")
    print("Accuracy:", acc_05)
    print("F1 Score:", f1_05)
    print("F1 Class 0:", f1_0_05)
    print("F1 Class 1:", f1_1_05)
    print("Macro F1:", macro_f1_05)

    # ------ Best Threshold für Macro-F1 --------------
    best_t, best_macro = find_best_threshold(spl_data["y_test"], y_scores)
    acc_best = accuracy(spl_data["y_test"], y_scores, threshold=best_t)
    f1_best  = f1_score(spl_data["y_test"], y_scores, threshold=best_t)
    f1_0_best, f1_1_best, _ = f1_binary_macro(
        spl_data["y_test"], y_scores, threshold=best_t
    )

    print("\n=== Best threshold search (Macro-F1) ===")
    print("Best threshold:", best_t)
    print("Accuracy @ best threshold:", acc_best)
    print("F1 Score @ best threshold:", f1_best)
    print("F1 Class 0 @ best threshold:", f1_0_best)
    print("F1 Class 1 @ best threshold:", f1_1_best)
    print("Macro F1 @ best threshold:", best_macro)

    # ------ Theta + best_threshold speichern ---------
    np.savez(
        "portfolio/dota_2/data/theta_model.npz",
        theta=theta,
        best_threshold=best_t
    )
    print("\nSaved theta and best_threshold to portfolio/dota_2/data/theta_model.npz")
    print("Theta shape:", theta.shape)

    # ------ Lossverlauf plotten ----------------------
    plot_loss(j_history)