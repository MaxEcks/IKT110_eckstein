import numpy as np

data = np.load("portfolio/dota_2/data/training_dataset.npz")
# data = np.load("training_dataset.npz", mmap_mode="r")

print(data.files)   # prints the keys in the .npz file

# --------------------------------------------------------------
# inspect shapes of X and y
# --------------------------------------------------------------

X = data["X"]
y = data["y"]

print("Shape X:", X.shape)
print("Shape y:", y.shape)

print("\n=== First 10 rows of X ===")
print(X[:10])   # each row = one match

print("\n=== First 10 y-values ===")
print(y[:10])

# --------------------------------------------------------------
# inspect one specific match
# --------------------------------------------------------------

i = 0  # index of the match to inspect
row = X[i]
picked_radiant = np.where(row == 1)[0] + 1
picked_dire    = np.where(row == -1)[0] + 1

print("Match", i)
print("Radiant picks:", picked_radiant)
print("Dire picks:", picked_dire)
print("Radiant win:", y[i])

# --------------------------------------------------------------
# print one specific line from X and y
# --------------------------------------------------------------
i = 1234
print("X[i]:")
print(X[i])
print("y[i]:", y[i])