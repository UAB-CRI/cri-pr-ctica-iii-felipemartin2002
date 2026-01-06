import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("B1_train_grows_vocab_grows.csv")

plt.figure()
plt.plot(df["train_frac"], df["acc_mean"], marker='o')
plt.xlabel("Fracció del conjunt de train")
plt.ylabel("Accuracy mitjana")
plt.title("Efecte de la mida del train (vocabulari creixent)")
plt.show()

# df = pd.read_csv("B2_fixed_train_vary_vocab.csv")

# plt.figure()
# plt.plot(df["vocab_size_target"], df["acc_mean"], marker='o')
# plt.xlabel("Mida del diccionari")
# plt.ylabel("Accuracy mitjana")
# plt.title("Efecte de la mida del diccionari")
# plt.xscale("log")
# plt.show()

# df = pd.read_csv("B3_fixed_vocab_vary_train.csv")

# plt.figure()
# plt.plot(df["train_frac"], df["acc_mean"], marker='o')
# plt.xlabel("Fracció del conjunt de train")
# plt.ylabel("Accuracy mitjana")
# plt.title("Efecte de la mida del train (diccionari fix)")
# plt.show()


# df = pd.read_csv("B_test_size_effect.csv")

# plt.figure()
# plt.plot(df["test_size"], df["acc"], marker='o')
# plt.xlabel("Fracció del conjunt de test")
# plt.ylabel("Accuracy")
# plt.title("Efecte de la mida del conjunt de test")
# plt.show()

# df = pd.read_csv("A2_fixed_train_vary_vocab_alphas.csv")
# df = df[df["max_vocab"] == 50000]

# plt.figure()
# plt.plot(df["alpha"], df["acc_mean"], marker='o')
# plt.xlabel("α (Laplace smoothing)")
# plt.ylabel("Accuracy mitjana")
# plt.title("Efecte del Laplace smoothing")
# plt.show()