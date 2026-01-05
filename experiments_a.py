import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diccionary import construir_diccionario
from training import kfold_cv_nb


def sample_stratified_df(df, frac, label_col="sentimentLabel", random_state=42):
    if frac >= 1.0:
        return df.copy()
    df_sub, _ = train_test_split(
        df,
        train_size=frac,
        random_state=random_state,
        stratify=df[label_col]
    )
    return df_sub


# -----------------------------
# variando alpha
# Train crece y vocab crece
# -----------------------------
def experiment_A1_train_grows_vocab_grows_alphas(
    df,
    alphas=(0.0, 0.1, 0.5, 1.0, 2.0),
    train_fracs=(0.10, 0.20, 0.40, 0.60, 0.80, 1.0),
    k=5,
    min_freq=2,
    label_col="sentimentLabel",
    tokens_col="tokens",
    random_state=42
):
    rows = []
    for alpha in alphas:
        for frac in train_fracs:
            df_sub = sample_stratified_df(df, frac, label_col, random_state)

            accs, mean_acc, std_acc = kfold_cv_nb(
                df_sub,
                k=k,
                label_col=label_col,
                tokens_col=tokens_col,
                min_freq=min_freq,
                max_vocab=None,      
                alpha=alpha,
                random_state=random_state
            )

            vocab, _ = construir_diccionario(
                df_sub,
                nombre_columna_tokens=tokens_col,
                min_freq=min_freq,
                max_vocab=None
            )

            rows.append({
                "strategy": "A1_train_grows_vocab_grows",
                "alpha": alpha,
                "train_frac": frac,
                "n_samples": len(df_sub),
                "vocab_size": len(vocab),
                "k": k,
                "min_freq": min_freq,
                "acc_mean": mean_acc,
                "acc_std": std_acc
            })

            print(f"[A1] α={alpha} frac={frac} n={len(df_sub)} vocab={len(vocab)} mean={mean_acc:.4f}")

    return pd.DataFrame(rows)


# -----------------------------
# variando alpha
# Train fijo vocab variable
# -----------------------------
def experiment_A2_fixed_train_vary_vocab_alphas(
    df,
    alphas=(0.0, 0.1, 0.5, 1.0, 2.0),
    vocab_sizes=(5000, 10000, 20000, 50000, 100000),
    train_frac=1.0,
    k=5,
    min_freq=2,
    label_col="sentimentLabel",
    tokens_col="tokens",
    random_state=42
):
    df_sub = sample_stratified_df(df, train_frac, label_col, random_state)

    rows = []
    for alpha in alphas:
        for max_vocab in vocab_sizes:
            accs, mean_acc, std_acc = kfold_cv_nb(
                df_sub,
                k=k,
                label_col=label_col,
                tokens_col=tokens_col,
                min_freq=min_freq,
                max_vocab=max_vocab,
                alpha=alpha,
                random_state=random_state
            )

            rows.append({
                "strategy": "A2_fixed_train_vary_vocab",
                "alpha": alpha,
                "train_frac": train_frac,
                "n_samples": len(df_sub),
                "max_vocab": max_vocab,
                "k": k,
                "min_freq": min_freq,
                "acc_mean": mean_acc,
                "acc_std": std_acc
            })

            print(f"[A2] α={alpha} max_vocab={max_vocab} mean={mean_acc:.4f}")

    return pd.DataFrame(rows)


# -----------------------------
# variando alpha
# Vocab fijo train variable
# -----------------------------
def experiment_A3_fixed_vocab_vary_train_alphas(
    df,
    alphas=(0.0, 0.1, 0.5, 1.0, 2.0),
    train_fracs=(0.10, 0.20, 0.40, 0.60, 0.80, 1.0),
    fixed_vocab_size=50000,
    k=5,
    min_freq=2,
    label_col="sentimentLabel",
    tokens_col="tokens",
    random_state=42
):
    rows = []
    for alpha in alphas:
        for frac in train_fracs:
            df_sub = sample_stratified_df(df, frac, label_col, random_state)

            accs, mean_acc, std_acc = kfold_cv_nb(
                df_sub,
                k=k,
                label_col=label_col,
                tokens_col=tokens_col,
                min_freq=min_freq,
                max_vocab=fixed_vocab_size,
                alpha=alpha,
                random_state=random_state
            )

            rows.append({
                "strategy": "A3_fixed_vocab_vary_train",
                "alpha": alpha,
                "train_frac": frac,
                "n_samples": len(df_sub),
                "fixed_vocab_size": fixed_vocab_size,
                "k": k,
                "min_freq": min_freq,
                "acc_mean": mean_acc,
                "acc_std": std_acc
            })

            print(f"[A3] α={alpha} frac={frac} vocab={fixed_vocab_size} mean={mean_acc:.4f}")

    return pd.DataFrame(rows)
