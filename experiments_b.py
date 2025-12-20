import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from diccionary import construir_diccionario
from bayes import entrenar_naive_bayes, predecir_df
from training import kfold_cv_nb

def sample_stratified_df(df, frac, label_col="sentimentLabel", random_state=42):
    if frac >= 1.0:
        return df.copy()

    y = df[label_col]
    df_sub, _ = train_test_split(
        df,
        train_size=frac,
        random_state=random_state,
        stratify=y
    )
    return df_sub


def fixed_train_test_split(df, test_size=0.2, label_col="sentimentLabel", random_state=42):
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col]
    )
    return df_train, df_test


# -----------------------------
# Ampliar train y dejar que crezca el diccionario
# -----------------------------
def experiment_strategy_1_train_grows_vocab_grows(
    df,
    train_fracs=(0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0),
    min_freq=2,
    alpha=1.0,
    k=5,
    max_vocab=None,
    label_col="sentimentLabel",
    tokens_col="tokens",
    random_state=42
):
    rows = []
    for frac in train_fracs:
        df_sub = sample_stratified_df(df, frac, label_col=label_col, random_state=random_state)

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

        vocab, _ = construir_diccionario(
            df_sub,
            nombre_columna_tokens=tokens_col,
            min_freq=min_freq,
            max_vocab=max_vocab
        )

        rows.append({
            "strategy": "S1_train_grows_vocab_grows",
            "train_frac": frac,
            "n_samples": len(df_sub),
            "vocab_size": len(vocab),
            "k": k,
            "min_freq": min_freq,
            "max_vocab": max_vocab if max_vocab is not None else "None",
            "alpha": alpha,
            "acc_mean": mean_acc,
            "acc_std": std_acc
        })

        print(f"[S1] frac={frac} n={len(df_sub)} vocab={len(vocab)} mean={mean_acc:.4f} std={std_acc:.4f}")

    return pd.DataFrame(rows)


# -----------------------------
# Train fijo variar tamaño del diccionario
# -----------------------------
def experiment_strategy_2_fixed_train_vary_vocab(
    df,
    vocab_sizes=(5000, 10000, 20000, 50000, 100000),
    train_frac=1.0,
    min_freq=2,
    alpha=1.0,
    k=5,
    label_col="sentimentLabel",
    tokens_col="tokens",
    random_state=42
):
    df_sub = sample_stratified_df(df, train_frac, label_col=label_col, random_state=random_state)

    rows = []
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
            "strategy": "S2_fixed_train_vary_vocab",
            "train_frac": train_frac,
            "n_samples": len(df_sub),
            "vocab_size_target": max_vocab,
            "k": k,
            "min_freq": min_freq,
            "alpha": alpha,
            "acc_mean": mean_acc,
            "acc_std": std_acc
        })

        print(f"[S2] max_vocab={max_vocab} n={len(df_sub)} mean={mean_acc:.4f} std={std_acc:.4f}")

    return pd.DataFrame(rows)


# -----------------------------
# Vocab fijo variar train
# -----------------------------
def experiment_strategy_3_fixed_vocab_vary_train(
    df,
    train_fracs=(0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0),
    fixed_vocab_size=50000,
    min_freq=2,
    alpha=1.0,
    k=5,
    label_col="sentimentLabel",
    tokens_col="tokens",
    random_state=42
):
    rows = []
    for frac in train_fracs:
        df_sub = sample_stratified_df(df, frac, label_col=label_col, random_state=random_state)

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
            "strategy": "S3_fixed_vocab_vary_train",
            "train_frac": frac,
            "n_samples": len(df_sub),
            "fixed_vocab_size": fixed_vocab_size,
            "k": k,
            "min_freq": min_freq,
            "alpha": alpha,
            "acc_mean": mean_acc,
            "acc_std": std_acc
        })

        print(f"[S3] frac={frac} n={len(df_sub)} vocab={fixed_vocab_size} mean={mean_acc:.4f} std={std_acc:.4f}")

    return pd.DataFrame(rows)


# -----------------------------
# Efecto del tamaño del TEST
# -----------------------------
def experiment_test_size_effect_holdout(
    df,
    test_sizes=(0.05, 0.10, 0.20, 0.30, 0.40),
    max_vocab=50000,
    min_freq=2,
    alpha=1.0,
    label_col="sentimentLabel",
    tokens_col="tokens",
    random_state=42
):
    rows = []

    for ts in test_sizes:
        df_train, df_test = fixed_train_test_split(df, test_size=ts, label_col=label_col, random_state=random_state)

        vocab, _ = construir_diccionario(
            df_train,
            nombre_columna_tokens=tokens_col,
            min_freq=min_freq,
            max_vocab=max_vocab
        )

        priors, model, clases = entrenar_naive_bayes(
            df_train,
            vocab,
            nombre_col_tokens=tokens_col,
            nombre_col_label=label_col,
            alpha=alpha
        )

        _, _, acc = predecir_df(
            df_test,
            vocab,
            priors,
            model,
            clases,
            nombre_col_tokens=tokens_col,
            nombre_col_label=label_col
        )

        rows.append({
            "strategy": "TEST_SIZE_HOLDOUT",
            "test_size": ts,
            "train_size": 1.0 - ts,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "vocab_size": len(vocab),
            "min_freq": min_freq,
            "max_vocab": max_vocab,
            "alpha": alpha,
            "acc": float(acc)
        })

        print(f"[TEST] test_size={ts} n_test={len(df_test)} acc={acc:.4f} vocab={len(vocab)}")

    return pd.DataFrame(rows)