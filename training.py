import numpy as np
from sklearn.model_selection import StratifiedKFold

from diccionary import construir_diccionario
from bayes import entrenar_naive_bayes, predecir_df


def kfold_cv_nb(df,
                k=5,
                label_col="sentimentLabel",
                tokens_col="tokens",
                min_freq=2,
                max_vocab=None,
                alpha=1.0,
                random_state=42):
    
    y = df[label_col].to_numpy()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        df_train = df.iloc[train_idx]
        df_val   = df.iloc[val_idx]

        vocab, _ = construir_diccionario(
            df_train,
            nombre_columna_tokens=tokens_col,
            min_freq=min_freq,
            max_vocab=max_vocab
        )

        priors, likelihoods, clases = entrenar_naive_bayes(
            df_train,
            vocab,
            nombre_col_tokens=tokens_col,
            nombre_col_label=label_col,
            alpha=alpha
        )

        _, _, acc = predecir_df(
            df_val,
            vocab,
            priors,
            likelihoods,
            clases,
            nombre_col_tokens=tokens_col,
            nombre_col_label=label_col
        )

        accs.append(acc)
        print(f"[Fold {fold}/{k}] vocab={len(vocab)} acc={acc:.4f}")

    accs = np.array(accs, dtype=np.float64)
    return accs, float(accs.mean()), float(accs.std())