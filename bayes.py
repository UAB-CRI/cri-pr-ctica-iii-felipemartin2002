from collections import Counter
import numpy as np

def entrenar_naive_bayes(df_train,
                         vocab,
                         nombre_col_tokens="tokens",
                         nombre_col_label="sentimentLabel",
                         alpha=1.0):
    clases = sorted(df_train[nombre_col_label].unique())
    n_clases = len(clases)
    V = len(vocab)

    counts_clase = df_train[nombre_col_label].value_counts()
    total_docs = len(df_train)
    class_log_priors = np.zeros(n_clases, dtype=np.float64)

    for idx, c in enumerate(clases):
        class_log_priors[idx] = np.log(counts_clase[c] / total_docs)

    word_counts_por_clase = [Counter() for _ in range(n_clases)]
    total_words_por_clase = np.zeros(n_clases, dtype=np.int64)

    for _, fila in df_train.iterrows():
        c = fila[nombre_col_label]
        tokens = fila[nombre_col_tokens]
        c_idx = clases.index(c)

        for tok in tokens:
            if tok in vocab:
                word_counts_por_clase[c_idx][tok] += 1
                total_words_por_clase[c_idx] += 1

    log_likelihoods = np.zeros((n_clases, V), dtype=np.float64)

    for c_idx in range(n_clases):
        total_c = total_words_por_clase[c_idx]
        for palabra, w_idx in vocab.items():
            count_wc = word_counts_por_clase[c_idx][palabra]
            
            prob = (count_wc + alpha) / (total_c + alpha * V)
            log_likelihoods[c_idx, w_idx] = np.log(prob)

    return class_log_priors, log_likelihoods, np.array(clases)

def predecir_tokens(tokens, vocab,
                    class_log_priors,
                    log_likelihoods,
                    clases):
    """
    Predice la clase de un tweet dado sus tokens.
    """
    n_clases = len(clases)
    log_probs = class_log_priors.copy()

    for c_idx in range(n_clases):
        for tok in tokens:
            if tok in vocab:
                w_idx = vocab[tok]
                log_probs[c_idx] += log_likelihoods[c_idx, w_idx]

    idx_max = np.argmax(log_probs)
    return clases[idx_max]

def predecir_df(df,
                vocab,
                class_log_priors,
                log_likelihoods,
                clases,
                nombre_col_tokens="tokens",
                nombre_col_label="sentimentLabel"):
    
    y_true = df[nombre_col_label].to_numpy()
    y_pred = []

    for _, fila in df.iterrows():
        tokens = fila[nombre_col_tokens]
        c_pred = predecir_tokens(tokens, vocab,
                                 class_log_priors,
                                 log_likelihoods,
                                 clases)
        y_pred.append(c_pred)

    y_pred = np.array(y_pred)
    acc = (y_pred == y_true).mean()
    return y_pred, y_true, acc
