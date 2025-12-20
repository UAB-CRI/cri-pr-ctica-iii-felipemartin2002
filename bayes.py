from collections import Counter
import numpy as np


def entrenar_naive_bayes(df_train,
                         vocab,
                         nombre_col_tokens="tokens",
                         nombre_col_label="sentimentLabel",
                         alpha=1.0):
    """
    Optimizado:
    - Evita iterrows() -> usa listas/arrays
    - Evita clases.index() -> usa mapping
    - Guarda log-probs de palabras vistas + default por clase (smoothing), sin matriz densa
    Devuelve:
      class_log_priors: np.array [n_clases]
      model: (clases, log_prob_seen, default_logprob)  <-- se devuelve en la variable log_likelihoods
      clases: np.array
    """
    # Clases y mapping a índice
    clases = np.array(sorted(df_train[nombre_col_label].unique()))
    n_clases = len(clases)
    clase2idx = {c: i for i, c in enumerate(clases)}

    V = len(vocab)
    total_docs = len(df_train)

    # Priors
    y = df_train[nombre_col_label].to_numpy()
    class_log_priors = np.zeros(n_clases, dtype=np.float64)
    for i, c in enumerate(clases):
        class_log_priors[i] = np.log((y == c).sum() / total_docs)

    # Tokens como lista (mucho más rápido que iterrows)
    tokens_list = df_train[nombre_col_tokens].to_list()

    # Contar palabras por clase (solo palabras dentro del vocab)
    word_counts_por_clase = [Counter() for _ in range(n_clases)]
    total_words_por_clase = np.zeros(n_clases, dtype=np.int64)

    for toks, c in zip(tokens_list, y):
        c_idx = clase2idx[c]
        wc = word_counts_por_clase[c_idx]
        tw = total_words_por_clase[c_idx]

        # contar solo tokens en vocab
        for tok in toks:
            if tok in vocab:
                wc[tok] += 1
                tw += 1

        total_words_por_clase[c_idx] = tw

    # Precomputar log-probs:
    # default_logprob[c] = log(alpha / (total_c + alpha*V))
    # log_prob_seen[c][tok] = log((count+alpha)/denom) solo para tok vistos en esa clase
    default_logprob = np.zeros(n_clases, dtype=np.float64)
    log_prob_seen = [dict() for _ in range(n_clases)]

    for c_idx in range(n_clases):
        denom = total_words_por_clase[c_idx] + alpha * V
        default_logprob[c_idx] = np.log(alpha / denom)

        for tok, cnt in word_counts_por_clase[c_idx].items():
            # tok seguro está en vocab por el if anterior
            log_prob_seen[c_idx][tok] = np.log((cnt + alpha) / denom)

    # Devolvemos el "modelo" en la variable que antes era log_likelihoods
    model = (clases, log_prob_seen, default_logprob, vocab)
    return class_log_priors, model, clases


def predecir_tokens(tokens, vocab,
                    class_log_priors,
                    log_likelihoods,
                    clases):
    """
    Optimizado:
    log_likelihoods ya no es matriz; es model = (clases, log_prob_seen, default_logprob, vocab_ref)
    """
    _, log_prob_seen, default_logprob, vocab_ref = log_likelihoods

    # Usamos el vocab del modelo (por seguridad), pero aceptamos el parámetro vocab
    vocab_use = vocab_ref if vocab_ref is not None else vocab

    log_probs = class_log_priors.copy()
    n_clases = len(clases)

    for c_idx in range(n_clases):
        lp_seen = log_prob_seen[c_idx]
        dlp = default_logprob[c_idx]

        # suma por tokens presentes en vocab
        for tok in tokens:
            if tok in vocab_use:
                log_probs[c_idx] += lp_seen.get(tok, dlp)

    return clases[int(np.argmax(log_probs))]


def predecir_df(df,
                vocab,
                class_log_priors,
                log_likelihoods,
                clases,
                nombre_col_tokens="tokens",
                nombre_col_label="sentimentLabel"):
    """
    Optimizado:
    - Evita iterrows() -> usa listas
    """
    y_true = df[nombre_col_label].to_numpy()
    tokens_list = df[nombre_col_tokens].to_list()

    y_pred = np.empty(len(y_true), dtype=y_true.dtype)

    for i, toks in enumerate(tokens_list):
        y_pred[i] = predecir_tokens(
            toks, vocab,
            class_log_priors,
            log_likelihoods,
            clases
        )

    acc = (y_pred == y_true).mean()
    return y_pred, y_true, float(acc)
