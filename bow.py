import numpy as np

def tokens_a_bow(tokens, vocab):
    v_size = len(vocab)
    vec = np.zeros(v_size, dtype=np.int32)

    for tok in tokens:
        if tok in vocab:
            idx = vocab[tok]
            vec[idx] += 1

    return vec

def df_a_matriz_bow(df,
                    vocab,
                    nombre_columna_tokens="tokens",
                    nombre_columna_label="sentimentLabel"):
    n = len(df)
    v_size = len(vocab)
    X = np.zeros((n, v_size), dtype=np.int32)
    y = df[nombre_columna_label].to_numpy()

    for i, tokens in enumerate(df[nombre_columna_tokens]):
        X[i, :] = tokens_a_bow(tokens, vocab)

    return X, y