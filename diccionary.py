from collections import Counter

def construir_diccionario(df,
                          nombre_columna_tokens="tokens",
                          min_freq=1,
                          max_vocab=None):
    contador = Counter()

    for tokens in df[nombre_columna_tokens]:
        contador.update(tokens)

    palabras = [w for w, f in contador.items() if f >= min_freq]

    palabras_sorted = sorted(palabras, key=lambda w: -contador[w])  # ordenadas por frecuencia desc.
    if max_vocab is not None:
        palabras_sorted = palabras_sorted[:max_vocab]

    vocab = {palabra: idx for idx, palabra in enumerate(palabras_sorted)}

    return vocab, contador