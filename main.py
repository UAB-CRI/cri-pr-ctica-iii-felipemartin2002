from diccionary import construir_diccionario
from load import cargar_dataset, obtener_stopwords_ingles, añadir_columna_tokens
from split import dividir_train_test_balanceado
from bow import df_a_matriz_bow

def main():
    ruta = "FinalStemmedSentimentAnalysisDataset.csv"

    df = cargar_dataset(ruta)

    stop_words = obtener_stopwords_ingles()

    df = añadir_columna_tokens(df,
                               nombre_columna_texto="tweetText",
                               nombre_columna_tokens="tokens",
                               stop_words=stop_words)

    df_train, df_test = dividir_train_test_balanceado(
        df,
        nombre_columna_label="sentimentLabel",
        test_size=0.2,
        random_state=42
    )

    vocab, contador = construir_diccionario(
        df_train,
        nombre_columna_tokens="tokens",
        min_freq=2,
        max_vocab=None
    )

    print("Tamaño vocabulario:", len(vocab))

    X_train, y_train = df_a_matriz_bow(df_train, vocab,
                                       nombre_columna_tokens="tokens",
                                       nombre_columna_label="sentimentLabel")

    X_test, y_test = df_a_matriz_bow(df_test, vocab,
                                     nombre_columna_tokens="tokens",
                                     nombre_columna_label="sentimentLabel")

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

if __name__ == "__main__":
    main()

