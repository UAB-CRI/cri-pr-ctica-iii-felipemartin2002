from diccionary import construir_diccionario
from load import cargar_dataset, obtener_stopwords_ingles, añadir_columna_tokens
from split import dividir_train_test_balanceado
from bayes import entrenar_naive_bayes, predecir_df
from training import kfold_cv_nb

def main():
    ruta = "FinalStemmedSentimentAnalysisDataset.csv"

    df = cargar_dataset(ruta)
    stop_words = obtener_stopwords_ingles()
    df = añadir_columna_tokens(df,
                               nombre_columna_texto="tweetText",
                               nombre_columna_tokens="tokens",
                               stop_words=stop_words)

    accs, mean_acc, std_acc = kfold_cv_nb(
        df,
        k=5,
        label_col="sentimentLabel",
        tokens_col="tokens",
        min_freq=5,
        max_vocab=50000,   
        alpha=1.0,
        random_state=42
    )

    print("Accuracies:", accs)
    print(f"Mean acc: {mean_acc:.4f} | Std: {std_acc:.4f}")

if __name__ == "__main__":
    main()
