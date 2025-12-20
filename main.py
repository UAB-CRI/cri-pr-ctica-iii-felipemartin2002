from diccionary import construir_diccionario
from load import cargar_dataset, obtener_stopwords_ingles, añadir_columna_tokens
from split import dividir_train_test_balanceado
from bayes import entrenar_naive_bayes, predecir_df
from training import kfold_cv_nb
from experiments_b import (
    experiment_strategy_1_train_grows_vocab_grows,
    experiment_strategy_2_fixed_train_vary_vocab,
    experiment_strategy_3_fixed_vocab_vary_train,
    experiment_test_size_effect_holdout
)

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

    df_b1 = experiment_strategy_1_train_grows_vocab_grows(
        df,
        train_fracs=(0.10, 0.20, 0.40, 0.60, 0.80, 1.0),
        min_freq=2,
        alpha=1.0,
        k=5,
        max_vocab=None,       
        random_state=42
    )
    df_b1.to_csv("B1_train_grows_vocab_grows.csv", index=False)

    df_b2 = experiment_strategy_2_fixed_train_vary_vocab(
        df,
        vocab_sizes=(5000, 10000, 20000, 50000, 100000),
        train_frac=1.0,
        min_freq=2,
        alpha=1.0,
        k=5,
        random_state=42
    )
    df_b2.to_csv("B2_fixed_train_vary_vocab.csv", index=False)

    df_b3 = experiment_strategy_3_fixed_vocab_vary_train(
        df,
        train_fracs=(0.10, 0.20, 0.40, 0.60, 0.80, 1.0),
        fixed_vocab_size=50000,
        min_freq=2,
        alpha=1.0,
        k=5,
        random_state=42
    )
    df_b3.to_csv("B3_fixed_vocab_vary_train.csv", index=False)

    df_test = experiment_test_size_effect_holdout(
        df,
        test_sizes=(0.05, 0.10, 0.20, 0.30, 0.40),
        max_vocab=50000,
        min_freq=2,
        alpha=1.0,
        random_state=42
    )
    df_test.to_csv("B_test_size_effect.csv", index=False)

if __name__ == "__main__":
    main()
