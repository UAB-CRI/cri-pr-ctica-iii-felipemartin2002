from diccionary import *
from load import *

def main():
    ruta = "FinalStemmedSentimentAnalysisDataset.csv"

    df = cargar_dataset(ruta)

    stop_words = obtener_stopwords_ingles()

    df = añadir_columna_tokens(df,
                               nombre_columna_texto="tweetText",
                               nombre_columna_tokens="tokens",
                               stop_words=stop_words)

    vocab, contador = construir_diccionario(df,
                                            nombre_columna_tokens="tokens",
                                            min_freq=2,
                                            max_vocab=None)

    print("Tamaño vocabulario:", len(vocab))
    print(contador.most_common(20))

if __name__ == "__main__":
    main()
