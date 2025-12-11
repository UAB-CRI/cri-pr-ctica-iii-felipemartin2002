from nltk.corpus import stopwords
import pandas as pd
import re

def cargar_dataset(ruta, sep=';', encoding='utf-8'):
    df = pd.read_csv(ruta, sep=sep, encoding=encoding)
    return df

def obtener_stopwords_ingles():
    return set(stopwords.words('english'))

def limpiar_y_tokenizar_texto(texto, stop_words):
    if not isinstance(texto, str):
        return []

    tokens = texto.split()

    tokens = [re.sub(r'[^a-z]', '', tok) for tok in tokens]

    tokens = [tok for tok in tokens if tok and tok not in stop_words]

    return tokens

def añadir_columna_tokens(df, nombre_columna_texto="tweetText",
                          nombre_columna_tokens="tokens",
                          stop_words=None):
    if stop_words is None:
        stop_words = obtener_stopwords_ingles()

    df[nombre_columna_tokens] = df[nombre_columna_texto].apply(
        lambda txt: limpiar_y_tokenizar_texto(txt, stop_words)
    )
    return df
