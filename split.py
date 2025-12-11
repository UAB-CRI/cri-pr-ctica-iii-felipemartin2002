from sklearn.model_selection import train_test_split

def dividir_train_test_balanceado(df,
                                  nombre_columna_label="sentimentLabel",
                                  test_size=0.2,
                                  random_state=42):
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[nombre_columna_label]
    )
    return df_train, df_test