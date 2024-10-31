# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os


# Função para carregar e preparar os dados
def carregar_dados():
    # Definir caminhos dos arquivos de dados
    train_files = [
        'lbp-train-fold_0.csv', 'lbp-train-fold_1.csv', 'lbp-train-fold_2.csv',
        'lbp-train-fold_3.csv', 'lbp-train-fold_4.csv'
    ]
    test_file = 'lbp-test.csv'

    # Carregar e concatenar os arquivos de treino
    train_df = pd.concat([pd.read_csv(file) for file in train_files])
    test_df = pd.read_csv(test_file)

    # Ajustar a coluna 'class' para obter apenas o nome da doença
    train_df['class'] = train_df['class'].str.split('/').str[-1]
    test_df['class'] = test_df['class'].str.split('/').str[-1]

    # Separar as features e o alvo
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']

    # Aplicar SMOTE para balancear o conjunto de treino
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled, X_test, y_test


# Função para treinar e avaliar o modelo com o relatório de classificação
def treinar_avaliar_modelo(X_train, y_train, X_test, y_test):
    # Inicializar o modelo Random Forest
    modelo = RandomForestClassifier(random_state=42)

    # Treinar o modelo
    modelo.fit(X_train, y_train)

    # Fazer previsões
    y_pred = modelo.predict(X_test)

    # Exibir o relatório de classificação com zero_division ajustado
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))


# Função para verificar o balanceamento das classes após o SMOTE
def verificar_balanceamento(y):
    balanceamento = y.value_counts(normalize=True) * 100
    print("Distribuição das Classes Após SMOTE:")
    print(balanceamento)


# Execução do fluxo completo
if __name__ == "__main__":
    # Carregar e balancear os dados
    X_train, y_train, X_test, y_test = carregar_dados()

    # Verificar o balanceamento das classes após SMOTE
    verificar_balanceamento(pd.Series(y_train))

    # Treinar e avaliar o modelo
    treinar_avaliar_modelo(X_train, y_train, X_test, y_test)
