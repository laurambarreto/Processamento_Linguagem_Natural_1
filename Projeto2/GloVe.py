## -- IMPORTS -- ##
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import optuna
from gensim.models import KeyedVectors

glove = KeyedVectors.load_word2vec_format('glove_s100.txt')

# Lista de stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Carregar os dados e passá-los para Dataframe
data = pd.read_csv ("factnews_dataset.csv", delimiter = ',')
df = pd.DataFrame (data)

print (df)

# Escolher apenas a coluna de sentences
X = df["sentences"]

# y: coluna de classes (target)
y = df.iloc[:, -1]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
# Divisão dos dados de treino em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42, stratify = y_train)

## -- FUNÇÃO QUE TRANSFORMA UMA FRASE NUM VETOR -- ##
def frase_para_vetor(frase, model, dim = 100):
    tokens = [w.lower() for w in word_tokenize(frase) if w.isalpha() and w not in stop_words]
    vetores = [model[w] for w in tokens if w in model]
    if len(vetores) == 0:
        return np.zeros(dim)
    return np.mean(vetores, axis = 0)

# Transforma todas as frases de todos os conjuntos de dados em vetores
X_train = np.array([frase_para_vetor(s, glove) for s in X_train])
X_val = np.array([frase_para_vetor(s, glove) for s in X_val])
X_test = np.array([frase_para_vetor(s, glove) for s in X_test])

X_trainval = np.vstack([X_train, X_val]) # Junta as features
y_trainval = np.concatenate([y_train, y_val]) # Junta as labels

## -- MODELO KNN - TESTE DE PARÂMETROS COM OPTUNA-- ##
def KNN_optuna (trial):
    n_neighbors = trial.suggest_int ("n_neighbors", 1, 50)
    weights = trial.suggest_categorical ("weights", ['uniform', 'distance'])
    metric = trial.suggest_categorical ("metric", ['euclidean', 'manhattan', 'minkowski'])
    KNN = KNeighborsClassifier (n_neighbors = n_neighbors, weights = weights, metric = metric) 
    KNN.fit (X_train, y_train)

    y_pred = KNN.predict (X_val)

    score = f1_score(y_val, y_pred, average = 'macro')

    return score

# Criação e execução do estudo
study = optuna.create_study(direction = 'maximize')
study.optimize(KNN_optuna, n_trials = 400)

print("Melhores parâmetros encontrados:")
print(study.best_params)
print()
print("Melhor F1 (macro):", study.best_value)
print()

## -- MODELO KNN COM MELHORES PARÂMETROS -- ##
def KNN_modelo ():
    KNN = KNeighborsClassifier (**study.best_params)
    # Adicionar medição de tempo de treino
    time_start = time.time()
    KNN.fit (X_trainval, y_trainval)
    time_end = time.time()
    print(f"Tempo de treino do modelo KNN: {time_end - time_start:.2f}")

    y_pred = KNN.predict (X_test)

    return y_pred

y_pred_KNN = KNN_modelo ()

print("-------- MODELO KNN --------")
print("Accuracy:", accuracy_score(y_test, y_pred_KNN))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred_KNN, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO KNN 
cm = confusion_matrix(y_test, y_pred_KNN)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão KNN", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()


## -- MODELO REGRESSÃO LOGÍSTICA - TESTE DE PARÂMETROS COM OPTUNA -- ##
def reg_log_optuna (trial):
    C = trial.suggest_float ("C", 0.01, 10.0, log = True)
    solver = trial.suggest_categorical ("solver", ['liblinear', 'saga', 'lbfgs'])
    max_iter = trial.suggest_int ("max_iter", 100, 1000)
    log_reg = LogisticRegression (C = C, solver = solver, max_iter = max_iter, random_state = 42)
    log_reg.fit (X_train, y_train)

    y_pred = log_reg.predict (X_val)

    score = f1_score (y_val, y_pred, average = 'macro')

    return score

# Criação e execução do estudo
study_reglog = optuna.create_study (direction = 'maximize')
study_reglog.optimize (reg_log_optuna, n_trials = 400)

print("Melhores parâmetros encontrados:")
print(study_reglog.best_params)
print()
print("Melhor F1 (macro):", study_reglog.best_value)
print()

## -- MODELO REGRESSÃO LOGÍSTICA COM MELHORES PARÂMETROS -- ##
def regressao_logistica ():
    log_reg = LogisticRegression (**study_reglog.best_params, random_state = 42)
    # Adicionar medição de tempo de treino
    time_start = time.time()
    log_reg.fit (X_trainval, y_trainval)
    time_end = time.time()
    print(f"Tempo de treino do modelo Regressão Logística: {time_end - time_start:.2f}")
    y_pred = log_reg.predict (X_test)
    return y_pred

y_pred_reglog = regressao_logistica ()

print("-------- MODELO REGRESSÃO LOGÍSTICA --------")
print("Accuracy:", accuracy_score(y_test, y_pred_reglog))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred_reglog, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO REGRESSÃO LOGÍSTICA
cm = confusion_matrix(y_test, y_pred_reglog)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão do Modelo Regressão Logística", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()

## -- MODELO NAIVE BAYES - TESTE DE PARÂMETROS COM OPTUNA -- ##
def naive_bayes_optuna (trial):
    var_smoothing = trial.suggest_float ("var_smoothing", 1e-11, 1e-5, log = True)
    nb = GaussianNB (var_smoothing = var_smoothing)
    nb.fit (X_train, y_train)

    y_pred = nb.predict (X_val)
    score = f1_score (y_val, y_pred, average = 'macro')
    return score

# Criação e execução do estudo
study_nb = optuna.create_study (direction = 'maximize')
study_nb.optimize (naive_bayes_optuna, n_trials = 400)

print("Melhores parâmetros encontrados:")
print(study_nb.best_params)
print()
print("Melhor F1 (macro):", study_nb.best_value)
print()

## -- NAIVE BAYES COM MELHORES PARÂMETROS -- ##
def naive_bayes_modelo ():
    nb = GaussianNB(**study_nb.best_params)
    time_start = time.time()
    nb.fit(X_trainval, y_trainval)
    time_end = time.time()
    print(f"Tempo de treino do modelo Naive Bayes: {time_end - time_start:.2f}")
    y_pred = nb.predict(X_test)
    return y_pred

y_pred_nb = naive_bayes_modelo()

print("-------- MODELO NAIVE BAYES --------")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred_nb, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO NAIVE BAYES
cm = confusion_matrix(y_test, y_pred_nb)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão do Modelo Naive Bayes", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()

## -- MODELO RANDOM FOREST - TESTE DE PARÂMETROS COM OPTUNA -- ##
def random_forest_optuna (trial):
    n_estimators = trial.suggest_int ("n_estimators", 100, 500)
    max_depth = trial.suggest_int ("max_depth", 5, 20)
    min_samples_split = trial.suggest_int ("min_samples_split", 2, 20)
    max_leaf_nodes = trial.suggest_int ("max_leaf_nodes", 10, 500)
    bootstrap = trial.suggest_categorical ("bootstrap", [True, False])
    criterion = trial.suggest_categorical ("criterion", ["gini", "entropy"])
    rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, 
                                max_leaf_nodes = max_leaf_nodes, bootstrap = bootstrap, criterion = criterion, random_state = 42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    score = f1_score (y_val, y_pred, average = 'macro')
    return score

# Criação e execução do estudo
study_rf = optuna.create_study (direction = 'maximize')
study_rf.optimize (random_forest_optuna, n_trials = 400)

print("Melhores parâmetros encontrados:")
print(study_rf.best_params)
print()
print("Melhor F1 (macro):", study_rf.best_value)
print()

## -- RANDOM FOREST COM MELHORES PARÂMETROS -- ##
def random_forest_modelo ():
    rf = RandomForestClassifier (**study_rf.best_params, random_state = 42)
    time_start = time.time()
    rf.fit(X_trainval, y_trainval)
    time_end = time.time()
    print(f"Tempo de treino do modelo Random Forest: {time_end - time_start:.2f}")

    y_pred = rf.predict (X_test)
    return y_pred

y_pred_rf = random_forest_modelo()

print("-------- MODELO RANDOM FOREST --------")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred_rf, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO RANDOM FOREST
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão do Modelo Random Forest", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()

## -- MODELO REDE NEURONAL - TESTE DE PARÂMETROS COM OPTUNA -- ##
def rede_neuronal_optuna (trial):
    hidden_layer_options = {
        "32": (32,),
        "64": (64,),
        "128": (128,),
        "64_32": (64, 32),
        "128_64": (128, 64),
        "128_64_32": (128, 64, 32)
    }
    
    hidden_choice = trial.suggest_categorical("hidden_layer_sizes", list(hidden_layer_options.keys()))
    hidden_layer_sizes = hidden_layer_options[hidden_choice]
    activation = trial.suggest_categorical ("activation", ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical ("solver", ['adam', 'lbfgs', 'sgd'])
    alpha = trial.suggest_float ("alpha", 1e-5, 1e-2, log = True)
    learning_rate_init = trial.suggest_float ("learning_rate_init", 1e-4, 1e-2, log = True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    early_stopping = trial.suggest_categorical("early_stopping", [True, False])
    n_iter_no_change = trial.suggest_int("n_iter_no_change", 10, 50)
    max_iter = trial.suggest_int("max_iter", 1200, 2000)
    rnn = MLPClassifier (hidden_layer_sizes = hidden_layer_sizes,
                        activation = activation,
                        solver = solver,
                        alpha = alpha,
                        learning_rate_init = learning_rate_init,
                        max_iter = max_iter,
                        batch_size = batch_size,
                        early_stopping = early_stopping,
                        n_iter_no_change = n_iter_no_change,
                        random_state = 42)
    
    rnn.fit (X_train, y_train)

    y_pred = rnn.predict (X_val)
    score = f1_score (y_val, y_pred, average = 'macro')
    return score

# Criação e execução do estudo
study_rnn = optuna.create_study (direction = 'maximize')
study_rnn.optimize (rede_neuronal_optuna, n_trials = 400)

print("Melhores parâmetros encontrados:")
print(study_rnn.best_params)
print()
print("Melhor F1 (macro):", study_rnn.best_value)
print()


## -- REDE NEURONAL COM MELHORES PARÂMETROS -- ##
def rede_neuronal_modelo ():
    best_params = study_rnn.best_params.copy()

    # Reconverter a string do Optuna
    hidden_layer_options = {
        "32": (32,),
        "64": (64,),
        "128": (128,),
        "64_32": (64, 32),
        "128_64": (128, 64),
        "128_64_32": (128, 64, 32)
    }

    best_params["hidden_layer_sizes"] = hidden_layer_options[best_params["hidden_layer_sizes"]]

    rnn = MLPClassifier(**best_params, random_state = 42)
    time_start = time.time()
    rnn.fit(X_trainval, y_trainval)
    time_end = time.time()
    print(f"Tempo de treino do modelo Rede Neuronal: {time_end - time_start:.2f}")

    y_pred = rnn.predict (X_test)
    return y_pred

y_pred_rnn = rede_neuronal_modelo ()

print("-------- MODELO REDE NEURONAL --------")
print("Accuracy:", accuracy_score(y_test, y_pred_rnn))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred_rnn, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO REDE NEURONAL
cm = confusion_matrix(y_test, y_pred_rnn)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão do Modelo Rede Neuronal", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()