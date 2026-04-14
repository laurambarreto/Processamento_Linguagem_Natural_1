## -- IMPORTS -- ##
from sklearn.neighbors import KNeighborsClassifier
import spacy 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter
import re
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import optuna
from sklearn.metrics import f1_score
import time

# Lista de stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Carregar modelo spacy para português
nlp = spacy.load("pt_core_news_sm")

# Carregar os dados e passá-los para Dataframe
data = pd.read_csv ("factnews_dataset.csv", delimiter = ',')
df = pd.DataFrame (data)

print (df)

# Drop de colunas não numéricas
X = df.drop(columns = ["file", "classe", "domain", "id_article"])

# y: coluna de classes (target)
y = df.iloc[:, -1]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
# Divisão dos dados de treino em treino e validação
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42, stratify = y_train)

## Criação do dataframe de Treino ##
train_df = X_train.copy ()
train_df["classe"] = y_train.values

# Contagem de amostras por classe nos dados de treino
print(f"\nDados de treino:", train_df["classe"].value_counts())

## Criação do dataframe de Treino no conjunto de Validação ##
train_val_df = X_train_val.copy()
train_val_df["classe"] = y_train_val.values

# Contagem de amostras por classe nos dados de treino e validação
print(f"\nConjunto de dados de treino e validação:", train_val_df["classe"].value_counts())

## Criação do dataframe de Validação ##
val_df = X_val.copy()
val_df ["classe"] = y_val.values

# Contagem de amostras por classe nos dados de validação
print(f"\nDados de validação:", val_df["classe"].value_counts())

## Criação dos dataframes de Teste ##
test_df = X_test.copy()
test_df["classe"] = y_test.values

# Contagem de amostras por classe nos dados de teste
print(f"\nDados de teste:", test_df["classe"].value_counts())

print (train_df.shape)
print (train_val_df.shape)
print (val_df.shape)
print (test_df.shape)

# -- FUNÇÃO DE LIMPEZA E TOKENIZAÇÃO -- #
def limpar_tokenizar(texto):
    tokens = word_tokenize(str(texto).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) != 1]
    return tokens

# Adiciona coluna de tokens aos Dataframes
train_df["tokens"] = train_df["sentences"].apply(limpar_tokenizar)
train_val_df["tokens"] = train_val_df["sentences"].apply(limpar_tokenizar)
val_df["tokens"] = val_df["sentences"].apply(limpar_tokenizar)
test_df["tokens"] = test_df["sentences"].apply(limpar_tokenizar)

# Adicionar coluna de docs spaCy a todos os Dataframes
train_df["doc"] = [
    nlp(" ".join([token.text for token in doc if not token.is_stop]))
    for doc in nlp.pipe(train_df["sentences"], batch_size = 50)
]

train_val_df["doc"] = [
    nlp(" ".join([token.text for token in doc if not token.is_stop]))
    for doc in nlp.pipe(train_val_df["sentences"], batch_size = 50)
]

val_df["doc"] = [
    nlp(" ".join([token.text for token in doc if not token.is_stop]))
    for doc in nlp.pipe(val_df["sentences"], batch_size = 50)
]

test_df["doc"] = [
    nlp(" ".join([token.text for token in doc if not token.is_stop]))
    for doc in nlp.pipe(test_df["sentences"], batch_size = 50)
]

# -- CRIAÇÃO DO DICIONÁRIO DE POLARIDADES (CHAVES: PALAVRAS, VALOR: Polaridade da palavra) -- #
def ficheiro_sentilex ():
    sentimentos = {}
    with open ("SentiLex.csv", encoding = "utf-8") as f:
        for line in f:
            line = line.strip ()
            parts = line.split(",")
            palavra = parts [0]
            POL = parts [5].split('=')[1]
            sentimentos[palavra] = POL

    return sentimentos

sentimentos = ficheiro_sentilex ()

# -- CONTAGEM DE PALAVRAS NEGATIVAS, NEUTRAS E POSITIVAS NUMA FRASE -- #
def contagem_por_polaridade (sentimentos, linha):
    positividade = 0
    negatividade = 0
    neutralidade = 0
    palavras_negativas = Counter()
    palavras_positivas = Counter()
    palavras_neutras = Counter()
    for word in linha.split ():
        word = word.lower()
        if word in sentimentos:
            POL = sentimentos[word]
            if POL == "-1":
                negatividade += 1
                palavras_negativas[word] += 1
            elif POL == "0":
                neutralidade += 1
                palavras_neutras[word] += 1
            else:
                positividade += 1
                palavras_positivas[word] += 1
    
    return negatividade,neutralidade, positividade, palavras_negativas, palavras_positivas, palavras_neutras

## -- PALAVRAS POR POLARIDADE NUMA CLASSE -- ##
def palavras_polaridade (classe, df):
    todas_palavras_negativas = Counter()
    todas_palavras_positivas = Counter()
    todas_palavras_neutras = Counter()
    for i, sentence in enumerate(df["sentences"]):
        if df["classe"].iloc[i] == classe:
            _, _, _, palavras_negativas, palavras_positivas, palavras_neutras = contagem_por_polaridade(sentimentos, sentence)
            todas_palavras_negativas.update(palavras_negativas)
            todas_palavras_positivas.update(palavras_positivas)
            todas_palavras_neutras.update(palavras_neutras)

    top_negativas = todas_palavras_negativas.most_common()
    top_positivas = todas_palavras_positivas.most_common()
    top_neutras = todas_palavras_neutras.most_common()
    return top_negativas, top_positivas, top_neutras

# -- PALAVRAS EXCLUSIVAS DE CADA CLASSE (DATAFRAME DE TREINO) -- #
# Obtém as palavras mais frequentes por polaridade e classe
so_vies_neg_train, so_vies_pos_train, so_vies_neu_train = palavras_polaridade(1, train_df)
so_facto_neg_train, so_facto_pos_train, so_facto_neu_train = palavras_polaridade(0, train_df)
so_citacao_neg_train, so_citacao_pos_train, so_citacao_neu_train = palavras_polaridade(-1, train_df)

# Extrai só as palavras negativas
palavras_vies_neg_train = set([p for p, _ in so_vies_neg_train])
palavras_facto_neg_train = set([p for p, _ in so_facto_neg_train])
palavras_citacao_neg_train = set([p for p, _ in so_citacao_neg_train])

# Calcula as palavras negativas exclusivas de cada classe
so_vies_neg_train = palavras_vies_neg_train - palavras_facto_neg_train - palavras_citacao_neg_train
so_facto_neg_train = palavras_facto_neg_train - palavras_vies_neg_train - palavras_citacao_neg_train
so_citacao_neg_train = palavras_citacao_neg_train - palavras_vies_neg_train - palavras_facto_neg_train

# Extrai só as palavras positivas
palavras_vies_pos_train = set([p for p, _ in so_vies_pos_train])
palavras_facto_pos_train = set([p for p, _ in so_facto_pos_train])
palavras_citacao_pos_train = set([p for p, _ in so_citacao_pos_train])

# Calcula as palavras positivas exclusivas de cada classe
so_vies_pos_train = palavras_vies_pos_train - palavras_facto_pos_train - palavras_citacao_pos_train
so_facto_pos_train = palavras_facto_pos_train - palavras_vies_pos_train - palavras_citacao_pos_train
so_citacao_pos_train = palavras_citacao_pos_train - palavras_vies_pos_train - palavras_facto_pos_train

# Extrai só as palavras neutras
palavras_vies_neu_train = set([p for p, _ in so_vies_neu_train])
palavras_facto_neu_train = set([p for p, _ in so_facto_neu_train])
palavras_citacao_neu_train = set([p for p, _ in so_citacao_neu_train])

# Calcula as palavras neutras exclusivas de cada classe
so_vies_neu_train = palavras_vies_neu_train - palavras_facto_neu_train - palavras_citacao_neu_train
so_facto_neu_train = palavras_facto_neu_train - palavras_vies_neu_train - palavras_citacao_neu_train
so_citacao_neu_train = palavras_citacao_neu_train - palavras_vies_neu_train - palavras_facto_neu_train

# -- PALAVRAS EXCLUSIVAS DE CADA CLASSE (DATAFRAME DE TREINO NO CONJUNTO DE VALIDAÇÃO) -- #
# Obtém as palavras mais frequentes por polaridade e classe
so_vies_neg_train_val, so_vies_pos_train_val, so_vies_neu_train_val = palavras_polaridade(1, train_val_df)
so_facto_neg_train_val, so_facto_pos_train_val, so_facto_neu_train_val = palavras_polaridade(0, train_val_df)
so_citacao_neg_train_val, so_citacao_pos_train_val, so_citacao_neu_train_val = palavras_polaridade(-1, train_val_df)

# Extrai só as palavras negativas
palavras_vies_neg_train_val = set([p for p, _ in so_vies_neg_train_val])
palavras_facto_neg_train_val = set([p for p, _ in so_facto_neg_train_val])
palavras_citacao_neg_train_val = set([p for p, _ in so_citacao_neg_train_val])

# Calcula as palavras negativas exclusivas de cada classe
so_vies_neg_train_val = palavras_vies_neg_train_val - palavras_facto_neg_train_val - palavras_citacao_neg_train_val
so_facto_neg_train_val = palavras_facto_neg_train_val - palavras_vies_neg_train_val - palavras_citacao_neg_train_val
so_citacao_neg_train_val = palavras_citacao_neg_train_val - palavras_vies_neg_train_val - palavras_facto_neg_train_val

# Extrai só as palavras positivas
palavras_vies_pos_train_val = set([p for p, _ in so_vies_pos_train_val])
palavras_facto_pos_train_val = set([p for p, _ in so_facto_pos_train_val])
palavras_citacao_pos_train_val = set([p for p, _ in so_citacao_pos_train_val])

# Calcula as palavras positivas exclusivas de cada classe
so_vies_pos_train_val = palavras_vies_pos_train_val - palavras_facto_pos_train_val - palavras_citacao_pos_train_val
so_facto_pos_train_val = palavras_facto_pos_train_val - palavras_vies_pos_train_val - palavras_citacao_pos_train_val
so_citacao_pos_train_val = palavras_citacao_pos_train_val - palavras_vies_pos_train_val - palavras_facto_pos_train_val

# Extrai só as palavras neutras
palavras_vies_neu_train_val = set([p for p, _ in so_vies_neu_train_val])
palavras_facto_neu_train_val = set([p for p, _ in so_facto_neu_train_val])
palavras_citacao_neu_train_val = set([p for p, _ in so_citacao_neu_train_val])

# Calcula as palavras neutras exclusivas de cada classe
so_vies_neu_train_val = palavras_vies_neu_train_val - palavras_facto_neu_train_val - palavras_citacao_neu_train_val
so_facto_neu_train_val = palavras_facto_neu_train_val - palavras_vies_neu_train_val - palavras_citacao_neu_train_val
so_citacao_neu_train_val = palavras_citacao_neu_train_val - palavras_vies_neu_train_val - palavras_facto_neu_train_val

# --- FUNÇÃO PARA CONTAR PALAVRAS EXCLUSIVAS EM CADA FRASE --- #
def contar_palavras_exclusivas(tokens, df):
    contagens = {
            "so_vies_pos": 0, "so_vies_neg": 0, "so_vies_neu": 0,
            "so_facto_pos": 0, "so_facto_neg": 0, "so_facto_neu": 0,
            "so_citacao_pos": 0, "so_citacao_neg": 0, "so_citacao_neu": 0
        }
    if df is train_df or df is test_df:
        for token in tokens:
            token = token.lower()
            if token in so_vies_pos_train:
                contagens["so_vies_pos"] += 1
            if token in so_vies_neg_train:
                contagens["so_vies_neg"] += 1
            if token in so_vies_neu_train:
                contagens["so_vies_neu"] += 1
            if token in so_facto_pos_train:
                contagens["so_facto_pos"] += 1
            if token in so_facto_neg_train:
                contagens["so_facto_neg"] += 1
            if token in so_facto_neu_train:
                contagens["so_facto_neu"] += 1
            if token in so_citacao_pos_train:
                contagens["so_citacao_pos"] += 1
            if token in so_citacao_neg_train:
                contagens["so_citacao_neg"] += 1
            if token in so_citacao_neu_train:
                contagens["so_citacao_neu"] += 1

    elif df is train_val_df or df is val_df:
        for token in tokens:
            token = token.lower()
            if token in so_vies_pos_train:
                contagens["so_vies_pos"] += 1
            if token in so_vies_neg_train:
                contagens["so_vies_neg"] += 1
            if token in so_vies_neu_train:
                contagens["so_vies_neu"] += 1
            if token in so_facto_pos_train:
                contagens["so_facto_pos"] += 1
            if token in so_facto_neg_train:
                contagens["so_facto_neg"] += 1
            if token in so_facto_neu_train:
                contagens["so_facto_neu"] += 1
            if token in so_citacao_pos_train:
                contagens["so_citacao_pos"] += 1
            if token in so_citacao_neg_train:
                contagens["so_citacao_neg"] += 1
            if token in so_citacao_neu_train:
                contagens["so_citacao_neu"] += 1

    return pd.Series(contagens)

# -- TOP ADJETIVOS POR CLASSE -- #
def adjetivos_frequentes (df):
    top_adjetivos_por_classe = {}
    for classe in df["classe"].unique():
        subset = df[df["classe"] == classe]
        todos_adjetivos = []

        for doc in subset["doc"]:
            todos_adjetivos.extend([token.text.lower() for token in doc if token.pos_ == "ADJ"])

        fdist = FreqDist(todos_adjetivos)
        top_adjetivos_por_classe[classe] = fdist.most_common(5) # Top 5 adjetivos
    
    return top_adjetivos_por_classe

# -- TOP ADVÉRBIOS POR CLASSE -- #
def adverbios_frequentes (df):
    top_adverbios_por_classe = {}

    for classe in df["classe"].unique():
        subset = df[df["classe"] == classe]
        todos_adverbios = []

        for doc in subset["doc"]:
            todos_adverbios.extend([token.text.lower() for token in doc if token.pos_ == "ADV"])

        fdist = FreqDist(todos_adverbios)
        top_adverbios_por_classe[classe] = fdist.most_common(5) # Top 5 advérbios
  
    return top_adverbios_por_classe

# ADJETIVOS E ADVÉRBIOS MAIS FREQUENTES NO CONJUNTO DE TREINO
adjs_train = adjetivos_frequentes(train_df)
advs_train = adverbios_frequentes(train_df)

top_adjetivos_citacao_train = [w for w, _ in adjs_train.get(-1, [])]
top_adjetivos_facto_train = [w for w, _ in adjs_train.get(0, [])]
top_adjetivos_vies_train = [w for w, _ in adjs_train.get(1, [])]

top_adverbios_citacao_train = [w for w, _ in advs_train.get(-1, [])]
top_adverbios_facto_train = [w for w, _ in advs_train.get(0, [])]
top_adverbios_vies_train = [w for w, _ in advs_train.get(1, [])]

# ADJETIVOS E ADVÉRBIOS MAIS FREQUENTES NOS DADOS DE TREINO NO CONJUNTO DE VALIDAÇÃO
adjs_train_val = adjetivos_frequentes(train_val_df)
advs_train_val = adverbios_frequentes(train_val_df)

top_adjetivos_citacao_train_val = [w for w, _ in adjs_train_val.get(-1, [])]
top_adjetivos_facto_train_val = [w for w, _ in adjs_train_val.get(0, [])]
top_adjetivos_vies_train_val = [w for w, _ in adjs_train_val.get(1, [])]

top_adverbios_citacao_train_val = [w for w, _ in advs_train_val.get(-1, [])]
top_adverbios_facto_train_val = [w for w, _ in advs_train_val.get(0, [])]
top_adverbios_vies_train_val = [w for w, _ in advs_train_val.get(1, [])]


# -- FUNÇÃO PARA CONTAR ADJETIVOS E ADVÉRBIOS FREQUENTES EM CADA FRASE -- #
def contar_adjetivos_adverbios_freq (tokens, df):
    contagens = {
            "adj_vies" : 0, "adj_facto" : 0, "adj_citacao" : 0, 
            "adv_vies" : 0, "adv_facto" : 0, "adv_citacao" : 0
        }
    if df is train_df or df is test_df:
        for token in tokens:
            token = token.lower()
            if token in top_adjetivos_citacao_train:
                contagens ["adj_citacao"] += 1
            if token in top_adjetivos_facto_train:
                contagens ["adj_facto"] += 1
            if token in top_adjetivos_vies_train:
                contagens ["adj_vies"] += 1
            if token in top_adverbios_citacao_train:
                contagens ["adv_citacao"] += 1
            if token in top_adverbios_facto_train:
                contagens ["adv_facto"] += 1
            if token in top_adverbios_vies_train:
                contagens ["adv_vies"] += 1

    elif df is train_val_df or df is val_df:
        for token in tokens:
            token = token.lower()
            if token in top_adjetivos_citacao_train_val:
                contagens ["adj_citacao"] += 1
            if token in top_adjetivos_facto_train_val:
                contagens ["adj_facto"] += 1
            if token in top_adjetivos_vies_train_val:
                contagens ["adj_vies"] += 1
            if token in top_adverbios_citacao_train_val:
                contagens ["adv_citacao"] += 1
            if token in top_adverbios_facto_train_val:
                contagens ["adv_facto"] += 1
            if token in top_adverbios_vies_train_val:
                contagens ["adv_vies"] += 1

    return pd.Series (contagens)

# -- TEM ASPAS -- #
def contar_aspas(sentence):
    return len(re.findall(r'"', sentence))

# -- NÚMERO DE PALAVRAS POR FRASE -- #
def num_palavras(sentence):
    return len(sentence.split())

# -- TOP PALAVRAS MAIS FREQUENTES POR CLASSE -- #
def palavras_frequentes (df):
    top_palavras_por_classe = {} 
    for classe in df["classe"].unique():
        subset = df[df["classe"] == classe]
        todas_palavras = [t.lower() for lista in subset["tokens"] for t in lista]
        fdist = FreqDist(todas_palavras)
        top_palavras_por_classe[classe] = fdist.most_common(5)

    return [w for w, _ in top_palavras_por_classe.get(-1, [])], [w for w, _ in top_palavras_por_classe.get(0, [])], [w for w, _ in top_palavras_por_classe.get(1, [])]

# Top 5 palavras no treino
top_facto_train, top_citacao_train, top_vies_train = palavras_frequentes(train_df)
# Top 5 palavras nos dados de treino no conjunto de validação
top_facto_train_val, top_citacao_train_val, top_vies_train_val = palavras_frequentes(train_val_df)

## -- CONTAGEM DE PALAVRAS MAIS FREQUENTES DE CADA CLASSE -- ##
def top5_palavras(sentence, classe):
    if classe == -1:
        return len([palavra for palavra in sentence.split() if palavra in top_citacao_train_val])
    elif classe == 0:
        return len([palavra for palavra in sentence.split() if palavra in top_facto_train_val])
    else:
        return len([palavra for palavra in sentence.split() if palavra in top_vies_train_val])
    
# Aplica as funções de contagem de palavras exclusivas aos Dataframes
train_counts_exclusivas = train_df["tokens"].apply(contar_palavras_exclusivas, df = train_df)
train_val_counts_exclusivas = train_val_df["tokens"].apply(contar_palavras_exclusivas, df = train_val_df)
val_counts_exclusivas = val_df["tokens"].apply(contar_palavras_exclusivas, df = val_df)
test_counts_exclusivas = test_df["tokens"].apply(contar_palavras_exclusivas, df = test_df)

# Aplica a função de contagem de adjetivos e advérbios aos Dataframes
train_counts_adj_adv = train_df["tokens"].apply(contar_adjetivos_adverbios_freq, df = train_df)
train_val_counts_adj_adv = train_val_df["tokens"].apply(contar_adjetivos_adverbios_freq, df = train_val_df)
val_counts_adj_adv = val_df["tokens"].apply(contar_adjetivos_adverbios_freq, df = val_df)
test_counts_adj_adv = test_df["tokens"].apply(contar_adjetivos_adverbios_freq, df = test_df)

# Junta as colunas 
train_df_num = pd.concat([train_counts_exclusivas, train_counts_adj_adv], axis = 1)
train_val_df_num = pd.concat([train_val_counts_exclusivas, train_val_counts_adj_adv], axis = 1)
val_df_num = pd.concat([val_counts_exclusivas, val_counts_adj_adv], axis = 1)
test_df_num = pd.concat([test_counts_exclusivas, test_counts_adj_adv], axis = 1)

# COLUNAS DOS DATAFRAMES
colunas = [
    "so_vies_pos", "so_vies_neg", "so_vies_neu",
    "so_facto_pos", "so_facto_neg", "so_facto_neu",
    "so_citacao_pos", "so_citacao_neg", "so_citacao_neu",
    "adj_vies", "adj_facto", "adj_citacao", "adv_vies",
    "adv_facto", "adv_citacao", "num_aspas", "num_palavras", 
    "top5_palavras_vies", "top5_palavras_facto", "top5_palavras_citacao"
]

# Adicionar as colunas com o número de aspas 
train_df_num["num_aspas"] = train_df["sentences"].apply(lambda s: contar_aspas(s))
train_val_df_num["num_aspas"] = train_val_df["sentences"].apply(lambda s: contar_aspas(s))
val_df_num["num_aspas"] = val_df["sentences"].apply(lambda s: contar_aspas(s))
test_df_num["num_aspas"] = test_df["sentences"].apply(lambda s: contar_aspas(s))

# Adicionar as colunas com o número de palavras
train_df_num["num_palavras"] = train_df["sentences"].apply(lambda s: num_palavras(s))
train_val_df_num["num_palavras"] = train_val_df["sentences"].apply(lambda s: num_palavras(s))
val_df_num["num_palavras"] = val_df["sentences"].apply(lambda s: num_palavras(s))
test_df_num["num_palavras"] = test_df["sentences"].apply(lambda s: num_palavras(s))

# Adicionar as colunas com o número de palavras mais frequentes de cada classe
## CLASSE 1
train_df_num["top5_palavras_vies"] = train_df["sentences"].apply(lambda s: top5_palavras(s, 1))
train_val_df_num["top5_palavras_vies"] = train_val_df["sentences"].apply(lambda s: top5_palavras(s, 1))
val_df_num["top5_palavras_vies"] = val_df["sentences"].apply(lambda s: top5_palavras(s, 1))
test_df_num["top5_palavras_vies"] = test_df["sentences"].apply(lambda s: top5_palavras(s, 1))

## CLASSE 0
train_df_num["top5_palavras_facto"] = train_df["sentences"].apply(lambda s: top5_palavras(s, 0))
train_val_df_num["top5_palavras_facto"] = train_val_df["sentences"].apply(lambda s: top5_palavras(s, 0))
val_df_num["top5_palavras_facto"] = val_df["sentences"].apply(lambda s: top5_palavras(s, 0))
test_df_num["top5_palavras_facto"] = test_df["sentences"].apply(lambda s: top5_palavras(s, 0) )

## CLASSE -1
train_df_num["top5_palavras_citacao"] = train_df["sentences"].apply(lambda s: top5_palavras(s, -1))
train_val_df_num["top5_palavras_citacao"] = train_val_df["sentences"].apply(lambda s: top5_palavras(s, -1))
val_df_num["top5_palavras_citacao"] = val_df["sentences"].apply(lambda s: top5_palavras(s, -1))
test_df_num["top5_palavras_citacao"] = test_df["sentences"].apply(lambda s: top5_palavras(s, -1))

# Dataframe de treino após adicionar todas as regras
X_train_num = train_df_num [colunas]
y_train_num = train_df["classe"]

# Dataframe de treino no conjunto de validação após adicionar todas as regras
X_train_val_num = train_val_df_num [colunas]
y_train_val_num = train_val_df ["classe"]

# Dataframe de validação após adicionar todas as regras
X_val_num = val_df_num [colunas]
y_val_num = val_df ["classe"]

# Dataframe de teste após adicionar todas as regras
X_test_num = test_df_num [colunas]
y_test_num = test_df["classe"]

# Para não existirem valores nulos
X_train_num = X_train_num.fillna(0)
X_train_val_num = X_train_val_num.fillna(0)
X_val_num = X_val_num.fillna(0)
X_test_num = X_test_num.fillna(0)

# -- MATRIZ DE CORRELAÇÃO DAS FEATURES NUMÉRICAS -- #
X_corr = pd.concat([X_train_num, y_train_num], axis = 1)
corr = X_corr.corr ()
print (corr)

plt.figure (figsize = (12, 10))
sns.heatmap (corr, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title ("Matriz de Correlação das Features Numéricas", fontsize = 16)
plt.show ()

## -- NORMALIZAÇÃO DE DADOS -- ##
scaler = StandardScaler()
scaler.fit (X_train_num)

# Normalização de dados (treino)
X_train_num_norm = scaler.transform (X_train_num)

# Normalização de dados (treino no conjunto de validação)
X_train_val_num_norm = scaler.transform (X_train_val_num)

# Normalização de dados (validação)
X_val_num_norm = scaler.transform (X_val_num)

# Normalização de dados (teste)
X_test_num_norm = scaler.transform (X_test_num)

# Escolhe apenas as colunas com maior correlação com a saída
X_train_corr = X_train_num_norm [:,[1,6,7,15]]
X_train_val_corr = X_train_val_num_norm [:,[1,6,7,15]]
X_val_corr = X_val_num_norm [:,[1,6,7,15]]
X_test_corr = X_test_num_norm [:,[1,6,7,15]]

## -- MODELO KNN - TESTE DE PARÂMETROS COM OPTUNA -- ##
def KNN_optuna (trial):
    n_neighbors = trial.suggest_int ("n_neighbors", 1, 50)
    weights = trial.suggest_categorical ("weights", ['uniform', 'distance'])
    metric = trial.suggest_categorical ("metric", ['euclidean', 'manhattan', 'minkowski'])
    KNN = KNeighborsClassifier (n_neighbors = n_neighbors, weights = weights, metric = metric) 
    KNN.fit (X_train_val_corr, y_train_val_num)

    y_pred = KNN.predict (X_val_corr)

    score = f1_score(y_val_num, y_pred, average = 'macro')

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

    # Medir tempo de treino
    time_start = time.time()
    KNN.fit (X_train_corr, y_train_num)
    time_end = time.time()
    print(f"Tempo de treino do modelo KNN: {time_end - time_start:.2f}")

    y_pred = KNN.predict (X_test_corr)

    return y_pred

y_pred_KNN = KNN_modelo ()

print("-------- MODELO KNN --------")
print("Accuracy:", accuracy_score(y_test_num, y_pred_KNN))
print("\nRelatório de classificação:")
print(classification_report(test_df ['classe'], y_pred_KNN, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO KNN 
cm = confusion_matrix(test_df ['classe'], y_pred_KNN)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()

## -- MODELO REGRESSÃO LOGÍSTICA - TESTE DE PARÂMETROS COM OPTUNA -- ##
def regressao_logistica_optuna (trial):
    C = trial.suggest_float ("C", 0.01, 10.0, log = True)
    solver = trial.suggest_categorical ("solver", ['saga', 'lbfgs'])
    max_iter = trial.suggest_int ("max_iter", 100, 1000)
    log_reg = LogisticRegression (C = C, solver = solver, max_iter = max_iter, random_state = 42)
    log_reg.fit (X_train_val_corr, y_train_val_num)

    y_pred = log_reg.predict (X_val_corr)

    score = f1_score(y_val_num, y_pred, average = 'macro')

    return score

# Criação e execução do estudo
study_logreg = optuna.create_study(direction = 'maximize')
study_logreg.optimize(regressao_logistica_optuna, n_trials = 400)
print("Melhores parâmetros encontrados para Regressão Logística:")
print(study_logreg.best_params)
print()
print("Melhor F1 (macro) para Regressão Logística:", study_logreg.best_value)
print() 

## -- MODELO REGRESSÃO LOGÍSTICA COM MELHORES PARÂMETROS -- ##
def regressao_logistica ():
    log_reg = LogisticRegression (**study_logreg.best_params, random_state = 42)

    # Medir tempo de treino
    time_start = time.time()
    log_reg.fit (X_train_corr, y_train_num)
    time_end = time.time()
    print(f"Tempo de treino do modelo Regressão Logística: {time_end - time_start:.2f}")

    y_pred = log_reg.predict (X_test_corr)
    return y_pred

y_pred_reglog = regressao_logistica ()

print("-------- MODELO REGRESSÃO LOGÍSTICA --------")
print("Accuracy:", accuracy_score(y_test_num, y_pred_reglog))
print("\nRelatório de classificação:")
print(classification_report(test_df ['classe'], y_pred_reglog, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO REGRESSÃO LOGÍSTICA
cm = confusion_matrix(test_df ['classe'], y_pred_reglog)
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
    nb.fit (X_train_val_corr, y_train_val_num)

    y_pred = nb.predict (X_val_corr)
    score = f1_score (y_val_num, y_pred, average = 'macro')
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

    # Medir tempo de treino
    time_start = time.time()
    nb.fit(X_train_corr, y_train_num)
    time_end = time.time()
    print(f"Tempo de treino do modelo Naive Bayes: {time_end - time_start:.2f}")

    y_pred = nb.predict(X_test_corr)
    return y_pred

y_pred_nb = naive_bayes_modelo()

print("-------- MODELO NAIVE BAYES --------")
print("Accuracy:", accuracy_score(y_test_num, y_pred_nb))
print("\nRelatório de classificação:")
print(classification_report(test_df ['classe'], y_pred_nb, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO NAIVE BAYES
cm = confusion_matrix(test_df ['classe'], y_pred_nb)
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
    rf.fit(X_train_val_corr, y_train_val_num)

    y_pred = rf.predict(X_val_corr)
    score = f1_score (y_val_num, y_pred, average = 'macro')
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
    # Medir tempo de treino
    time_start = time.time()
    rf.fit (X_train_corr, y_train_num)
    time_end = time.time()
    print(f"Tempo de treino do modelo Random Forest: {time_end - time_start:.2f}")

    y_pred = rf.predict (X_test_corr)
    return y_pred

y_pred_rf = random_forest_modelo()

print("-------- MODELO RANDOM FOREST --------")
print("Accuracy:", accuracy_score(y_test_num, y_pred_rf))
print("\nRelatório de classificação:")
print(classification_report(test_df ['classe'], y_pred_rf, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO RANDOM FOREST
cm = confusion_matrix(test_df ['classe'], y_pred_rf)
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
    
    rnn.fit (X_train_val_corr, y_train_val_num)

    y_pred = rnn.predict (X_val_corr)
    score = f1_score (y_val_num, y_pred, average = 'macro')
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
    # Medir tempo de treino
    time_start = time.time()
    rnn.fit (X_train_corr, y_train_num)
    time_end = time.time()
    print(f"Tempo de treino do modelo Rede Neuronal: {time_end - time_start:.2f}")

    y_pred = rnn.predict (X_test_corr)
    return y_pred

y_pred_rnn = rede_neuronal_modelo ()

print("-------- MODELO REDE NEURONAL --------")
print("Accuracy:", accuracy_score(y_test_num, y_pred_rnn))
print("\nRelatório de classificação:")
print(classification_report(test_df ['classe'], y_pred_rnn, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO REDE NEURONAL
cm = confusion_matrix(test_df ['classe'], y_pred_rnn)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão do Modelo Rede Neuronal", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()
