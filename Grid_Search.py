# Import de bibliotecas necessárias
import spacy 
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from itertools import product

# Download dos recursos necessários do NLTK
nltk.download('wordnet')
nltk.download('stopwords')

# Lista de stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Leitura do ficheiro csv com o dataset
data = pd.read_csv ('factnews_dataset.csv', delimiter = ',') 
df = pd.DataFrame (data)

# Seleção da coluna de frases
sentences = df.iloc [:,-2]

# Contagem de amostras por classe
tam_classes = df["classe"].value_counts()
print ("\n", tam_classes)

# Confirmar que não há linhas duplicadas
print(f"\nNúmero de linhas duplicadas: {df.duplicated().sum()}")

# Verificar valores nulos
print(f"\nNúmero de células nulas: {df.isnull().sum().sum()}")

# Carregar modelo spacy para português
nlp = spacy.load("pt_core_news_sm")

# -- FUNÇÃO DE LIMPEZA E TOKENIZAÇÃO -- #
def limpar_tokenizar(texto):
    tokens = word_tokenize(str(texto).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) != 1]
    return tokens

# Drop de colunas não necessárias 
X = df.drop(columns = ["file", "classe"])

# Usar dummies para transformar colunas categóricas em binárias (True, False)
X_encoded = pd.get_dummies(X, columns = ["id_article", "domain"])

# y é a coluna de classes (target)
y = df.iloc[:, -1]

# Divisão dos dados em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split (X_encoded, y, test_size = 0.30, random_state = 42, stratify = y)

## Criação dos dataframes de Treino ##
train_df = X_train.copy()
train_df["classe"] = y_train.values
# Transformar True e False em 1 e 0 para calcular correlações
train_df = train_df.apply(lambda col: col.astype(int) if col.dtypes == 'bool' else col)

# Contagem de amostras por classe nos dados de treino
print(f"\n{train_df["classe"].value_counts()}")

# Não usar as frases para calcular a correlação
train_df_without_sentences = train_df.drop(columns = ["sentences"])

# Calcular correlação
corr = train_df_without_sentences.corr()["classe"].sort_values(ascending = False)

print("\n--Correlação das colunas com a classe:--")
print(corr)

# Adicionar as duas colunas de "tokens" e "doc" ao dataframe de treino
train_df["tokens"] = train_df["sentences"].apply(limpar_tokenizar)
train_df["doc"] = list(nlp.pipe(train_df["sentences"], batch_size = 50))

## Criação dos dataframes de Teste ##
test_df = X_test.copy()
test_df["classe"] = y_test.values
# Criar coluna "doc" com objetos spaCy para análise das frases
test_df["doc"] = list(nlp.pipe(test_df["sentences"], batch_size = 50))


# -- ADJETIVOS POR CLASSE -- #
def adjetivos_frequentes ():
    top_adjetivos_por_classe = {}
    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todos_adjetivos = []

        for doc in subset["doc"]:
            todos_adjetivos.extend([token.text.lower() for token in doc if token.pos_ == "ADJ"])

        fdist = FreqDist(todos_adjetivos)
        top_adjetivos_por_classe[classe] = fdist.most_common(5)  # Top 5 adjetivos
    
    return top_adjetivos_por_classe


# -- ADVÉRBIOS POR CLASSE -- #
def adverbios_frequentes ():
    top_adverbios_por_classe = {}

    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todos_adverbios = []

        for doc in subset["doc"]:
            todos_adverbios.extend([token.text.lower() for token in doc if token.pos_ == "ADV" and token.text.lower().endswith("mente")])

        fdist = FreqDist(todos_adverbios)
        top_adverbios_por_classe[classe] = fdist.most_common(5)  # Top 5 advérbios
  
    return top_adverbios_por_classe


# -- PALAVRAS MAIS FREQUENTES POR CLASSE -- #
def palavras_frequentes ():
    top_palavras_por_classe = {} 
    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todas_palavras = [t.lower() for lista in subset["tokens"] for t in lista]
        fdist = FreqDist(todas_palavras)
        top_palavras_por_classe[classe] = fdist.most_common(5)

    return top_palavras_por_classe   


# -- VERBOS MAIS FREQUENTES POR CLASSE -- #
def verbos_frequentes ():
    top_verbos_por_classe = {} 
    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todos_verbos = []

        for doc in subset["doc"]:
            todos_verbos.extend([token.lemma_.lower() for token in doc if token.pos_ == "VERB"])

        fdist = FreqDist(todos_verbos)
        top_verbos_por_classe[classe] = fdist.most_common(5)  # Top 5 verbos
    
    return top_verbos_por_classe


# -- SINÓNIMOS DE UMA PALAVRA -- #
def get_sinonimos (word):
    sinonimos_ls = []
    for syn in wordnet.synsets(word, lang = 'por'):
        for lemma in syn.lemmas('por'):
            sinonimos_ls.append(lemma.name())

    sinonimos_ls = list(set(sinonimos_ls)) # Remove palavras duplicadas
    return sinonimos_ls

# -- TEM ASPAS -- #
def contar_aspas(sentence):
    return len(re.findall(r'"', sentence))


# -- PRIMEIRA LETRA MAIÚSCULA DEPOIS DAS ASPAS -- #
def letras_maiusculas (sentence):
    return len(re.findall(r'"[A-ZÁÉÍÓÚÃÕÂÊÔÇ]', sentence))


# -- Chamadas das funções -- #
adjetivos_frequentes_por_classe = adjetivos_frequentes()
adverbios_frequentes_por_classe = adverbios_frequentes()
palavras_frequentes_por_classe = palavras_frequentes()
verbos_frequentes_por_classe = verbos_frequentes()


# -- SINÓNIMOS DE UMA LISTA DE VERBOS -- #
def verbos_sinonimos (verbos_frequentes, classe):
    sinonimos = []
    verbos_frequentes = [t[0] for t in verbos_frequentes.get(classe)]
    for verbo in verbos_frequentes:
        if verbo not in sinonimos: 
            sinonimos.append (verbo)

        sinonimos_verbo = get_sinonimos (verbo)

        for sinonimo in sinonimos_verbo:
            if sinonimo not in sinonimos:
                sinonimos.append (sinonimo)
    
    return sinonimos


# -- CRIAÇÃO DO FICHEIRO (CHAVES: PALAVRAS, VALORES: PoS e Polaridade da palavra) -- #
def ficheiro_sentilex ():
    sentimentos = {}
    with open ("SentiLex.csv", encoding = "utf-8") as f:
        for line in f:
            line = line.strip ()
            parts = line.split(",")
            palavra = parts [0]
            PoS = parts [2].split("=")[1]
            POL = parts [5].split('=')[1]
            sentimentos[palavra] = [PoS, POL]

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
            POL = sentimentos[word][1]
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


# -- PALAVRAS NEGATIVAS E POSITIVAS MAIS FREQUENTES POR CLASSE -- #
def palavras_polaridade_top (classe):
    todas_palavras_negativas = Counter()
    todas_palavras_positivas = Counter()
    todas_palavras_neutras = Counter()
    for i, sentence in enumerate(train_df["sentences"]):
        if train_df["classe"].iloc[i] == classe:
            _, _, _, palavras_negativas, palavras_positivas, palavras_neutras = contagem_por_polaridade(sentimentos, sentence)
            todas_palavras_negativas.update(palavras_negativas)
            todas_palavras_positivas.update(palavras_positivas)
            todas_palavras_neutras.update(palavras_neutras)

    top_negativas = todas_palavras_negativas.most_common()
    top_positivas = todas_palavras_positivas.most_common()
    top_neutras = todas_palavras_neutras.most_common()
    return top_negativas, top_positivas, top_neutras


# -- PALAVRAS EXCLUSIVAS DA CLASSE VIÉS -- #
# Obtém as palavras mais frequentes por polaridade e classe
so_vies_neg, so_vies_pos, so_vies_neu = palavras_polaridade_top(1)
so_facto_neg, so_facto_pos, so_facto_neu = palavras_polaridade_top(0)
so_citacao_neg, so_citacao_pos, so_citacao_neu = palavras_polaridade_top(-1)

# Extrai só as palavras negativas
palavras_vies_neg = set([p for p, _ in so_vies_neg])
palavras_facto_neg = set([p for p, _ in so_facto_neg])
palavras_citacao_neg = set([p for p, _ in so_citacao_neg])

# Calcula as palavras negativas exclusivas da classe viés
so_vies_neg = palavras_vies_neg - palavras_facto_neg - palavras_citacao_neg

# Extrai só as palavras positivas
palavras_vies_pos = set([p for p, _ in so_vies_pos])
palavras_facto_pos = set([p for p, _ in so_facto_pos])
palavras_citacao_pos = set([p for p, _ in so_citacao_pos])

# Calcula as palavras positivas exclusivas da classe viés
so_vies_pos = palavras_vies_pos - palavras_facto_pos - palavras_citacao_pos

# Extrai só as palavras neutras
palavras_vies_neu = set([p for p, _ in so_vies_neu])
palavras_facto_neu = set([p for p, _ in so_facto_neu])
palavras_citacao_neu = set([p for p, _ in so_citacao_neu])

# Calcula as palavras neutras exclusivas da classe viés
so_vies_neu = palavras_vies_neu - palavras_facto_neu - palavras_citacao_neu


# -- CONTAGEM DE ASPAS NUMA FRASE -- #
def contagem_aspas(sentence):
    if len(sentence.strip()) == 0:
        return 0
    return sentence.count('"')

# -- CONTAGEM DE ASPAS POR CLASSE -- #
def contagem_aspas_por_classes ():
    cont_aspas_menos1 = []
    cont_aspas_0 = []
    cont_aspas_1 = []
    for i, sentence in enumerate (train_df["sentences"]):
        classe = train_df["classe"].iloc[i]
        cont = contagem_aspas(sentence)

        if classe == -1:
            cont_aspas_menos1.append(cont)
        elif classe == 0:
            cont_aspas_0.append(cont)
        elif classe == 1:
            cont_aspas_1.append(cont)

    return cont_aspas_menos1, cont_aspas_0, cont_aspas_1

aspas_menos1, aspas_0, aspas_1 = contagem_aspas_por_classes()


# -- CONTAGEM DE PALAVRAS POR POLARIDADE E POR CLASSE
def contagem_polaridade (classe, sentimentos):
    cont_negatividade = []
    cont_neutralidade = []  
    cont_positividade = []
    for i, sentence in enumerate (train_df["sentences"]):
        if train_df["classe"].iloc[i] == classe:
            negatividade, neutralidade, positividade, _, _, _ = contagem_por_polaridade (sentimentos, sentence)
            cont_negatividade.append (negatividade)
            cont_neutralidade.append (neutralidade)
            cont_positividade.append (positividade)

    return cont_negatividade, cont_neutralidade, cont_positividade     

cont_neg_1, cont_neu_1, cont_pos_1 = contagem_polaridade(1, sentimentos)
cont_neg_0, cont_neu_0, cont_pos_0 = contagem_polaridade(0, sentimentos)
cont_neg_n1, cont_neu_n1, cont_pos_n1 = contagem_polaridade(-1, sentimentos)


# -- CONTAGEM DE ADJETIVOS E ADVÉRBIOS POR CLASSE -- #
def contagem_adj_adv_por_classe():
    cont_adj_adv_menos1 = []
    cont_adj_adv_0 = []
    cont_adj_adv_1 = []

    for i, doc in enumerate(train_df["doc"]):
        classe = train_df["classe"].iloc[i]
        
        # Conta tokens que são ADJ (adjetivos) ou ADV (advérbios)
        num_adj_adv = sum(1 for token in doc if token.pos_ in ["ADJ", "ADV"])

        if classe == -1:
            cont_adj_adv_menos1.append(num_adj_adv)
        elif classe == 0:
            cont_adj_adv_0.append(num_adj_adv)
        elif classe == 1:
            cont_adj_adv_1.append(num_adj_adv)

    return cont_adj_adv_menos1, cont_adj_adv_0, cont_adj_adv_1

cont_adj_adv_menos1, cont_adj_adv_0, cont_adj_adv_1 = contagem_adj_adv_por_classe()


# -- CONTAGEM DE MAIÚSCULAS POR CLASSE -- #
def contagem_maiusculas_por_classe():
    cont_maiuscula_menos1 = []
    cont_maiuscula_0 = []
    cont_maiuscula_1 = []

    for i, doc in enumerate(train_df["doc"]):
        classe = train_df["classe"].iloc[i]
        
        cont_maiuscula = letras_maiusculas(train_df["sentences"].iloc[i])

        if classe == -1:
            cont_maiuscula_menos1.append(cont_maiuscula)
        elif classe == 0:
            cont_maiuscula_0.append(cont_maiuscula)
        elif classe == 1:
            cont_maiuscula_1.append(cont_maiuscula)

    return cont_adj_adv_menos1, cont_adj_adv_0, cont_adj_adv_1

cont_maiuscula_menos1, cont_maiuscula_0, cont_maiuscula_1 = contagem_maiusculas_por_classe()


# -- DEFINE A POLARIDADE DA FRASE -- #
def polaridade_frase (linha):
    negatividade, neutralidade, positividade, _, _, _ = contagem_por_polaridade (sentimentos, linha)
    valores = [negatividade, neutralidade, positividade]
    valor_maximo = max (valores)
    indice_maximo = valores.index (valor_maximo)
    return indice_maximo - 1, valores


# -- FREQUÊNCIAS DE PALAVRAS NEGATIVAS NUMA FRASE -- #
def frequencias_negativas_frase (linha):
    sentence = test_df.iloc[linha]["sentences"]
    tam = len(sentence.split())
    negatividade = polaridade_frase (sentence)[1][0]
    return negatividade/tam


# -- NÚMERO DE ADJETIVOS E ADVÉRBIOS NUMA FRASE -- #
def num_adj_adv(linha):
    doc = test_df.iloc[linha]["doc"]
    return sum(1 for token in doc if token.pos_ in ["ADV", "ADJ"])


# -- NÚMERO DE PALAVRAS NUMA FRASE -- #
def num_palavras(linha):
    frase = test_df.iloc[linha]["sentences"]
    return len(frase.split(" "))


# --- APLICA PESOS DE 0.5, 1 E 2 ÀS REGRAS E OBTÉM A PONTUAÇÃO EM CADA CLASSE --- #
def detetar_por_pesos(linha, classe, pesos):
    
    if classe == -1:
        doc_linha = test_df.iloc[linha]["doc"]
        adverbios = [t[0] for t in adverbios_frequentes_por_classe.get(-1)]
        adjetivos = [t[0] for t in adjetivos_frequentes_por_classe.get(-1)]
        palavras = [t[0] for t in palavras_frequentes_por_classe.get(-1)]
        verbos = set(verbos_sinonimos(verbos_frequentes_por_classe, -1))
        cond_aspas = 1 if contagem_aspas(test_df.iloc[linha]["sentences"]) == 1 else 0

    elif classe == 0:
        doc_linha = test_df.iloc[linha]["doc"]
        adverbios = [t[0] for t in adverbios_frequentes_por_classe.get(0)]
        adjetivos = [t[0] for t in adjetivos_frequentes_por_classe.get(0)]
        palavras = [t[0] for t in palavras_frequentes_por_classe.get(0)]
        verbos = set(verbos_sinonimos(verbos_frequentes_por_classe, 0))
        cond_aspas = 1 if contagem_aspas(test_df.iloc[linha]["sentences"]) == 0 else 0

    elif classe == 1:
        doc_linha = test_df.iloc[linha]["doc"]
        adverbios = [t[0] for t in adverbios_frequentes_por_classe.get(1)]
        adjetivos = [t[0] for t in adjetivos_frequentes_por_classe.get(1)]
        palavras = [t[0] for t in palavras_frequentes_por_classe.get(1)]
        verbos = set(verbos_sinonimos(verbos_frequentes_por_classe, 1))
        cond_aspas = 1 if contagem_aspas(test_df.iloc[linha]["sentences"]) == 0 else 0

    # --- Calcula a pontuação ponderada --- #
    pontuacao = 0
    pontuacao += cond_aspas * 2
    pontuacao += pesos["peso_verbos"] * any(token.lemma_.lower() in verbos for token in doc_linha)
    pontuacao += pesos["peso_adverbios"] * any(token.text.lower() in adverbios for token in doc_linha)
    pontuacao += pesos["peso_adjetivos"] * any(token.text.lower() in adjetivos for token in doc_linha)
    pontuacao += pesos["peso_palavras"] * any(token.text.lower() in palavras for token in doc_linha)
    

    # --- Regras adicionais só para a classe viés --- #
    if classe == 1:
        # Frequência negativa
        pontuacao += pesos["peso_freq_neg"] * (frequencias_negativas_frase(linha) > 0.075)
        # Nova regra (adjetivos + advérbios + maiúsculas + nº de palavras)
        if (num_adj_adv(linha) + letras_maiusculas(test_df["sentences"].iloc[linha])) >= 4 and num_palavras(linha) > 18:
            pontuacao += pesos["peso_comb_vies"]
        pontuacao += 0.5 * any(token.text.lower() in so_vies_neg for token in doc_linha)
        pontuacao += 0.5 * any(token.text.lower() in so_vies_pos for token in doc_linha)
        pontuacao += 0.5 * any(token.text.lower() in so_vies_neu for token in doc_linha)

    return pontuacao


# --- GRID SEARCH SEPARADO POR CLASSE --- #
pesos_possiveis = [0.5, 1, 2]

condicoes = [
    "peso_verbos", "peso_adverbios",
    "peso_adjetivos", "peso_palavras",
    "peso_freq_neg", "peso_comb_vies"  
]

melhor_f1 = 0
melhor_combinacao = None
iter = 0

# Testa todas as combinações possíveis de pesos
for comb in product(pesos_possiveis, repeat = len(condicoes)):
    iter += 1
    pesos = dict(zip(condicoes, comb))

    classificacao_temp = []
    for i in range(len(test_df["sentences"])):
        scores = [
            detetar_por_pesos(i, -1, pesos),
            detetar_por_pesos(i, 0, pesos),
            detetar_por_pesos(i, 1, pesos),
        ]
        indice_max = scores.index(max(scores))
        classificacao_temp.append(indice_max - 1)

    f1_macro = f1_score(test_df["classe"], classificacao_temp, average = 'macro')
    acc = accuracy_score(test_df["classe"], classificacao_temp)

    print(f"\n{iter}ª combinação: F1 (macro avg): {f1_macro} | Accuracy: {acc}")
    print(pesos)

    # Atualiza a melhor combinação se o F1 for maior
    if f1_macro > melhor_f1:
        melhor_f1 = f1_macro
        melhor_combinacao = pesos

print(f"\nMelhor combinação: {melhor_combinacao}")
print(f"Melhor média de F1: {melhor_f1}")