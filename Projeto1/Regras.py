# Import de bibliotecas necessárias
import spacy 
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

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

# Criar vectorizer para bigramas
vectorizer = CountVectorizer(ngram_range = (2, 2), token_pattern = r'\b\w+\b', min_df = 1)
# Transformar tokens em matriz de bigramas
X_ngrams = vectorizer.fit_transform(train_df["tokens"].apply(lambda tokens: ' '.join(tokens)))
# Obter nomes dos bigramas
bigrams = vectorizer.get_feature_names_out()
# Adicionar coluna com a matriz de bigramas
train_df["bigrams"] = list(X_ngrams.toarray())

## Criação dos dataframes de Teste ##
test_df = X_test.copy()
test_df["classe"] = y_test.values
# Criar coluna "doc" com objetos spaCy para análise das frases
test_df["doc"] = list(nlp.pipe(test_df["sentences"], batch_size = 50))

# -- BIGRAMAS MAIS FREQUENTES POR CLASSE -- #
def bigramas_frequentes ():
    top_bigrams_por_classe = {}
    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todos_bigrams = []

        for bigrams_array in subset["bigrams"]:
            indices = bigrams_array.nonzero()[0]
            todos_bigrams.extend([bigrams[i] for i in indices])

        fdist = FreqDist(todos_bigrams)
        top_bigrams_por_classe[classe] = fdist.most_common(5) # Top 5 bigramas

    return top_bigrams_por_classe


# -- ADJETIVOS POR CLASSE -- #
def adjetivos_frequentes ():
    top_adjetivos_por_classe = {}
    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todos_adjetivos = []

        for doc in subset["doc"]:
            todos_adjetivos.extend([token.text.lower() for token in doc if token.pos_ == "ADJ"])

        fdist = FreqDist(todos_adjetivos)
        top_adjetivos_por_classe[classe] = fdist.most_common(5) # Top 5 adjetivos
    
    return top_adjetivos_por_classe


# -- ADVÉRBIOS POR CLASSE -- #
def adverbios_frequentes ():
    top_adverbios_por_classe = {}

    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todos_adverbios = []

        for doc in subset["doc"]:
            todos_adverbios.extend([token.text.lower() for token in doc if token.pos_ == "ADV"])

        fdist = FreqDist(todos_adverbios)
        top_adverbios_por_classe[classe] = fdist.most_common(5) # Top 5 advérbios
  
    return top_adverbios_por_classe


# -- ENTIDADES NOMINADAS POR CLASSE -- #
def entidades_frequentes ():
    top_entidades_por_classe = {}

    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        tipos_entidades = []
        for doc in subset["doc"]:
            tipos_entidades.extend([ent.label_ for ent in doc.ents])

        fdist = FreqDist(tipos_entidades)
        top_entidades_por_classe[classe] = fdist.most_common(5) # Top 5 entidades
  
    return top_entidades_por_classe


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
        top_verbos_por_classe[classe] = fdist.most_common(5) # Top 5 verbos
    
    return top_verbos_por_classe


# Dicionário de tradução das funções sintáticas mais comuns
traduzir_dep = {
    "nsubj": "sujeito",
    "obj": "objeto direto",
    "iobj": "objeto indireto",
    "obl": "complemento oblíquo",
    "advmod": "modificador adverbial",
    "amod": "modificador adjetival",
    "det": "determinante",
    "nmod": "modificador nominal",
    "case": "preposição",
    "punct": "pontuação",
    "cc": "conjunção coordenativa",
    "mark": "conjunção subordinativa",
    "ROOT": "raiz da frase",
    "cop": "verbo copulativo",
    "aux": "verbo auxiliar",
    "compound": "palavra composta",
    "appos": "aposição",
    "ccomp": "oração completiva",
    "xcomp": "complemento oracional sem sujeito",
}


# -- FUNÇÕES SINTÁTICAS MAIS FREQUENTES POR CLASSE (em português) -- #
def funcoes_sintaticas_frequentes():
    top_funcoes_por_classe = {}
    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todas_funcoes = []

        for doc in subset["doc"]:
            for token in doc:
                dep = traduzir_dep.get(token.dep_, token.dep_) # Traduz, se existir
                todas_funcoes.append(dep)

        fdist = FreqDist(todas_funcoes)
        top_funcoes_por_classe[classe] = fdist.most_common(5) # Top 5 funções sintáticas

    return top_funcoes_por_classe


# -- SINÓNIMOS DE UMA PALAVRA -- #
def get_sinonimos (word):
    sinonimos_ls = []
    for syn in wordnet.synsets(word, lang = 'por'):
        for lemma in syn.lemmas('por'):
            sinonimos_ls.append(lemma.name())

    sinonimos_ls = list(set(sinonimos_ls)) # Remove palavras duplicadas
    return sinonimos_ls


# -- PRIMEIRA LETRA MAIÚSCULA DEPOIS DAS ASPAS -- #
def letras_maiusculas (sentence):
    return len(re.findall(r'"[A-ZÁÉÍÓÚÃÕÂÊÔÇ]', sentence))


# -- TEM ASPAS -- #
def contar_aspas(sentence):
    return len(re.findall(r'"', sentence))


# -- TEM PONTO DE INTERROGAÇÃO -- #
def tem_pontuacao (sentence, pontuacao):
    return pontuacao in sentence

print("\nVerbos mais frequentes por classe:")
verbos_frequentes_por_classe = verbos_frequentes()
for classe, palavras in verbos_frequentes_por_classe.items():
    print(f"Classe {classe}: {palavras}")

print("\nAdjetivos mais frequentes por classe:")
adjetivos_frequentes_por_classe = adjetivos_frequentes()
for classe, palavras in adjetivos_frequentes_por_classe.items():
    print(f"Classe {classe}: {palavras}")

print("\nAdvérbios mais frequentes por classe:")
adverbios_frequentes_por_classe = adverbios_frequentes()
for classe, palavras in adverbios_frequentes_por_classe.items():
    print(f"Classe {classe}: {palavras}")

print("\nPalavras mais frequentes por classe:")
palavras_frequentes_por_classe = palavras_frequentes()
for classe, palavras in palavras_frequentes_por_classe.items():
    print(f"Classe {classe}: {palavras}")

print("\nEntidades mais frequentes por classe:")
entidades_frequentes_por_classe = entidades_frequentes()
for classe, palavras in entidades_frequentes_por_classe.items():
    print(f"Classe {classe}: {palavras}")

print("\nFunções sintáticas mais frequentes por classe:")
funcoes_frequentes_por_classe = funcoes_sintaticas_frequentes()
for classe, palavras in funcoes_frequentes_por_classe.items():
    print(f"Classe {classe}: {palavras}")

print("\nBigramas mais frequentes por classe:")
bigramas_frequentes_por_classe = bigramas_frequentes()
for classe, palavras in bigramas_frequentes_por_classe.items():
    print(f"Classe {classe}: {palavras}")
print()

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


# -- PALAVRAS NEGATIVAS E POSITIVAS MAIS FREQUENTES POR CLASSE -- #
def palavras_polaridade_top (classe):
    todas_palavras_negativas = Counter()
    todas_palavras_positivas = Counter()
    todas_palavras_neutras = Counter()
    freq_neg = []
    for i, sentence in enumerate(train_df["sentences"]):
        if train_df["classe"].iloc[i] == classe:
            negativas, _, _, palavras_negativas, palavras_positivas, palavras_neutras = contagem_por_polaridade(sentimentos, sentence)
            todas_palavras_negativas.update(palavras_negativas)
            todas_palavras_positivas.update(palavras_positivas)
            todas_palavras_neutras.update(palavras_neutras)

            # Frequência de palavras negativas nas frases
            freq_neg.append(negativas/len(sentence.split()))

    top_negativas = todas_palavras_negativas.most_common()
    top_positivas = todas_palavras_positivas.most_common()
    top_neutras = todas_palavras_neutras.most_common()
    return top_negativas, top_positivas, top_neutras, freq_neg

#print ("\nPalavras negativas mais frequentes na classe Viés: ", palavras_polaridade_top(1)[0])
#print ("\nPalavras negativas mais frequentes na classe Facto: ", palavras_polaridade_top(0)[0])
#print ("\nPalavras negativas mais frequentes na classe Citação: ", palavras_polaridade_top(-1)[0])

#print ("\nPalavras positivas mais frequentes na classe Viés: ", palavras_polaridade_top(1)[1])
#print ("\nPalavras positivas mais frequentes na classe Facto: ", palavras_polaridade_top(0)[1])
#print ("\nPalavras positivas mais frequentes na classe Citação: ", palavras_polaridade_top(-1)[1])

#print ("\nPalavras neutras mais frequentes na classe Viés: ", palavras_polaridade_top(1)[2])
#print ("\nPalavras neutras mais frequentes na classe Facto: ", palavras_polaridade_top(0)[2])
#print ("\nPalavras neutras mais frequentes na classe Citação: ", palavras_polaridade_top(-1)[2])


# -- PALAVRAS EXCLUSIVAS DE CADA CLASSE -- #
# Obtém as palavras mais frequentes por polaridade e classe
so_vies_neg, so_vies_pos, so_vies_neu, _ = palavras_polaridade_top(1)
so_facto_neg, so_facto_pos, so_facto_neu, _ = palavras_polaridade_top(0)
so_citacao_neg, so_citacao_pos, so_citacao_neu, _ = palavras_polaridade_top(-1)

# Extrai só as palavras negativas
palavras_vies_neg = set([p for p, _ in so_vies_neg])
palavras_facto_neg = set([p for p, _ in so_facto_neg])
palavras_citacao_neg = set([p for p, _ in so_citacao_neg])

# Calcula as palavras negativas exclusivas de cada classe
so_vies_neg = palavras_vies_neg - palavras_facto_neg - palavras_citacao_neg
so_facto_neg = palavras_facto_neg - palavras_vies_neg - palavras_citacao_neg
so_citacao_neg = palavras_citacao_neg - palavras_vies_neg - palavras_facto_neg

#print("Palavras só de Viés:", so_vies)
#print("Palavras só de Facto:", so_facto)
#print("Palavras só de Citação:", so_citacao)

# Extrai só as palavras positivas
palavras_vies_pos = set([p for p, _ in so_vies_pos])
palavras_facto_pos = set([p for p, _ in so_facto_pos])
palavras_citacao_pos = set([p for p, _ in so_citacao_pos])

# Calcula as palavras positivas exclusivas de cada classe
so_vies_pos = palavras_vies_pos - palavras_facto_pos - palavras_citacao_pos
so_facto_pos = palavras_facto_pos - palavras_vies_pos - palavras_citacao_pos
so_citacao_pos = palavras_citacao_pos - palavras_vies_pos - palavras_facto_pos

#print("Palavras só de Viés:", so_vies_pos)
#print("Palavras só de Facto:", so_facto_pos)
#print("Palavras só de Citação:", so_citacao_pos)

# Extrai só as palavras neutras
palavras_vies_neu = set([p for p, _ in so_vies_neu])
palavras_facto_neu = set([p for p, _ in so_facto_neu])
palavras_citacao_neu = set([p for p, _ in so_citacao_neu])

# Calcula as palavras neutras exclusivas de cada classe
so_vies_neu = palavras_vies_neu - palavras_facto_neu - palavras_citacao_neu
so_facto_neu = palavras_facto_neu - palavras_vies_neu - palavras_citacao_neu
so_citacao_neu = palavras_citacao_neu - palavras_vies_neu - palavras_facto_neu

#print("Palavras só de Viés:", so_vies_neu)
#print("Palavras só de Facto:", so_facto_neu)
#print("Palavras só de Citação:", so_citacao_neu)


# -- FUNÇÃO QUE CRIA BOXPOTS DAS TRÊS CLASSES -- #
def boxplot(cont_n1, cont_0, cont_1, titulo, y_label, linhas, linha_freq_neg):
    plt.figure(figsize = (8, 6))
    dados = [cont_n1, cont_0, cont_1]

    plt.boxplot(dados,
                tick_labels = ['Citação', 'Facto', 'Vies'],
                patch_artist = True,
                boxprops = dict(facecolor = "#bcd1fc"),
                medianprops=dict(color = "#ff0000", linewidth = 2))
    if linhas:
        # Adiciona o valor da mediana sobre cada linha vermelha
        for i, d in enumerate(dados):
            med = np.median(d)
            plt.text(i + 1, med, f'{med:.2f}', ha = 'center', va = 'bottom', color = 'red', fontsize = 16)
    
    if linha_freq_neg:
        plt.axhline(y = 0.075, color = 'gray', linestyle = '--', linewidth = 2, alpha = 0.6)
        plt.text(3.1, 0.075, '0.075', color = 'gray', fontsize = 12, va = 'center')

    # Título e rótulos
    plt.title(titulo, fontsize = 22)
    plt.ylabel(y_label, fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.grid(True, linestyle = '--', alpha = 0.6)

    plt.show()


# Frequência de palavras negativas nas classes
_, _, _, freq_neg_menos1 = palavras_polaridade_top(-1)
_, _, _, freq_neg_0 = palavras_polaridade_top(0)
_, _, _, freq_neg_1 = palavras_polaridade_top(1)
boxplot(freq_neg_menos1, freq_neg_0, freq_neg_1, "Frequência de palavras negativas nas frases", "Nº de palavras negativas", True, True)


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
boxplot(aspas_menos1, aspas_0, aspas_1, "Contagem de Aspas Por Classe", "Contagem de Aspas", False, False)


# -- CONTAGEM DE PALAVRAS POR POLARIDADE E POR CLASSE
def contagem_polaridade (classe, sentimentos):
    cont_negatividade = []
    cont_neutralidade = []  
    cont_positividade = []
    for i, sentence in enumerate (train_df["sentences"]):
        if train_df["classe"].iloc[i] == classe:
            negatividade, neutralidade, positividade, _, _, _= contagem_por_polaridade (sentimentos, sentence)
            cont_negatividade.append (negatividade)
            cont_neutralidade.append (neutralidade)
            cont_positividade.append (positividade)

    return cont_negatividade, cont_neutralidade, cont_positividade     


cont_neg_1, cont_neu_1, cont_pos_1 = contagem_polaridade(1, sentimentos)
cont_neg_0, cont_neu_0, cont_pos_0 = contagem_polaridade(0, sentimentos)
cont_neg_n1, cont_neu_n1, cont_pos_n1 = contagem_polaridade(-1, sentimentos)

boxplot (cont_neg_n1, cont_neg_0, cont_neg_1, "Distribuição do número de palavras negativas nas frases por classe", "Contagem de Palavras Negativas", False, False)
boxplot (cont_neu_n1, cont_neu_0, cont_neu_1, "Contagem de Palavras Neutras Por Classe", "Contagem de Palavras Neutras", False, False)
boxplot (cont_pos_n1, cont_pos_0, cont_pos_1, "Contagem de Palavras Positivas Por Classe", "Contagem de Palavras Positivas", False, False)


# -- CONTAGEM DE ENTIDADES "PER" POR CLASSE -- #
def contagem_entidade_PER_por_classe():
    cont_per_menos1 = []
    cont_per_0 = []
    cont_per_1 = []
    for i, doc in enumerate (train_df["doc"]):
        classe = train_df["classe"].iloc[i]
        num_per = sum(1 for ent in doc.ents if ent.label_ == "PER")

        if classe == -1:
            cont_per_menos1.append(num_per)
        elif classe == 0:
            cont_per_0.append(num_per)
        elif classe == 1:
            cont_per_1.append(num_per)

    return cont_per_menos1, cont_per_0, cont_per_1

cont_per_menos1, cont_per_0, cont_per_1 = contagem_entidade_PER_por_classe()
boxplot(cont_per_menos1, cont_per_0, cont_per_1, 'Contagem de Entidades "PER" Por Classe', 'Contagens de "PER"', False, False)


# -- CONTAGEM DE ADJETIVOS E ADVÉRBIOS POR CLASSE -- #
def contagem_adj_adv_por_classe():
    cont_adj_adv_menos1 = []
    cont_adj_adv_0 = []
    cont_adj_adv_1 = []

    for i, doc in enumerate(train_df["doc"]):
        classe = train_df["classe"].iloc[i]

        num_adj_adv = sum(1 for token in doc if token.pos_ in ["ADJ", "ADV"])

        if classe == -1:
            cont_adj_adv_menos1.append(num_adj_adv)

        elif classe == 0:
            cont_adj_adv_0.append(num_adj_adv)

        elif classe == 1:
            cont_adj_adv_1.append(num_adj_adv)

    return cont_adj_adv_menos1, cont_adj_adv_0, cont_adj_adv_1


cont_adj_adv_menos1, cont_adj_adv_0, cont_adj_adv_1 = contagem_adj_adv_por_classe()
boxplot(cont_adj_adv_menos1, cont_adj_adv_0, cont_adj_adv_1, 'Contagem de Adjetivos e Advérbios nas frases', 'Nº de adjetivos + advérbios', True, False)


# -- CONTAGEM DE LETRAS MAIÚSCULAS APÓS ASPAS POR CLASSE -- #
def contagem_maiusculas_por_classe():
    cont_maiuscula_menos1 = []
    cont_maiuscula_0 = []
    cont_maiuscula_1 = []

    for i in range(len(train_df["doc"])):
        classe = train_df["classe"].iloc[i]
        
        cont_maiuscula = letras_maiusculas(train_df["sentences"].iloc[i])

        if classe == -1:
            cont_maiuscula_menos1.append(cont_maiuscula)

        elif classe == 0:
            cont_maiuscula_0.append(cont_maiuscula)

        elif classe == 1:
            cont_maiuscula_1.append(cont_maiuscula)

    return cont_maiuscula_menos1, cont_maiuscula_0, cont_maiuscula_1

cont_maiuscula_menos1, cont_maiuscula_0, cont_maiuscula_1 = contagem_maiusculas_por_classe()
boxplot(cont_maiuscula_menos1, cont_maiuscula_0, cont_maiuscula_1, 'Contagem de Maiúsculas nas frases', 'Nº de maiúsculas', True, False)


# Soma elemento a elemento (adv + adj + maiúsculas)
comb_menos1 = [a + b for a, b in zip(cont_adj_adv_menos1, cont_maiuscula_menos1)]
comb_0 = [a + b for a, b in zip(cont_adj_adv_0, cont_maiuscula_0)]
comb_1 = [a + b for a, b in zip(cont_adj_adv_1, cont_maiuscula_1)]

boxplot(comb_menos1, comb_0, comb_1, 'Adjetivos + Advérbios + Maiúsculas', 'Nº total (Adjetivos+Advérbios+Maiúsculas)', True, False)


# -- CONTAGEM DE PALAVRAS NAS FRASES POR CLASSES -- #
def contagem_dePalavras_nasFrases_por_classe():
    cont_palavras_menos1 = []
    cont_palavras_0 = []
    cont_palavras_1 = []

    for i, sentence in enumerate(train_df["sentences"]):
        classe = train_df["classe"].iloc[i]
        
        # Conta todas as palavras na frase
        cont_palavras = len(sentence.split())

        if classe == -1:
            cont_palavras_menos1.append(cont_palavras)

        elif classe == 0:
            cont_palavras_0.append(cont_palavras)

        elif classe == 1:
            cont_palavras_1.append(cont_palavras)

    return cont_palavras_menos1, cont_palavras_0, cont_palavras_1


cont_menos1, cont_0, cont_1 = contagem_dePalavras_nasFrases_por_classe()
boxplot(cont_menos1, cont_0, cont_1, 'Contagem de Palavras nas frases', 'Nº de palavras', True, False)


# -- DEFINE A POLARIDADE DA FRASE -- #
def polaridade_frase (linha):
    negatividade, neutralidade, positividade, _, _, _ = contagem_por_polaridade (sentimentos, linha)
    valores = [negatividade, neutralidade, positividade]
    valor_maximo = max (valores)
    indice_maximo = valores.index (valor_maximo)
    return indice_maximo - 1, valores


# -- POLARIDADE DO CONJUNTO DE FRASES POR CLASSE -- #
def polaridade_conjunto (train_df, classe):
    positivas = 0
    neutras = 0
    negativas = 0
    for i, sentence in enumerate (train_df["sentences"]):
        if train_df["classe"].iloc[i] == classe:
            polaridade = polaridade_frase (sentence)[0]
            if polaridade == -1:
                negativas += 1
            elif polaridade == 0:
                neutras += 1
            else:
                positivas += 1
        
    print ("CLASSE: ", classe, ";Negativas: ", negativas, "; Neutras: ", neutras, "Positivas: ",positivas)
     
polaridade_conjunto(train_df, -1)
polaridade_conjunto(train_df, 0)
polaridade_conjunto(train_df, 1)


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


# -- NÚMERO DE ENTIDADES NUMA FRASE -- #
def numero_entidades(linha):
    num = test_df["doc"].iloc[linha]
    contador = sum(1 for ent in num if ent.label_ == "PER")
    return contador

# -- NÚMERO DE PALAVRAS NUMA FRASE -- #
def num_palavras(linha):
    frase = test_df.iloc[linha]["sentences"]
    return len(frase.split(" "))


# DETEÇÃO DE VIÉS
def detetar_vies(verbos_vies, linha):
    num_condicoes_verificadas = 0

    verbos_vies = set(verbos_vies)

    adverbios_vies = [t[0] for t in adverbios_frequentes_por_classe.get (1)]
    adjetivos_vies = [t[0] for t in adjetivos_frequentes_por_classe.get (1)]
    palavras_vies = [t[0] for t in palavras_frequentes_por_classe.get (1)]
    #bigrams =  [t[0] for t in bigramas_frequentes_por_classe.get (1)]
    
    
    doc_linha = test_df.iloc[linha]["doc"]

    #bigramas_linha = train_df.iloc[linha]["bigrams"]

    
    if (num_adj_adv(linha) + letras_maiusculas(test_df["sentences"].iloc[linha])) >= 4 and num_palavras(linha) > 18:
        num_condicoes_verificadas += 0.5

    if contagem_aspas(test_df.iloc[linha]["sentences"]) == 0:
        num_condicoes_verificadas += 2

    if frequencias_negativas_frase (linha) > 0.075:
        num_condicoes_verificadas += 0.5

    if any(token.lemma_.lower() in verbos_vies for token in doc_linha):
        num_condicoes_verificadas += 1

    if any(token.text.lower() in adverbios_vies for token in doc_linha):
        num_condicoes_verificadas += 0.5

    if any(token.text.lower() in adjetivos_vies for token in doc_linha):
        num_condicoes_verificadas += 2

    if any(token.text.lower() in palavras_vies for token in doc_linha):
        num_condicoes_verificadas += 1

    #if any(gram in bigrams for gram in bigramas_linha):
    #    num_condicoes_verificadas += 1

    # Verifica palavras negativas exclusivas de viés 
    if any(token.text.lower() in so_vies_neg for token in doc_linha):
        num_condicoes_verificadas += 0.5

    # Verifica palavras positivas exclusivas de viés
    if any(token.text.lower() in so_vies_pos for token in doc_linha):
        num_condicoes_verificadas += 0.5

    # Verifica palavras neutras exclusivas de viés 
    if any(token.text.lower() in so_vies_neu for token in doc_linha):
        num_condicoes_verificadas += 0.5

    return num_condicoes_verificadas


# DETEÇÃO DE FACTO
def detetar_factos(verbos_factos, linha):
    num_condicoes_verificadas = 0

    verbos_factos = set(verbos_factos)

    adverbios_factos = [t[0] for t in adverbios_frequentes_por_classe.get (0)]
    adjetivos_factos = [t[0] for t in adjetivos_frequentes_por_classe.get (0)]
    palavras_factos = [t[0] for t in palavras_frequentes_por_classe.get (0)]
    #bigrams =  [t[0] for t in bigramas_frequentes_por_classe.get (0)]
    
    doc_linha = test_df.iloc[linha]["doc"]

    #bigramas_linha = train_df.iloc[linha]["bigrams"]

    #if num_adj_adv(linha) == 1:
    #    num_condicoes_verificadas += 1

    #if letras_maiusculas(test_df["sentences"].iloc[linha]) == 1:
    #    num_condicoes_verificadas += 1

    if contagem_aspas(test_df.iloc[linha]["sentences"]) == 0:
        num_condicoes_verificadas += 2

    if any(token.lemma_.lower() in verbos_factos for token in doc_linha):
        num_condicoes_verificadas += 1

    if any(token.text.lower() in adverbios_factos for token in doc_linha):
        num_condicoes_verificadas += 0.5

    if any(token.text.lower() in adjetivos_factos for token in doc_linha):
        num_condicoes_verificadas += 2
    
    if any(token.text.lower() in palavras_factos for token in doc_linha):
        num_condicoes_verificadas += 1

    #if any(gram in bigrams for gram in bigramas_linha):
    #    num_condicoes_verificadas += 1
    
    return num_condicoes_verificadas


# DETEÇÃO DE CITAÇÕES
def detetar_citacoes(verbos_citacoes, linha):
    num_condicoes_verificadas = 0

    verbos_citacoes = set(verbos_citacoes)

    adverbios_citacoes = [t[0] for t in adverbios_frequentes_por_classe.get (-1)]
    adjetivos_citacoes = [t[0] for t in adjetivos_frequentes_por_classe.get (-1)]
    palavras_citacoes = [t[0] for t in palavras_frequentes_por_classe.get (-1)]
    #bigrams =  [t[0] for t in bigramas_frequentes_por_classe.get (-1)]

    doc_linha = test_df.iloc[linha]["doc"]
    #bigramas_linha = test_df.iloc[linha]["bigrams"]

    #if num_adj_adv(linha) == 1:
        #num_condicoes_verificadas += 1
    
    #if letras_maiusculas(test_df["sentences"].iloc[linha]) == 1:
        #num_condicoes_verificadas += 1
    
    if contagem_aspas(test_df.iloc[linha]["sentences"]) == 1:
        num_condicoes_verificadas += 2
        
    if any(token.text.lower() in verbos_citacoes for token in doc_linha):
        num_condicoes_verificadas += 1

    if any(token.text.lower() in adverbios_citacoes for token in doc_linha):
        num_condicoes_verificadas += 0.5

    if any(token.text.lower() in adjetivos_citacoes for token in doc_linha):
        num_condicoes_verificadas += 2
    
    if any(token.text.lower() in palavras_citacoes for token in doc_linha):
        num_condicoes_verificadas += 1

    #if any(gram in bigrams for gram in bigramas_linha):
     #   num_condicoes_verificadas += 1

    return num_condicoes_verificadas


verbos_citacoes = [v for v in verbos_sinonimos(verbos_frequentes_por_classe, -1)]
verbos_factos = [v for v in verbos_sinonimos(verbos_frequentes_por_classe, 0)]
verbos_vies = [v for v in verbos_sinonimos(verbos_frequentes_por_classe, 1)]

nova_classificacao = []
for i in range(len(test_df["sentences"])):
    num_factos = detetar_factos (verbos_factos, i)
    num_citacoes = detetar_citacoes (verbos_citacoes, i)
    num_vies = detetar_vies (verbos_vies, i)

    valores = [num_citacoes, num_factos, num_vies]
    valor_maximo = max (valores)
    indice_maximo = valores.index (valor_maximo)
    nova_classificacao.append (indice_maximo - 1)
    #print(i, nova_classificacao[-1], train_df.iloc[i]["classe"])

print(classification_report(test_df["classe"], nova_classificacao))

# --- Gerar a matriz de confusão ---
cm = confusion_matrix(test_df["classe"], nova_classificacao)

# --- Mostrar a figura da matriz de confusão ---
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=[-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw={'fontsize':16})  
plt.title("Matriz de Confusão", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()