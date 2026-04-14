from langchain_ollama import OllamaLLM
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv
import os

df = pd.read_csv("factnews_dataset.csv")
print(df.head())

# Drop de colunas não necessárias
X = df["sentences"]

# y é a coluna de classes (target)
y = df.iloc[:, -1]

# Divisão dos dados em treino, validação e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
print(len(y_test))

# Para os índices começarem a 0
X_test = X_test.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

y_pred = [1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 0, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 0, 0, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, 1, 1, 0, 0, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 0, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 0, -1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, -1, 1, 1, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 0, 0, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
invalid_indices = [77, 109, 132, 180, 251, 328, 366, 487, 508, 749, 762, 799, 952, 973, 976, 989, 1039, 1164, 1178, 1228, 1286, 1350, 1361, 1373, 1418, 1509, 1538, 1541, 1562, 1595, 1630, 1634, 1698, 1712, 1800, 1848]

for i in (invalid_indices):
  print(f"{i}: {X_test[i]}")

# Remover índices inválidos
y_test_filtered = y_test.drop(index = invalid_indices)
X_test_filtered = X_test.drop(index = invalid_indices)

## -- FUNÇÃO DE AVALIAÇÃO DO LLM -- ##
def llm_output(y_test, y_pred):
  print("-------- MODELO LLM --------")
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("\nRelatório de classificação:")
  print(classification_report(y_test, y_pred, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
  print()

  # MATRIZ DE CONFUSÃO
  cm = confusion_matrix(y_test, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
  disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})
  plt.title("Matriz de Confusão", fontsize = 22)
  plt.xlabel("Classe Prevista", fontsize = 14)
  plt.ylabel("Classe Verdadeira", fontsize = 14)
  plt.show()

df = pd.read_csv("frases_representativas.csv", on_bad_lines = 'skip')

# Abrir o CSV com csv.reader, respeitando aspas
with open("frases_representativas.csv", newline = '', encoding = 'utf-8') as f:
    reader = csv.reader(f, quotechar = '"', skipinitialspace = True)
    citacoes = []
    factos = []
    vieses = []
    for row in reader:
        if len(row) >= 3:
            citacoes.append(row[0])
            factos.append(row[1])
            vieses.append(row[2])

# Ignorar o primeiro item que é apenas o nome da coluna
citacoes = citacoes[1:]
factos = factos[1:]
vies = vieses[1:]

# Função para imprimir de forma limpa
def imprimir_classe(frases, nome_classe):
    print(f"\n### {nome_classe.upper()} ###\n")
    for i, frase in enumerate(frases):
        print(f"{i}: {frase}\n")  # sem numeração

# Imprimir todas as frases
imprimir_classe(citacoes, "citacao")
imprimir_classe(factos, "facto")
imprimir_classe(vieses, "vies")

## -- LLM (FEW-SHOT-PROMPTING) -- ##
def llm_model_fewshot(X_t, y_t, citacoes, factos, vies):

    # Prompt Few-shot com exemplos e descrição das classes
    prompt = [
    # Introdução aos exemplos
    ('user', 'A seguir apresento **EXEMPLOS** de frases já classificadas como "-1" (citação), "0" (facto), e "1" (viés). Estes são apenas exemplos:'),
    # CITACOES
    ('user', 'Se não passou, vamos colocar as alternativas seguintes", disse.'),
    ('assistant', '-1'),
    ('user', 'Ele tentou criar um ambiente de rejeição ao PT que não existe", disse o presidente do partido.'),
    ('assistant', '-1'),
    ('user', 'Não julgo prudente para o próprio PT a concentração de poder", disse.'),
    ('assistant', '-1'),
    ('user', 'E por isso é que ele ficou apavorado e tentou se explicar o mais rápido possível", afirmou o ex-presidente.'),
    ('assistant', '-1'),
    ('user', 'Infelizmente, por outros motivos que desconhecemos, os recursos não foram aplicados na conservação e manutenção dessas estradas, que ficaram sob a responsabilidade dos governos estaduais", disse Luís Munhoz, coordenador geral de construção do Dnit.'),
    ('assistant', '-1'),
    ('user', 'Há regras boas e ruins, mas não ter nenhuma regra e gerar uma incerteza enorme nos gastos públicos é muito ruim para a economia", disse Arida em entrevista ao Estadão.'),
    ('assistant', '-1'),

    # FACTOS
    ('user', 'O valor representa 10,45% do total de óbitos por Covid-19 no mundo, que corresponde a um total de mais de 6,5 milhões de mortes, segundo a mesma plataforma Worldometers.'),
    ('assistant', '0'),
    ('user', 'Ele teve problemas com sua hepatite aguda, mas não é grave, segundo médico'),
    ('assistant', '0'),
    ('user', 'Segundo Nogueira, o salário do governador é 95% do que recebe um deputado federal, mas não tem vinculação direta ao salários dos secretários de Estado.'),
    ('assistant', '0'),
    ('user', 'Também nesta sexta, às 19h, a Folha realiza a segunda live sobre a pesquisa Datafolha do segundo turno das eleições.'),
    ('assistant', '0'),
    ('user', 'Segundo informações da assessoria do apresentador, ele não poderia comparecer ao Deic na quarta-feira para reconhecer o suspeito devido às gravações de seu programa no Rio de Janeiro.'),
    ('assistant', '0'),
    ('user', 'Um dos centros da investigação é o município de Igarapé Grande, a 300 km de São Luís, onde foram detectados indícios de desvios de recursos e fraudes em contratos firmados pela cidade, segundo a PF.Em 2020, o município informou ter realizado mais de 12,7 mil radiografias de dedo no sistema público de saúde -a população da cidade é de 11,5 mil habitantes.'),
    ('assistant', '0'),

    # VIÉS
    ('user', 'Desde que Bolsonaro teve um desempenho superior ao que os entrevistados diziam no primeiro turno, seus apoiadores iniciaram uma campanha para criminalizar os institutos de pesquisa por supostos erros, aprovando urgência de um projeto nesse sentido na Câmara -embora os levantamentos não se prestem a errar ou acertar resultados.'),
    ('assistant', '1'),
    ('user', 'O presidente da Câmara, Arlindo Chinaglia (PT-SP), evitou criticar o mérito da proposta, mas lembrou que as matérias sempre podem sofrer alterações e disse que primeiro a Casa vai terminar de votar os projetos infraconstitucionais da reforma política.'),
    ('assistant', '1'),
    ('user', 'Lula ressaltou que, com o Programa Fome Zero, conseguiu atingir o primeiro ponto das Metas do Milênio - erradicar a fome -, com dez anos de antecedência, reduzindo em mais da metade a pobreza extrema.'),
    ('assistant', '1'),
    ('user', 'Como mostrou a Folha, o Ministério da Defesa adotou o silêncio após o primeiro turno das eleições e não responde a pedidos de informação sobre a fiscalização realizada no sistema eletrônico de votação em 2 de outubro.'),
    ('assistant', '1'),
    ('user', 'Sem citar nominalmente o adversário, Alckmin criticou de novo Lula ao comentar especulações de que o petista, convicto na vitória no primeiro turno, já estaria fazendo planos sobre sua nova equipe ministerial.'),
    ('assistant', '1'),
    ('user', 'A afirmação do comitê não tem nenhum efeito jurídico, mas tem peso político para fortalecer o discurso de Lula de perseguição política às vésperas da disputa eleitoral na qual lidera as pesquisas de intenção de voto.'),
    ('assistant', '1')]

    # Definir LLM
    llm = OllamaLLM(model = "llama3.2:3b", temperature = 0.1)

    # Se existir um CSV anterior, carregar e continuar a partir de onde parou
    if os.path.exists("few_shot.csv"):
        df_antigo = pd.read_csv("few_shot.csv")
        y_pred = df_antigo["pred"].tolist()
        n = df_antigo["n_next"].iloc[-1] + 1 # Último valor guardado
        print(f"A retomar do índice {n} (já existem {len(y_pred)} previsões)")
    else:
        y_pred = []
        n = 0
        print("Nenhum resultado anterior encontrado. A iniciar do zero.")

    # --- Loop principal --- #
    for i in range(n, len(X_t)):
        frase = X_t.iloc[i]
        tries = []
        # Cria uma cópia do prompt base e adiciona a frase de teste separada
        prompt_final = prompt.copy()

        # Prompt que junta a frase a classificar
        prompt_final.append(("user",
                            f"""As frases podem estar classificadas com um dos seguintes números:
                            -1 (CITAÇÃO): Declarações diretas, geralmente com pelo menos uma aspa.
                            0 (FACTO): Informação apresentada de forma imparcial, centrada em factos objetivos.
                            1 (VIÉS): Informação tendenciosa, apresentando um ponto de vista parcial ou influenciado.
                            
                            Classifica a frase seguinte dizendo apenas o número correspondente: {frase}"""
                            ))
        
        # 3 tentativas para majority voting
        for _ in range(3):
            try:
                resposta = llm.invoke(prompt_final).strip()
                print(resposta) # Pega só o primeiro caractere da resposta
            except:
                resposta = "0" # Caso dê erro, atribui 0 (classe maioritária)
                print("NÃO CONSEGUIU RESPONDER")
            
            if " -1" in resposta or "-1 " in resposta or " -1 " in resposta or "-1" in resposta:
                resposta = "-1"
            elif " 1"  in resposta or " 1 " in resposta or "1 " in resposta or "1" in resposta:
                resposta = "1"
            elif " 0 " in resposta or " 0" in resposta or "0 " in resposta or "0" in resposta:
                resposta = "0"
            else:
                print("NÃO RESPONDEU COM 1 0 -1")
                resposta = resposta # Mantém a resposta original para validação abaixo

            tries.append(resposta)

        final = Counter(tries).most_common(1)[0][0]
        y_pred.append(int(final))

        print(f"{i}/{len(X_t)} | {tries} → {final} | Real = {y_t.iloc[i]}")
        
        df_resultados = pd.DataFrame({
        "n_next": list(range(len(y_pred))),
        "pred": y_pred
        })

        df_resultados.to_csv("few_shot.csv", index = False, encoding = "utf-8")
     
    print("Terminado.")

    return y_pred

#y_pred = llm_model_fewshot(X_test_filtered, y_test_filtered, citacoes, factos, vies)

df_antigo = pd.read_csv("few_shot.csv")
y_pred = df_antigo["pred"].tolist()
llm_output(y_test_filtered[:len(y_pred)], y_pred)