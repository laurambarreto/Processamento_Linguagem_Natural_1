# Introdução:
Este trabalho foi realizado no âmbito da cadeira de Processamento de Linguagem Natural (PLN). O dataset encontra-se no ficheiro "factnews_dataset.csv" e o objetivo é classificar as suas frases como citações (-1), facto (0), viés (1) através de regras por nós criadas.

# Ficheiros 
    1) Trabalho.py
    2) Grid_Search.py
    3) factnews_dataset.csv
    4) SentiLex.csv

1): Ficheiro principal para execução do nosso classificador com regras
2): Grid Search usado para testar os melhores pesos a utilizar nas funções de deteção de citação, facto e viés
3): Dataset principal com as frases das três classes
4): Dataset com as palavras e as suas polaridades, usado para perceção de sentimento

# Dependências:
    -> spacy
    -> nltk
    -> pandas
    -> scikit-learn
    -> matplotlib
    -> numpy

# Como executar os ficheiros:
-> python <nome_do_ficheiro>.py