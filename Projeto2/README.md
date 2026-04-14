## Introdução 
Foram utilizadas tanto as nossas regras da primeira meta, como os métodos IF-IDF, Glove e Bert aplicados nos modelos: KNN, Regressão logística, Naive-Bayes, Random Forest e Redes Neuronais.

## Ficheiros
- bal.py: Dados das regras balanceados normalizados 
- bal_corr.py: Dados das regras balanceados e normalizados e só com colunas mais correlacionadas
- nbal.py: Dados das regras não balanceados e normalizados 
- nbal_corr.py: Dados das regras não balanceados e normalizados e só com colunas mais correlacionadas
- IDF.py : Utiliza o DF/IDF para representação de texto, através das frequências de cada palavra nas frases
- GloVe.py : Utiliza o glove que representa cada frase com a média dos vetores das palavras dessa frase
- Bert.jypht : Utiliza o Bert para representação linguística, só foi possível correr esta abordagem no colab devido a probelmas de incompatibilidades de bibliotecas
- SentiLex.csv
- glove_s100.txt
- factnews_dataset.csv

## Executar os ficheiros
python [nome do ficheiro].py

## Dependências
scikit-learn
spacy
nltk
gensim
transformers
torch
optuna
pandas
numpy
scipy
matplotlib
seaborn