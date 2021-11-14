#!/usr/bin/env python
# coding=utf-8

"""AcquisitorAndCleaner engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['AcquisitorAndCleaner']


logger = get_logger('acquisitor_and_cleaner')


class AcquisitorAndCleaner(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(AcquisitorAndCleaner, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        from marvin_python_toolbox.common.data import MarvinData
        import pandas as pd
        import nltk
        from nltk.corpus import stopwords

        products_data = pd.read_csv(MarvinData.download_file("https://s3.amazonaws.com/automl-example/produtos.csv"), delimiter=';', encoding='utf-8')
        # concatenando as colunas nome e descricao
        products_data['informacao'] = products_data['nome'] + products_data['descricao']
        # excluindo linhas com valor de informacao ou categoria NaN
        products_data.dropna(subset=['informacao', 'categoria'], inplace=True)
        products_data.drop(columns=['nome', 'descricao'], inplace=True)

        stop_words = set(stopwords.words("portuguese"))
        # transforma a string em caixa baixa e remove stopwords
        products_data['sem_stopwords'] = products_data['informacao'].str.lower().apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

        tokenizer = nltk.RegexpTokenizer(r"\w+")
        products_data['tokens'] = products_data['sem_stopwords'].apply(tokenizer.tokenize)  # aplica o regex tokenizer
        products_data.drop(columns=['sem_stopwords', 'informacao'], inplace=True)  # Exclui as colunas antigas

        products_data["strings"] = products_data["tokens"].str.join(" ")  # reunindo cada elemento da lista
        products_data.head()
        self.marvin_initial_dataset = products_data

