import pandas as pd
import numpy as np


class NaiveBayes(object):

    def __init__(self, freq_table_out):
        """Initializing Bayes classificator with a list of conditional probabilities tables"""
        self.freq_table = freq_table_out

    def train(self, df):
        """Training Bayes classificator, making a list of tables with conditional
        probabilities for each feature"""
        for name in df.columns[:-1]:
            self.freq_table[name] = pd.DataFrame(columns=df['label'].unique(),
                                                 index=df[name].unique())
            for i in self.freq_table[name].index:
                for j in self.freq_table[name].columns:
                    mask = df[name][(df[name] == i) & (df['label'] == j)]
                    self.freq_table[name][j].loc[i] = mask.size / len(df.index)

    def classify_one(self, df, test):
        """Using conditional probabilities tables to analise current test item"""
        probabilities = {}
        for labels in df['label'].unique():
            probabilities[labels] = 1
            for name in df.columns[:-1]:
                probabilities[labels] *= self.freq_table[name][labels].loc[test[name]]
        result = max(probabilities.values())
        for key in probabilities.keys():
            if probabilities.get(key) == result:
                return key

    def classification(self, df, test_table):
        """"Using conditional probabilities tables to analise testing dataset"""
        keys = []
        for i in test_table.index:
            keys.append(self.classify_one(df, test_table.loc[i]))
        return keys


def reader(name):
    """Reading DataFrame, drop rows with missing values"""
    df = pd.read_csv(name)
    column_list = []
    for i in range(len(df.columns) - 1):
        column_list.append('x' + str(i))
    column_list.append('label')
    df.columns = column_list
    df.replace(' ?', np.nan, inplace=True)
    df = df.dropna(axis=0, how='any')
    return df


def check(df, keys):
    """Calculating accuracy of labeling testing dataset"""
    accuracy = 0.0
    for i in range(len(keys)):
        if df['label'].iloc[i] == keys[i]:
            accuracy += 1
    return accuracy / len(keys)


freq_table = {}
classif = NaiveBayes(freq_table)

'''Large Soybean Dataset classification'''
#df = reader('soybean.csv')

'''Golf dataset classification'''
df = reader('weather.nominal.csv')

classif.train(df[:-len(df.index) // 4])
test = df[-len(df.index) // 4:]

'''Calculating accuracy for current testing dataset labeling'''
keys = classif.classification(df, test)
print('KEYS==', keys)
print('TOTAL AMOUNT OF LABELS ==', len(df['label'].unique()))
print('SIZE OF TEST DATASET == ', len(keys))
print('ACCURACY ==', check(test, keys))

