__author__ = 'yuxinsun'

from scipy.io import loadmat, savemat
import numpy as np
from Features.process_data import save_pickle, read_pickle
from LPBoost.lpboost import lpboost
from Features.generate_features import feature_generation, create_word_list

# a = ['CASSLQDTQYFGPG CASGDQGAYAEQFFGPG CASSRGEYNSPLYFGEG CGAREGGQNTLYFGAG CASSLLNERLFFGHG CASSGQGETLYFGSG CASSLLGEDTQYFGPG',
#      'CASSLGGPDTQYFGPG CASSRTGRDEQYFGPG CARNVTPPKSYAVFFGKG CAWSLQGYNSPLYFGEG CASRQGDTNERLFFGHG CASSLDGGPEQYFGPG CASSDARGDQDTQYFGPG CASSPRDNYNSPLYFGEG']
# save_pickle('', 'toy_data', a)

print('Generate string features using toy_data.cpickle')
data = read_pickle('', 'toy_data')
f = feature_generation(feature_type='fisher')
data_norm = f.process(data)

print('\nPerform LPBoost using hh1.mat')
data = loadmat('hh1.mat', variable_names=['data_norm'])
data = np.hstack((data['data_norm'], -data['data_norm']))
label = np.hstack((-np.ones(15), np.ones(18)))
lp = lpboost(nu=0.5, threshold=10**-3, verbose=1)

a = lp.fit(data, label)  # fit LPBoost model
b = lp.fit_transform(data, label)  # fit and transform
c = lp.transform(data)  # transform data using fitted LPBoost model
d = lp.predict(data)  # predict labels using LPBoost classifier