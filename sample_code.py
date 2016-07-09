__author__ = 'yuxinsun'

from scipy.io import loadmat, savemat
import numpy as np
from sklearn.metrics import accuracy_score
from Features.process_data import save_pickle, read_pickle
from LPBoost.lpboost import lpboost
from Features.generate_features import feature_generation, create_word_list

# a = ['CASSLQDTQYFGPG CASGDQGAYAEQFFGPG CASSRGEYNSPLYFGEG CGAREGGQNTLYFGAG CASSLLNERLFFGHG CASSGQGETLYFGSG CASSLLGEDTQYFGPG',
#      'CASSLGGPDTQYFGPG CASSRTGRDEQYFGPG CARNVTPPKSYAVFFGKG CAWSLQGYNSPLYFGEG CASRQGDTNERLFFGHG CASSLDGGPEQYFGPG CASSDARGDQDTQYFGPG CASSPRDNYNSPLYFGEG']
# save_pickle('', 'toy_data', a)

print('This step shows how to generate feature matrix using CDR3 sequences.'
      '\nGenerate string features using toy_data.cpickle')
data = read_pickle('', 'toy_data')
f = feature_generation(feature_type='fisher')
data_norm = f.process(data)

print('\nThis step shows how to perform LPBoost using a toy feature matrix containing 33 samples.'
      '\nPerform LPBoost using toy_feature.mat')
data = loadmat('toy_feature.mat', variable_names=['data_norm'])
data = np.hstack((data['data_norm'], -data['data_norm']))
label = np.hstack((-np.ones(15), np.ones(18)))
lp = lpboost(nu=0.5, threshold=10**-3, verbose=1)

# for convenience data is not split into training and test set
lp.fit(data, label)  # fit LPBoost model
label_pred = lp.predict(data)  # predict labels using LPBoost classifier
acc = accuracy_score(label, label_pred)  # compute training accuracy

print('\nTraining accuracy is %.2f' % acc)
