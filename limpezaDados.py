# summarize the number of unique values for each column using numpy
from numpy import loadtxt
from numpy import unique
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from numpy import arange
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from pandas import read_csv
from pandas import set_option
import numpy
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Binarizer

# load the dataset
'''
var = ['Proveniência', 'Local_SU', 'Idade','Genero','Interna_Dias', 'GrupoDiagn', 'SIRS', 'Glicose', 'Sodio', 'Ureia',
       'Creatinina', 'PCR', 'pH', 'Ca_ionizado', 'pCO2', 'pO2', 'HCO3','Antidislipidemicos', 'Antipsicóticos', 'Antidepressores',
       'Anti-hipertensores', 'Anti-histaminicos', 'Ansioliticos',
       'Analgésicos ', 'Anticoagulantes ', 'Corticosteroides',
       'Antiespasmódicos', 'Antiparkinsónico', 'Cardiotonico', 'Antiacido ',
       'Geniturinario', 'Obito', 'Alcoolico', 'ResultDelirium']
     
'''
data = pd.read_csv('./dadosDeliriumLimpos.csv', na_values='NA')
# summarize the number of unique values in each column

print(data.head(20))
print(data.shape)

dups = data.duplicated()
# report if there are any duplicates
print("Duplicados:\n",dups.any())
print(data[dups])
# list all duplicate data

# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
pyplot.show()



#Mostra os dados que são únicos 
for x in data.nunique().items():
    #printing unique values
    print(x)


print(data.head())
print("Shape: ",data.shape)
print(data.iloc[0].isnull().any())

#Verificação de linhas com valores nulos 
for i in range(0, data.shape[1] - 1):
 	miss_val = data.iloc[i].isnull().sum()
 	perc = miss_val / data.shape[0] * 100
 	print('> %d, Missing: %d (%.1f%%)' % (i, miss_val, perc))


# split data into inputs and outputs
dados = data.values
X = dados[:, :-1].astype(str)
y = dados[:, -1].astype(str)
# summarize
print('Input', X.shape)
print('Output', y.shape)

set_option('display.width', 100)
set_option('precision', 3)
correlations = data.corr(method='pearson')
print('Correlações\n',correlations)

# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)

scatter_matrix(data)
pyplot.show()

# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()





skew = data.skew()
print('Skews\n',skew)

data.plot(kind='density', subplots=True, layout=(7,5), sharex=False)
pyplot.show()

# define thresholds to check
thresholds = arange(0.0, 0.60, 0.01)
# apply transform with each threshold
results = list()
input_features = data.keys().tolist()
input_features = input_features[:-1]
print(input_features)
print("O tamanho: \n", len(input_features))
for t in thresholds:
	# define the transform
	transform = VarianceThreshold(threshold=t)
	# transform the input data
	print(X.shape)
	X_sel = transform.fit_transform(X)
	# determine the number of input features
	n_features = X_sel.shape[1]
	print('>Threshold=%.2f, Features=%d' % (t, n_features))
	# store the result
	print(transform.get_feature_names_out(input_features))
	results.append(n_features)


# plot the threshold vs the number of selected features
pyplot.plot(thresholds, results)
pyplot.show()

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)


# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)

# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]

# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)

# fit the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# evaluate the model
yhat = model.predict(X_test)
print(yhat)

# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


model = LinearDiscriminantAnalysis()
# define the model evaluation procedure
cv = KFold(n_splits=3, shuffle=True, random_state=1)
# evaluate the model
result = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
# report the mean performance
print('Accuracy: %.3f' % result.mean())



# define min max scaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(data)
print('Dados escalados de 0 a 1:\n',scaled)

print(data.shape)
# summarize each variable
print(data.describe())
# histograms of the variables
fig = data.hist(xlabelsize=3, ylabelsize=3)
[x.title.set_size(3) for x in fig.ravel()]
# show the plot
pyplot.show()


dadosNorm = data.values[:, :-1]
print(dadosNorm)
# perform a robust scaler transform of the dataset
trans = MinMaxScaler()
dadosNorm = trans.fit_transform(dadosNorm)


# convert the array back to a dataframe
data = DataFrame(dadosNorm)
# summarize
print(data.describe())

# histograms of the variables
fig = data.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()

# define ordinal encoding
encoder = OrdinalEncoder()
# transform data
result = encoder.fit_transform(data)
print(result)

# one hot encode input variables
onehot_encoder = OneHotEncoder(sparse=False)
X = onehot_encoder.fit_transform(X)
# ordinal encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# summarize the transformed data
print('Input', X.shape)
print(X[:5, :])




