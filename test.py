from pandas import read_csv, DataFrame
from numpy.random import seed
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Dense, Merge
from keras.layers.normalization import BatchNormalization
import data
 
#df = read_csv("credit_count.txt")
#Y = df[df.CARDHLDR == 1].DEFAULTS
#X1 = scale(df[df.CARDHLDR == 1][["MAJORDRG", "MINORDRG", "OWNRENT", "SELFEMPL"]])
#X2 = scale(df[df.CARDHLDR == 1][["AGE", "ACADMOS", "ADEPCNT", "INCPER", "EXP_INC", "INCOME"]])

X1, Y, a, b = data.getVideoData()
X2, Y, a, b = data.getVideoData()
 
branch1 = Sequential()
branch1.add(Dense(X1.shape[1], input_shape = (X1.shape[1],), init = 'normal', activation = 'relu'))
branch1.add(BatchNormalization())

branch1.summary()
 
branch2 = Sequential()
branch2.add(Dense(X2.shape[1], input_shape =  (X2.shape[1],), init = 'normal', activation = 'relu'))
branch2.add(BatchNormalization())
branch2.add(Dense(X2.shape[1], init = 'normal', activation = 'relu', W_constraint = maxnorm(5)))
branch2.add(BatchNormalization())
branch2.add(Dense(X2.shape[1], init = 'normal', activation = 'relu', W_constraint = maxnorm(5)))
branch2.add(BatchNormalization())

branch2.summary()
 
model = Sequential()
model.add(Merge([branch1, branch2], mode = 'concat'))
model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
seed(2017)
model.fit([X1, X2], Y.values, batch_size = 2000, nb_epoch = 100, verbose = 1)
