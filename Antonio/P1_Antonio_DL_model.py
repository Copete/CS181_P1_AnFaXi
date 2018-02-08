
# coding: utf-8

# # 0. Sample code

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[6]:


"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("../train.csv.gz")
df_test  = pd.read_csv("../test.csv.gz")


# In[4]:


df_train.head()


# In[5]:


df_test.head()


# In[7]:


#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)


# In[8]:


#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_all.head()


# In[9]:


"""
Example Feature Engineering

this calculates the length of each smile string and adds a feature column with those lengths
Note: this is NOT a good feature and will result in a lower score!
"""
#smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
#df_all['smiles_len'] = pd.DataFrame(smiles_len)


# In[10]:


#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print("Train features:", X_train.shape)
print("Train gap:", Y_train.shape)
print("Test features:", X_test.shape)


# In[10]:


LR = LinearRegression()
LR.fit(X_train, Y_train)
LR_pred = LR.predict(X_test)


# In[11]:


RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)


# In[12]:


def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


# In[16]:


# Skip writing sample files
#write_to_file("sample1.csv", LR_pred)
#write_to_file("sample2.csv", RF_pred)


# ## 1. Linear Regression and Random Forest exploration

# In[17]:


help(LR)


# In[18]:


plt.figure(figsize=(15,10))
plt.bar(range(len(LR.coef_)), LR.coef_)
plt.yscale('symlog')
plt.show()


# In[19]:


# Calculate initial R^2
print(LR.score(X_train, Y_train))
print(RF.score(X_train, Y_train))


# In[20]:


help(RF)


# In[21]:


plt.bar(range(len(RF.feature_importances_)), RF.feature_importances_, 2)
plt.yscale('symlog')
plt.show()


# In[22]:


# Drop predictors that are unimportant in both LR and RF
LR_coef = LR.coef_
RF_imp = RF.feature_importances_
i0 = (LR_coef == 0) & (RF_imp == 0)
X_train1 = X_train[:,i0 == False]
X_test1 = X_test[:,i0 == False]


# In[23]:


# Redo regressions
LR1 = LinearRegression()
LR1.fit(X_train1, Y_train)
RF1 = RandomForestRegressor()
RF1.fit(X_train1, Y_train)


# In[24]:


# Recalculate R^2
print(LR1.score(X_train1, Y_train))
print(RF1.score(X_train1, Y_train))


# In[25]:


plt.bar(range(len(LR1.coef_)), LR1.coef_)
plt.yscale('symlog')
plt.show()


# In[26]:


plt.bar(range(len(RF1.feature_importances_)), RF1.feature_importances_, 2)
plt.yscale('symlog')
plt.show()


# In[27]:


# Test only features with RF importance > 0
RF_imp1 = RF1.feature_importances_
LR_coef1 = LR.coef_
i1 = (RF_imp1 > 0)
X_train2 = X_train1[:,i1]
X_test2 = X_test1[:,i1]


# In[28]:


LR2 = LinearRegression()
LR2.fit(X_train2, Y_train)
RF2 = RandomForestRegressor()
RF2.fit(X_train2, Y_train)
print(LR2.score(X_train2, Y_train))
print(RF2.score(X_train2, Y_train))


# In[29]:


X_train2.shape


# In[30]:


plt.bar(range(len(LR2.coef_)), LR2.coef_)
plt.yscale('symlog')
plt.show()


# In[31]:


plt.bar(range(len(RF2.feature_importances_)), RF2.feature_importances_)
plt.yscale('symlog')
plt.show()


# In[32]:


LR_coef2 = LR2.coef_
RF_imp2 = RF2.feature_importances_


# In[91]:


# Drop the least important RF feature in each iteration
X_train0 = X_train2
nx = np.array([])
R2_LR = np.array([])
R2_RF = np.array([])
for i in range(30):
    LR0 = LinearRegression()
    LR0.fit(X_train0, Y_train)
    RF0 = RandomForestRegressor()
    RF0.fit(X_train0, Y_train)
    print('LR(',i+1,'): ',LR0.score(X_train0, Y_train))
    print('RF(',i+1,'): ',RF0.score(X_train0, Y_train))
    n0 = X_train0.shape[1]
    nx = np.append(nx,n0)
    R2_LR = np.append(R2_LR,LR0.score(X_train0, Y_train))
    R2_RF = np.append(R2_RF,RF0.score(X_train0, Y_train))
    i0 = (range(n0) != RF0.feature_importances_.argmin())
    X_train0 = X_train0[:,i0]


# In[33]:


plt.plot(nx,R2_LR)
plt.plot(nx,R2_RF,c='r')
plt.show()


# ## 2. Linear regression methods with Shrinkage and Cross-Validation:
# ##    Ridge Regression, Lasso, Elastic Net
# ##    Hyperparameter tuning by Grid Search

# In[130]:


#Implement Ridge regression
from sklearn.linear_model import Ridge


# In[115]:


ridge = Ridge(alpha=0.1, normalize=True) #Inialize alpha to 0.1
ridge_coef = ridge.fit(X_train2, Y_train).coef_
ridge.score(X_train2, Y_train)


# In[116]:


plt.bar(range(len(ridge_coef)), ridge_coef)
plt.show()


# In[117]:


#Implement Lasso regression
from sklearn.linear_model import Lasso


# In[121]:


lasso = Lasso(alpha=0.001,normalize=True) #Inialize alpha to 0.1
lasso_coef = lasso.fit(X_train2, Y_train).coef_
lasso.score(X_train2, Y_train)
# Why is Lasso fit not working??? Model not sensitive enough to the data?


# In[122]:


plt.bar(range(len(lasso_coef)), lasso_coef)
plt.show()


# In[127]:


#Implement Cross-validation
from sklearn.model_selection import cross_val_score
LR_cv = LinearRegression()


# In[129]:


#Try 3-, 5- and 10-fold Cross-validation
cv3_results = cross_val_score(LR_cv, X_train2, Y_train, cv=3)
print(cv3_results)
print(np.mean(cv3_results), np.std(cv3_results))
cv5_results = cross_val_score(LR_cv, X_train2, Y_train, cv=5)
print(cv5_results)
print(np.mean(cv5_results), np.std(cv5_results))
cv10_results = cross_val_score(LR_cv, X_train2, Y_train, cv=10)
print(cv10_results)
print(np.mean(cv10_results), np.std(cv10_results))
# 5-fold CV seems OK


# In[132]:


#Implement Grid search cross-validation
from sklearn.model_selection import GridSearchCV
# with Ridge regression
alpha_grid = {'alpha':np.logspace(-4,0,50)}
ridge = Ridge(alpha=0.1, normalize=True)
ridge_cv = GridSearchCV(ridge,alpha_grid,cv=5) #Instantiate Grid search CV regressor
ridge_cv.fit(X_train2,Y_train) #Fit data and tune alpha to optimal value
print('Optimal Ridge parameter:', ridge_cv.best_params_)
print('Best score: ', ridge_cv.best_score_)


# In[143]:


# Create hold-out set (i.e. a test set from given data)
from sklearn.model_selection import train_test_split
X_train3, X_holdout3, Y_train3, Y_holdout3 = train_test_split(X_train2, Y_train, test_size=0.2, random_state=42) #How to set random_state?
print(X_train3.shape, X_holdout3.shape, Y_train3.shape, Y_holdout3.shape)


# In[144]:


# Try ElasticNet regression (combination of Ridge and Lasso)
from sklearn.linear_model import ElasticNet
# Create the hyperparameter grid (L1=1 for Lasso, <1 for Lasso/Ridge combination)
l1_space = np.linspace(0, 1, 30)
l1_grid = {'l1_ratio': l1_space}
# Instantiate the ElasticNet regressor: EN
EN = ElasticNet()
# Setup the GridSearchCV object: EN_cv
EN_cv = GridSearchCV(EN, l1_grid, cv=5)


# In[145]:


# Fit it to the new training data
EN_cv.fit(X_train3, Y_train3)
# Predict on the test set and compute metrics
Y_pred3 = EN_cv.predict(X_holdout3)
r2_EN_cv = EN_cv.score(X_holdout3, Y_holdout3)
from sklearn.metrics import mean_squared_error
mse_EN_cv = mean_squared_error(Y_holdout3, Y_pred3)
print("Tuned ElasticNet l1 ratio: {}".format(EN_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2_EN_cv))
print("Tuned ElasticNet MSE: {}".format(mse_EN_cv))


# In[146]:


# Optimize Ridge alpha on new training data. Try lower range
alpha_grid = {'alpha':np.logspace(-6,-1,100)}
ridge = Ridge(normalize=True) # Initialize Ridge regressor
ridge_cv = GridSearchCV(ridge,alpha_grid,cv=5) #Instantiate GridSearch CV regressor
ridge_cv.fit(X_train3,Y_train3) #Fit data and tune alpha to optimal value
print('Optimal Ridge parameter:', ridge_cv.best_params_)
print('Best score: ', ridge_cv.best_score_)
# Predict on the holdout set and compute metrics
Y_pred3 = ridge_cv.predict(X_holdout3)
r2_ridge_cv = ridge_cv.score(X_holdout3, Y_holdout3)
mse_ridge_cv = mean_squared_error(Y_holdout3, Y_pred3)
print("Tuned Ridge R^2: {}".format(r2_ridge_cv))
print("Tuned Ridge MSE: {}".format(mse_ridge_cv))


# In[159]:


# Drop features according to results from Ridge optimized by Grid Search Cross-Validation
X_train0 = X_train3
Y_train0 = Y_train3
X_holdout0 = X_holdout3
Y_holdout0 = Y_holdout3
nx = np.array([])
R2_train_rcv = np.array([])
R2_holdout_rcv = np.array([])
R2_train_RF = np.array([])
R2_holdout_RF = np.array([])
alpha_grid = {'alpha':np.logspace(-6,-1,50)}
for i in range(31):
    n0 = X_train0.shape[1]
    nx = np.append(nx,n0)
    print('No. of predictors:', n0)
    ridge = Ridge(alpha=0.1,normalize=True) # Initialize Ridge regressor
    rcv = GridSearchCV(ridge,alpha_grid,cv=5) # Initiate GridSearch CV regressor
    rcv.fit(X_train0,Y_train0) #Fit data and tune alpha to optimal value
    print('   Optimal Ridge parameter:', rcv.best_params_)
    print('   Best score: ', rcv.best_score_)
    # Predict on both training and holdout set and compute metrics
    Y_pred0 = rcv.predict(X_holdout0)
    R2_train_rcv0   = rcv.score(X_train0,   Y_train0)
    R2_holdout_rcv0 = rcv.score(X_holdout0, Y_holdout0)
    mse_rcv = mean_squared_error(Y_holdout0, Y_pred0)
    print("   Tuned Ridge R^2 (train,holdout): {}, {}".format(R2_train_rcv0,R2_holdout_rcv0))
    print("   Tuned Ridge MSE (holdout): {}".format(mse_rcv))
    R2_train_rcv = np.append(R2_train_rcv, R2_train_rcv0)
    R2_holdout_rcv = np.append(R2_holdout_rcv, R2_holdout_rcv0)
    # Compare to Random Forest result
    RF0 = RandomForestRegressor()
    RF0.fit(X_train0, Y_train0)
    R2_train_RF0 = RF0.score(X_train0, Y_train0)
    R2_holdout_RF0 = RF0.score(X_holdout0, Y_holdout0)
    print('   Random Forest R^2 (train,holdout): {}, {}'.format(R2_train_RF0,R2_holdout_RF0))
    R2_train_RF = np.append(R2_train_RF, R2_train_RF0)
    R2_holdout_RF = np.append(R2_holdout_RF, R2_holdout_RF0)
    # Drop least important RF feature
    i0 = (range(n0) != RF0.feature_importances_.argmin())
    X_train0 = X_train0[:,i0]
    X_holdout0 = X_holdout0[:,i0]


# In[156]:


# Try the same in non-regularized linear regression
X_train0 = X_train3
Y_train0 = Y_train3
X_holdout0 = X_holdout3
Y_holdout0 = Y_holdout3
nx = np.array([])
R2_train_LR = np.array([])
R2_holdout_LR = np.array([])
R2_train_RF = np.array([])
R2_holdout_RF = np.array([])
for i in range(31):
    n0 = X_train0.shape[1]
    nx = np.append(nx,n0)
    print('No. of predictors:', n0)
    LR0 = LinearRegression()
    LR0.fit(X_train0, Y_train0)
    R2_train_LR0 = LR0.score(X_train0, Y_train0)
    R2_holdout_LR0 = LR0.score(X_holdout0, Y_holdout0) 
    print('   NR Linear Regression R^2 (train,holdout): {}, {}'.format(R2_train_LR0,R2_holdout_LR0))
    R2_train_LR = np.append(R2_train_LR, R2_train_LR0)
    R2_holdout_LR = np.append(R2_holdout_LR, R2_holdout_LR0)
    # Compare to Random Forest result
    RF0 = RandomForestRegressor()
    RF0.fit(X_train0, Y_train0)
    R2_train_RF0 = RF0.score(X_train0, Y_train0)
    R2_holdout_RF0 = RF0.score(X_holdout0, Y_holdout0)
    print('   Random Forest R^2 (train,holdout): {}, {}'.format(R2_train_RF0,R2_holdout_RF0))
    R2_train_RF = np.append(R2_train_RF, R2_train_RF0)
    R2_holdout_RF = np.append(R2_holdout_RF, R2_holdout_RF0)
    # Drop least important RF feature
    i0 = (range(n0) != RF0.feature_importances_.argmin())
    X_train0 = X_train0[:,i0]
    X_holdout0 = X_holdout0[:,i0]


# In[160]:


plt.plot(nx,R2_train_rcv)
plt.plot(nx,R2_train_LR)
plt.plot(nx,R2_train_RF)
plt.plot(nx,R2_holdout_rcv)
plt.plot(nx,R2_holdout_LR)
plt.plot(nx,R2_holdout_RF)
plt.show()


# In[54]:


# Create array of feature indices ordered by importance
print(X_train.shape, Y_train.shape, RF.feature_importances_.shape)
imp = np.flip(np.argsort(RF.feature_importances_),0)
print(imp[0])


# In[55]:


h0 = np.logical_not(X_train[:,imp[0]])
h1 = np.logical_and(X_train[:,imp[0]],1)
print(Y_train[h0].mean(), Y_train[h0].std())
print(Y_train[h1].mean(), Y_train[h1].std())
print(Y_train[h1].mean()-Y_train[h0].mean())
plt.figure(figsize=(17,5))
plt.hist(Y_train[h0],1000)
plt.hist(Y_train[h1],1000)
#plt.yscale('symlog')
plt.legend(['0','1'])
plt.show()


# In[243]:


n_train = X_train.shape[0]
n_test  = X_test.shape[0]
for i in range(256):
    h1 = (X_train[:,imp[i]] == 1)
    t1 = (X_test[:,imp[i]] == 1)
    print(i+1, ') Feature ', imp[i],': train ',sum(h1), 100*sum(h1)/n_train,'%, test ',sum(t1), 100*sum(t1)/n_test,'%')


# In[294]:


# Flip X such that Y(1) > Y(0) for all features
n2 = X_train2.shape[1]
X_trainb = (X_train == 1)
for i in range(n2): 
    b0 = X_trainb[:,imp[i]]
    if Y_train[np.logical_not(b0)].mean() > Y_train[b0].mean():
        X_trainb[b0] = np.logical_not(X_trainb[b0])


# In[295]:


'mean1'
Y_train2_mean1 = np.array([[Y_train[np.logical_and(X_trainb[:,imp[i]],X_trainb[:,imp[j]])].mean() for i in range(n2)] for j in range(n2)])
'mean0'
Y_train2_mean0 = np.array([[Y_train[np.logical_and(np.logical_not(X_trainb[:,imp[i]]),np.logical_not(X_trainb[:,imp[j]]))].mean() for i in range(n2)] for j in range(n2)])
'std1'
Y_train2_std1  = np.array([[Y_train[np.logical_and(X_trainb[:,imp[i]],X_trainb[:,imp[j]])].std() for i in range(n2)] for j in range(n2)])
'std0'
Y_train2_std0  = np.array([[Y_train[np.logical_and(np.logical_not(X_trainb[:,imp[i]]),np.logical_not(X_trainb[:,imp[j]]))].std() for i in range(n2)] for j in range(n2)])


# In[307]:


print(np.nanmax(Y_train2_mean1), np.nanmin(Y_train2_mean1))
plt.imshow(Y_train2_mean1)
plt.show()


# In[299]:


print(np.nanmax(Y_train2_mean0), np.nanmin(Y_train2_mean0))
plt.imshow(Y_train2_mean0)
plt.show()


# In[301]:


print(np.nanmax(Y_train2_mean1-Y_train2_mean0), np.nanmin(Y_train2_mean1-Y_train2_mean0))
plt.imshow(Y_train2_mean1-Y_train2_mean0)
plt.show()


# In[308]:


# Try 3D plota
from mpl_toolkits.mplot3d import Axes3D


# In[338]:


x = np.resize(np.array(range(1,32)).reshape(-1,1),(31,31))
y = x.T
z = Y_train2_mean1
ax = plt.gca(projection='3d')
ax.plot_surface(x,y,z)
plt.show()


# ## 3. Deep Learning implementation

# In[15]:


### IMPLEMENT DEEP LEARNING
# Import deep learning modules from keras library
from keras.layers import Dense # For dense layers
from keras.models import Sequential # For sequential layering
from keras.callbacks import EarlyStopping # For stopping execution


# In[34]:


# Initialize model
input_shape = (X_train2.shape[1],) # Shape of input data
model_DL = Sequential()
# Start with 1 hidden layer with same nodes as input data
model_DL.add(Dense(input_shape[0],activation='relu',input_shape=input_shape))
# Output layer
model_DL.add(Dense(1))
# Compile model
model_DL.compile(optimizer='adam',loss='mean_squared_error')
model_DL.summary()
# Early stopping monitor w/ patience=3 (stop after 3 runs without improvements)
early_stopping_monitor = EarlyStopping(patience=2)


# In[35]:


# Fit model using 20% of data for validation
model_DL.fit(X_train2, Y_train, validation_split=0.2, epochs=20, callbacks=[early_stopping_monitor])
# To save model: model_DL.save('file.h5')
# To predict: model_DL.predict(X_train2)


# In[38]:


Y_train_DLpred = model_DL.predict(X_train2)
Y_test_DLpred  = model_DL.predict(X_test2)


# In[42]:


from sklearn.metrics import mean_squared_error
mse_DL = mean_squared_error(Y_train, Y_train_DLpred)
print("Deep Learning MSE: {}".format(mse_DL))


# In[43]:


help(model_DL)


# In[52]:


# Write results to file
write_to_file("sample_Copete_DL_v1.csv", Y_test_DLpred[:,0])


# In[65]:


# Function to train 1-layer neural network of a given number of nodes
def train_model_DL1(X_train,Y_train,n_nodes):
    input_shape = (X_train.shape[1],) # Shape of input data
    # Initialize model
    model_DL = Sequential()
    # First layer
    model_DL.add(Dense(n_nodes,activation='relu',input_shape=input_shape))
    # Output layer
    model_DL.add(Dense(1))
    # Compile model
    model_DL.compile(optimizer='adam',loss='mean_squared_error')
    model_DL.summary()
    # Early stopping monitor w/ patience=3 (stop after 3 runs without improvements)
    early_stopping_monitor = EarlyStopping(patience=3)
    # Fit model using 20% of data for validation
    model_DL.fit(X_train, Y_train, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
    Y_train_DLpred = model_DL.predict(X_train)
    mse_DL = mean_squared_error(Y_train, Y_train_DLpred)
    print('DONE')
    return mse_DL


# In[66]:


### Optimize number of nodes for 1-layer deep learning model
n_nodes = 20
mse_DL1 = train_model_DL1(X_train2,Y_train,n_nodes)
print("Deep Learning MSE ({} nodes): {}".format(n_nodes,mse_DL1))


# In[67]:


n_nodes = 10
mse_DL1 = train_model_DL1(X_train2,Y_train,n_nodes)
print("Deep Learning MSE ({} nodes): {}".format(n_nodes,mse_DL1))


# In[68]:


n_nodes = 5
mse_DL1 = train_model_DL1(X_train2,Y_train,n_nodes)
print("Deep Learning MSE ({} nodes): {}".format(n_nodes,mse_DL1))


# In[69]:


n_nodes = [40,50,60]
for n in n_nodes:
    mse_DL1 = train_model_DL1(X_train2,Y_train,n)
    print("Deep Learning MSE ({} nodes): {}".format(n,mse_DL1))


# In[70]:


n_nodes = [25,35]
for n in n_nodes:
    mse_DL1 = train_model_DL1(X_train2,Y_train,n)
    print("Deep Learning MSE ({} nodes): {}".format(n,mse_DL1))


# In[11]:


# Function to train multi-layered neural network of a given number of nodes
from sklearn.metrics import mean_squared_error
def train_model_DL(X_train,Y_train,n_nodes):
    input_shape = (X_train.shape[1],) # Shape of input data
    # Initialize model
    model_DL = Sequential()
    for i in range(len(n_nodes)):
        if i == 0:
            # First layer
            model_DL.add(Dense(n_nodes[i],activation='relu',input_shape=input_shape))
        else:
            # Subsequent layers
            model_DL.add(Dense(n_nodes[i],activation='relu'))    
    # Output layer
    model_DL.add(Dense(1))
    # Compile model
    model_DL.compile(optimizer='adam',loss='mean_squared_error')
    model_DL.summary()
    # Early stopping monitor w/ patience=3 (stop after 3 runs without improvements)
    early_stopping_monitor = EarlyStopping(patience=3)
    # Fit model using 20% of data for validation
    model_DL.fit(X_train, Y_train, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
    Y_train_DLpred = model_DL.predict(X_train)
    mse_DL = mean_squared_error(Y_train, Y_train_DLpred)
    print('DONE')
    return mse_DL


# In[76]:


# Loop over size of second layer
n_nodes = [[31,5],[31,10],[31,20],[31,31],[31,40],[31,50],[31,60]]
mse_DL = []
for n in n_nodes:
    mse_DL0 = train_model_DL(X_train2,Y_train,n)
    print("Deep Learning MSE ({} nodes): {}".format(n,mse_DL0))
    mse_DL = np.append(mse_DL, mse_DL0)


# In[77]:


mse_DL


# In[94]:


plt.scatter(np.array(n_nodes)[:,1],mse_DL)
#plt.title('MSE vs second layer nodes')
plt.show()


# In[95]:


# Loop over size of third layer
n_nodes = np.array([[31,31,5],[31,31,10],[31,31,20],[31,31,31],[31,31,40],[31,31,50],[31,31,60]])
mse_DL = []
for n in n_nodes:
    mse_DL0 = train_model_DL(X_train2,Y_train,n)
    print("Deep Learning MSE ({} nodes): {}".format(n,mse_DL0))
    mse_DL = np.append(mse_DL, mse_DL0)


# In[98]:


plt.scatter(np.array(n_nodes)[:,2],mse_DL)
#plt.title('MSE vs third layer nodes')
plt.show()


# In[99]:


# Loop over size of first layer
n_nodes = np.array([[5],[10],[20],[31],[40],[50],[60]])
mse_DL1 = []
for n in n_nodes:
    mse_DL0 = train_model_DL(X_train2,Y_train,n)
    print("Deep Learning MSE ({} nodes): {}".format(n,mse_DL0))
    mse_DL1 = np.append(mse_DL1, mse_DL0)


# In[109]:


plt.scatter(n_nodes,mse_DL1)
#plt.title('MSE vs third layer nodes')
plt.ylim((0.075,0.084))
plt.show()


# In[117]:


# Loop over size of first layer (pt 2)
n_nodes_DL1_1 = np.array([[70],[80],[100],[150],[200]])
mse_DL1_1 = []
for n in n_nodes_DL1_1:
    mse_DL0 = train_model_DL(X_train2,Y_train,n)
    print("Deep Learning MSE ({} nodes): {}".format(n,mse_DL0))
    mse_DL1_1 = np.append(mse_DL1_1, mse_DL0)


# In[126]:


plt.scatter(np.append(n_nodes,n_nodes_DL1_1),np.append(mse_DL1,mse_DL1_1))
#plt.title('MSE vs third layer nodes')
plt.ylim((0.075,0.084))
plt.xscale('log')
plt.show()


# In[116]:


mse_DL1_1


# In[41]:


# Redo run with 1 layer, 150 nodes (outlier)
n_nodes_DL1_2 = np.array([[150]])
mse_DL1_2 = []
for n in n_nodes_DL1_2:
    mse_DL0 = train_model_DL(X_train2,Y_train,n)
    print("Deep Learning MSE ({} nodes): {}".format(n,mse_DL0))
    mse_DL1_2 = np.append(mse_DL1_2, mse_DL0)


# # 4. Deep Learning on full set of features

# In[3]:


datafiles_X_train = ['../Fangli/XY_train_sub/X_fp_train_sub'+s+'.csv' for s in np.char.mod('%d', range(8))]
datafiles_X_train


# In[4]:


from pathlib import Path
import time
X_train_fp = np.array([])
for file in datafiles_X_train:
    print('Reading file '+file+'...')
    while not Path(file).is_file(): time.sleep(10)
    if X_train_fp.shape[0] == 0: 
        X_train_fp = pd.read_csv(file).values
    else: X_train_fp = np.vstack((X_train_fp, pd.read_csv(file).values))
X_train_fp.shape


# In[17]:


# Train with 1 layer, same nodes as input (2048)
n_nodes_fp_DL = np.array([[X_train_fp.shape[1]]])
mse_fp_DL = []
for n in n_nodes_fp_DL:
    mse_fp_DL0 = train_model_DL(X_train_fp,Y_train,n)
    print("Deep Learning MSE ({} nodes): {}".format(n,mse_fp_DL0))
    mse_fp_DL = np.append(mse_fp_DL, mse_fp_DL0)


# In[26]:


# Features with all 0 values
i0 = (np.sum(X_train_fp,axis=0) == 0)
i1 = np.logical_not(i0)
print('Total expressed / unexpressed molecular features: {} / {}'.format(np.sum(i1),np.sum(i0)))


# In[29]:


# Drop unexpressed features
X_train_fp1 = X_train_fp[:,i1]


# In[ ]:


# Run RF regression on full training set
RF_fp1 = RandomForestRegressor()
RF_fp1.fit(X_train_fp1, Y_train)
R2_RF_fp1 = RF_fp1.score(X_train_fp1, Y_train)
Y_train_fp_RFpred = RF_fp1.predict(X_train_fp1)
mse_RF_fp1 = mean_squared_error(Y_train, Y_train_fp_RFpred)
print("Random Forest (full training set): R^2 = {}, MSE = {}".format(R2_RF_fp1,mse_RF_fp1))   


# In[65]:


# Plot RF feature importances
plt.figure(figsize=(17,5))
plt.bar(range(len(RF_fp1.feature_importances_)), RF_fp1.feature_importances_)
plt.yscale('symlog')
plt.show()

