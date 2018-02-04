
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[6]:


"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("train.csv.gz")
df_test = pd.read_csv("test.csv.gz")


# In[7]:


df_train.head()


# In[8]:


df_test.head()


# In[9]:


#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)


# In[10]:


#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_all.head()


# In[11]:


"""
Example Feature Engineering

this calculates the length of each smile string and adds a feature column with those lengths
Note: this is NOT a good feature and will result in a lower score!
"""
#smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
#df_all['smiles_len'] = pd.DataFrame(smiles_len)


# In[13]:


#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print("Train features:", X_train.shape)
print("Train gap:", Y_train.shape)
print("Test features:", X_test.shape)


# In[14]:


LR = LinearRegression()
LR.fit(X_train, Y_train)
LR_pred = LR.predict(X_test)


# In[15]:


RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)


# In[16]:


def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


# In[ ]:


write_to_file("sample1.csv", LR_pred)
write_to_file("sample2.csv", RF_pred)


# In[20]:


help(LR)


# In[45]:


plt.figure(figsize=(15,10))
plt.bar(range(len(LR.coef_)), LR.coef_)
plt.yscale('symlog')
plt.show()


# In[38]:


# Calculate initial R^2
print(LR.score(X_train, Y_train))
print(RF.score(X_train, Y_train))


# In[43]:


help(RF)


# In[44]:


plt.bar(range(len(RF.feature_importances_)), RF.feature_importances_, 2)
plt.yscale('symlog')
plt.show()


# In[46]:


# Drop predictors that are unimportant in both LR and RF
LR_coef = LR.coef_
RF_imp = RF.feature_importances_
i0 = (LR_coef == 0) & (RF_imp == 0)
X_train1 = X_train[:,i0 == False]


# In[47]:


# Redo regressions
LR1 = LinearRegression()
LR1.fit(X_train1, Y_train)
RF1 = RandomForestRegressor()
RF1.fit(X_train1, Y_train)


# In[49]:


# Recalculate R^2
print(LR1.score(X_train1, Y_train))
print(RF1.score(X_train1, Y_train))


# In[51]:


plt.bar(range(len(LR1.coef_)), LR1.coef_)
plt.yscale('symlog')
plt.show()


# In[52]:


plt.bar(range(len(RF1.feature_importances_)), RF1.feature_importances_, 2)
plt.yscale('symlog')
plt.show()


# In[53]:


# Test only features with RF importance > 0
RF_imp1 = RF1.feature_importances_
LR_coef1 = LR.coef_
i1 = (RF_imp1 > 0)
X_train2 = X_train1[:,i1]


# In[54]:


LR2 = LinearRegression()
LR2.fit(X_train2, Y_train)
RF2 = RandomForestRegressor()
RF2.fit(X_train2, Y_train)
print(LR2.score(X_train2, Y_train))
print(RF2.score(X_train2, Y_train))


# In[55]:


X_train2.shape


# In[57]:


plt.bar(range(len(LR2.coef_)), LR2.coef_)
plt.yscale('symlog')
plt.show()


# In[59]:


plt.bar(range(len(RF2.feature_importances_)), RF2.feature_importances_)
plt.yscale('symlog')
plt.show()


# In[70]:


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


# In[92]:


plt.plot(nx,R2_LR)
plt.plot(nx,R2_RF,c='r')
plt.show()


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


# In[174]:


# Create array of feature indices ordered by importance
print(X_train.shape, Y_train.shape, RF.feature_importances_.shape)
imp = np.flip(np.argsort(RF.feature_importances_),0)
print(imp[0])


# In[276]:


h0 = np.logical_not(X_train[:,imp[0]])
h1 = np.logical_and(X_train[:,imp[0]],1)
print(Y_train[h0].mean(), Y_train[h0].std())
print(Y_train[h1].mean(), Y_train[h1].std())
print(Y_train[h1].mean()-Y_train[h0].mean())
plt.figure(figsize=(17,5))
plt.hist(Y_train[h0],1000)
plt.hist(Y_train[h1],1000)
plt.yscale('symlog')
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


# In[333]:


zhelp("(Axes3D.plot_surface)")


# In[328]:


np.resize(np.array(range(1,32)).reshape(-1,1),(31,31)).T

