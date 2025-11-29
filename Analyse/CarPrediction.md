# **Analyse et Préparation des Données pour la Prédiction Automobile**

## **Sommaire**
- Introduction
- Description du Notebook
- Explications détaillées des cellules de code
- Conclusion

## **Introduction**
Ce document est une version Markdown enrichie du notebook Python fourni. Chaque partie du code est expliquée en français avec titres et sous‑titres.

## **Contenu du Notebook avec Explications**


### 💻 Code Python

```python
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#For ignoring warning
import warnings
warnings.filterwarnings("ignore")
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
df=pd.read_csv('/content/drive/MyDrive/Analyse_PayGapEurope/car_price_dataset_medium.csv')
df
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
df.shape
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Checking for Duplicates
df.duplicated().sum()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Removing Duplicates
df=df.drop_duplicates()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Checking for null values
df.isnull().sum()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
df.info()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
df.describe()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['Brand'] = le.fit_transform(df['Brand'])
df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Transmission'] = le.fit_transform(df['Transmission'])
df['Owner_Type'] = le.fit_transform(df['Owner_Type'])
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Let's check what's happened now
df
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
df.info()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# Let's check the distribution of Target variable.
# Using histplot because Price_USD is numerical
plt.figure(figsize=(10, 6))
sns.histplot(df['Price_USD'], kde=True)
plt.title('Target Distribution (Price_USD)')
plt.show()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# Define your threshold
price_limit = 60000

# Count values greater than the limit
expensive_cars = df[df['Price_USD'] > price_limit].shape[0]

# Count values less than the limit
cheaper_cars = df[df['Price_USD'] < price_limit].shape[0]

print(f"Cars with Price > {price_limit}: {expensive_cars}")
print(f"Cars with Price < {price_limit}: {cheaper_cars}")
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# function for plotting
def plot(col, df=df):
    return df.groupby(col)['Price_USD'].mean().plot(kind='bar', figsize=(8,5), ylabel='Average Price USD')
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
plot('Brand')
plt.title('Average Price by Brand')
plt.show()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
plot('Model_Year')
plt.title('Average Price by Model_Year')
plt.show()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
plot('Fuel_Type')
plt.title('Average Price by Fuel_Type')
plt.show()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
plot('Transmission')
plt.title('Average Price by Transmission')
plt.show()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# We split power into 3 groups: Low (<200), Medium (200-400), High (>400)
bins = [0, 200, 400, 1000]
labels = ['Low Power', 'Medium Power', 'High Power']
df['Power_Category'] = pd.cut(df['Max_Power_bhp'], bins=bins, labels=labels)

plot('Power_Category')
plt.title('Average Price by Power Category')
plt.show()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
plot('Seats')
plt.title('Average Price by number of Seats')
plt.show()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# We split Kilometers_Driven into 3 groups: Low (<200), Medium (200-400), High (>400)
bins = [0, 20000, 40000, 100000]
labels = ['Low KM', 'Medium KM', 'High KM']
df['Kilometers_Category'] = pd.cut(df['Kilometers_Driven'], bins=bins, labels=labels)

plot('Kilometers_Category')
plt.title('Average Price by Kilometers_Driven')
plt.show()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# Dropping columns with the weakest correlation to Price_USD
df_new = df.drop(columns=['Car_ID', 'Brand', 'Transmission', 'Fuel_Type'])
df_new
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Finding Correlation
# Drop the categorical columns that are not suitable for direct numerical correlation calculation
df_numerical_corr = df_new.drop(columns=['Power_Category', 'Kilometers_Category'])
cn=df_numerical_corr.corr()
cn
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Correlation
cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(cn,cmap=cmap,annot=True, square=True)
plt.show()
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
kot = cn[cn>=.40]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Purples")
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
df_new['Mileage_Year_Interaction'] = df_new['Mileage_kmpl'] * df_new['Model_Year']
df_new
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# Splitting independent (X) and dependent (y) variables
X = df_new.drop('Price_USD', axis=1)
y = df_new['Price_USD']
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# ADASYN is for classification tasks to handle imbalanced classes.
# Since 'Price_USD' is a continuous target variable for a regression problem, ADASYN is not applicable.
# If you were performing a classification task (e.g., classifying cars as 'cheap', 'medium', 'expensive'),
# you would first need to convert 'Price_USD' into discrete categories before using ADASYN.
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
len(X)
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Splitting data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# One-hot encode the categorical columns 'Power_Category' and 'Kilometers_Category'
X_train_encoded = pd.get_dummies(X_train, columns=['Power_Category', 'Kilometers_Category'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=['Power_Category', 'Kilometers_Category'], drop_first=True)

# Ensure all columns are aligned between training and testing sets after encoding
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# For regression problems, use a regression model, not LogisticRegression
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()

# Fit the regression model with the numerically encoded data
regression_model.fit(X_train_encoded, y_train)
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Predicting result using testing data with the preprocessed test set
y_lr_pred= regression_model.predict(X_test_encoded)
y_lr_pred
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# Model accuracy - using regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_lr_pred)
mse = mean_squared_error(y_test, y_lr_pred)
rmse = np.sqrt(mse) # Root Mean Squared Error
r2 = r2_score(y_test, y_lr_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# One-hot encode the categorical columns 'Power_Category' and 'Kilometers_Category'
X_train_encoded_dt = pd.get_dummies(X_train, columns=['Power_Category', 'Kilometers_Category'], drop_first=True)
X_test_encoded_dt = pd.get_dummies(X_test, columns=['Power_Category', 'Kilometers_Category'], drop_first=True)

# Ensure all columns are aligned between training and testing sets after encoding
X_train_encoded_dt, X_test_encoded_dt = X_train_encoded_dt.align(X_test_encoded_dt, join='left', axis=1, fill_value=0)

# Use DecisionTreeRegressor for regression problems
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=0)

# Fit the regression model with the numerically encoded data
dt_model.fit(X_train_encoded_dt, y_train)
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Predicting result using testing data
y_dt_pred= dt_model.predict(X_test_encoded_dt)
y_dt_pred
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# Model accuracy - using regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_dt_pred)
mse = mean_squared_error(y_test, y_dt_pred)
rmse = np.sqrt(mse) # Root Mean Squared Error
r2 = r2_score(y_test, y_dt_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Fitting K-NN regressor to the training set
from sklearn.neighbors import KNeighborsRegressor
# Use X_train_encoded_dt (or X_train_encoded) which has categorical columns one-hot encoded
knn_model= KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_encoded_dt, y_train)
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
#Predicting result using testing data
y_knn_pred= knn_model.predict(X_test_encoded_dt)
y_knn_pred
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# Model accuracy - using regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_knn_pred)
mse = mean_squared_error(y_test, y_knn_pred)
rmse = np.sqrt(mse) # Root Mean Squared Error
r2 = r2_score(y_test, y_knn_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*


### 💻 Code Python

```python
# K-Fold Cross Validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# One-hot encode the categorical columns for cross-validation
X_cv_encoded = pd.get_dummies(X, columns=['Power_Category', 'Kilometers_Category'], drop_first=True)

# Linear Regression model
lr_scores = cross_val_score(LinearRegression(), X_cv_encoded, y, cv=kf, scoring='r2')

# Decision tree model
dt_scores = cross_val_score(DecisionTreeRegressor(random_state=0), X_cv_encoded, y, cv=kf, scoring='r2')

# KNN model
knn_scores = cross_val_score(KNeighborsRegressor(n_neighbors=5), X_cv_encoded, y, cv=kf, scoring='r2')

print("Linear Regression models' average R2 score:", np.mean(lr_scores))
print("Decision tree models' average R2 score:", np.mean(dt_scores))
print("KNN models' average R2 score:", np.mean(knn_scores))
```

#### ✨ Explication du code

*(Ajoutez ici une explication détaillée selon votre besoin — vous pourrez me demander d’enrichir chaque cellule !)*
