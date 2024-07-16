# README

## House Price Prediction Project

This project aims to analyze and predict house prices using a dataset from King County, which includes Seattle. We employ various data visualization techniques and machine learning models to understand the data and build predictive models.

### Project Structure

1. **Data Loading and Initial Exploration**


```python
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Loading the dataset
data = pd.read_csv("kc_house_data.csv")

# Displaying the first few rows of the dataset
print(data.head())

# Displaying basic statistics of the dataset
print(data.describe())
```

2. **Data Visualization**

   - **Bar Plot of Number of Bedrooms**

   ```python
   data['bedrooms'].value_counts().plot(kind='bar')
   plt.title('Number of Bedrooms')
   plt.xlabel('Bedrooms')
   plt.ylabel('Count')
   sns.despine()
   plt.show()
   ```

   - **Geographical Plot of Houses**

   ```python
   plt.figure(figsize=(10,10))
   sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
   plt.ylabel('Longitude', fontsize=12)
   plt.xlabel('Latitude', fontsize=12)
   plt.show()
   ```

   - **Scatter Plots**

     - **Price vs Square Feet**

     ```python
     plt.scatter(data.price, data.sqft_living)
     plt.title("Price vs Square Feet")
     plt.xlabel("Price")
     plt.ylabel("Square Feet")
     plt.show()
     ```

     - **Price vs Location**

     ```python
     plt.scatter(data.price, data.long)
     plt.title("Price vs Location of the Area")
     plt.xlabel("Price")
     plt.ylabel("Longitude")
     plt.show()

     plt.scatter(data.price, data.lat)
     plt.title("Latitude vs Price")
     plt.xlabel("Price")
     plt.ylabel("Latitude")
     plt.show()
     ```

     - **Bedroom vs Price**

     ```python
     plt.scatter(data.bedrooms, data.price)
     plt.title("Bedrooms and Price")
     plt.xlabel("Bedrooms")
     plt.ylabel("Price")
     plt.show()
     ```

     - **Waterfront vs Price**

     ```python
     plt.scatter(data.waterfront, data.price)
     plt.title("Waterfront vs Price (0 = no waterfront)")
     plt.xlabel("Waterfront")
     plt.ylabel("Price")
     plt.show()
     ```

3. **Data Preprocessing**

   ```python
   train1 = data.drop(['id', 'price'], axis=1)
   data.floors.value_counts().plot(kind='bar')
   plt.title("Number of Floors")
   plt.xlabel("Floors")
   plt.ylabel("Count")
   plt.show()
   ```

   - **Scatter Plot: Price vs Floors**

   ```python
   plt.scatter(data.floors, data.price)
   plt.title("Price vs Number of Floors")
   plt.xlabel("Floors")
   plt.ylabel("Price")
   plt.show()
   ```

   - **Scatter Plot: Price vs Zipcode**

   ```python
   plt.scatter(data.zipcode, data.price)
   plt.title("Which is the Pricey Location by Zipcode?")
   plt.xlabel("Zipcode")
   plt.ylabel("Price")
   plt.show()
   ```

4. **Model Building and Evaluation**

   - **Linear Regression**

   ```python
   from sklearn.linear_model import LinearRegression

   reg = LinearRegression()
   labels = data['price']
   conv_dates = [1 if values == 2014 else 0 for values in data.date]
   data['date'] = conv_dates
   train1 = data.drop(['id', 'price'], axis=1)

   from sklearn.model_selection import train_test_split
   x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.10, random_state=2)

   reg.fit(x_train, y_train)
   print(reg.score(x_test, y_test))
   ```

   - **Gradient Boosting Regressor**

   ```python
   from sklearn import ensemble

   clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                            learning_rate=0.1, loss='ls')
   clf.fit(x_train, y_train)
   print(clf.score(x_test, y_test))

   t_sc = np.zeros((clf.n_estimators), dtype=np.float64)
   for i, y_pred in enumerate(clf.staged_predict(x_test)):
       t_sc[i] = clf.loss_(y_test, y_pred)

   testsc = np.arange((clf.n_estimators)) + 1
   plt.figure(figsize=(12, 6))
   plt.subplot(1, 2, 1)
   plt.plot(testsc, clf.train_score_, 'b-', label='Set dev train')
   plt.plot(testsc, t_sc, 'r-', label='Set dev test')
   plt.legend()
   plt.show()
   ```

5. **PCA for Dimensionality Reduction**

   ```python
   from sklearn.preprocessing import scale
   from sklearn.decomposition import PCA

   pca = PCA()
   pca.fit_transform(scale(train1))
   ```

### Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### How to Run

1. Ensure all the required libraries are installed.
2. Load the dataset `kc_house_data.csv`.
3. Run the script to see data visualizations and model performance outputs.

### Conclusion

This project provides an analysis of house prices in King County using visualizations and machine learning models. It demonstrates the application of Linear Regression and Gradient Boosting Regressor to predict house prices based on various features.

---

Feel free to customize and expand this README as per your specific requirements and findings from the project.


The tutorial and write up for the code can be found here 
https://medium.com/towards-data-science/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f


