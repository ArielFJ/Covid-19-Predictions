import io
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
plt.style.use('seaborn')

url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'

res = requests.get(url).content
df = pd.read_csv(io.StringIO(res.decode('utf-8')), error_bad_lines=False)

grpd = df.groupby('location')

for name, data in grpd:        

    # Name of the country to predict, or "World" to make it global
    if(name == "Dominican Republic"):
        dates = data['date']        
        new_cases = data['new_cases']
        total_cases = data['total_cases']
        new_deaths = data['new_deaths']
        total_deaths = data['total_deaths']

       
        days_since_31_12_2019 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
        new_cases_reshaped = np.array(new_cases).reshape(-1, 1)
        new_deaths_reshaped = np.array(new_deaths).reshape(-1, 1)
        total_cases_reshaped = np.array(total_cases).reshape(-1, 1)
        total_deaths_reshaped = np.array(total_deaths).reshape(-1, 1)

              
        # Furute forecasting for next 365 days
        days_in_future = 340
        future_forecast = np.array([i for i in range(len(dates) + days_in_future)]).reshape(-1, 1)
        adjusted_dates = future_forecast[:-days_in_future]

        # Convert all the integers into datetime for better visualization
        start = '31-12-2019'
        start_date = datetime.datetime.strptime(start, '%d-%m-%Y')
        future_forecast_dates = []
        for i in range(len(future_forecast)):
            future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%d-%m-%Y'))

        # Tain data and test data for the model
        x_new_cases_train, x_new_cases_test, y_new_cases_train, y_new_cases_test = train_test_split(adjusted_dates, new_cases_reshaped, test_size=0.2)        
        x_total_cases_train, x_total_cases_test, y_total_cases_train, y_total_cases_test = train_test_split(adjusted_dates, total_cases_reshaped, test_size=0.2)        
        x_new_deaths_train, x_new_deaths_test, y_new_deaths_train, y_new_deaths_test = train_test_split(adjusted_dates, new_deaths_reshaped, test_size=0.2)        
        x_total_deaths_train, x_total_deaths_test, y_total_deaths_train, y_total_deaths_test = train_test_split(adjusted_dates, total_deaths_reshaped, test_size=0.2)        

        # Linear regression
        '''
        from sklearn.linear_model import LinearRegression
        
        # Linear regression for cases
        linear_model_cases = LinearRegression(normalize=True, fit_intercept=True)
        linear_model_cases.fit(x_new_cases_train, y_new_cases_train)
        cases_test_linear_pred = linear_model_cases.predict(x_new_cases_test)
        cases_linear_pred = linear_model_cases.predict(future_forecast)

        # Linear regression for deaths
        linear_model_deaths = LinearRegression(normalize=True, fit_intercept=True)
        linear_model_deaths.fit(x_new_deaths_train, y_new_deaths_train)
        deaths_test_linear_pred = linear_model_deaths.predict(x_new_deaths_test)
        deaths_linear_pred = linear_model_deaths.predict(future_forecast)
        

        # Prediction of cases by linear regression        
        plt.figure(figsize=(20,12))
        plt.scatter(adjusted_dates, new_cases_reshaped, color="blue")
        plt.scatter(adjusted_dates, new_deaths_reshaped, color="green")

        plt.plot(future_forecast, cases_linear_pred, linestyle="dashed", color="orange")
        plt.plot(future_forecast, deaths_linear_pred, linestyle="dashed", color="red")

        plt.title('Número de casos de Covid a lo largo del tiempo', size=30)
        plt.xlabel('Días desde 31/12/2019', size=30)
        plt.ylabel('Número de casos', size=30)
        plt.legend(['Predicciones de casos', 'Predicciones de muertes', 'Casos confirmados', 'Muertes confirmadas'])
        plt.xticks(size=15)
        plt.yticks(size=15)
        '''

        # SVM model
        
        # SVM parameters
        kernel = ['poly', 'sigmoid', 'rbf']
        c = [0.01, 0.1, 1, 10]
        gamma = [0.01, 0.1, 1]
        epsilon = [0.01, 0.1, 1]
        shrinking = [True, False]
        svm_grid = {'kernel': kernel, 'C': c, 'gamma': gamma, 'epsilon': epsilon, 'shrinking':shrinking }

        # SVM for cases
        svm = SVR()
        svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=3, verbose=1)
        svm_search.fit(x_new_cases_train, y_new_cases_train.ravel())

        svm_confirmed = svm_search.best_estimator_
        svm_pred = svm_confirmed.predict(future_forecast)        
        svm_test_pred = svm_confirmed.predict(x_new_cases_test)

        # SVM for deaths
        svm2 = SVR()
        svm_search2 = RandomizedSearchCV(svm2, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=3, verbose=1)
        svm_search2.fit(x_new_deaths_train, y_new_deaths_train.ravel())

        svm_confirmed2 = svm_search2.best_estimator_
        svm_pred2 = svm_confirmed2.predict(future_forecast)        
        svm_test_pred2 = svm_confirmed2.predict(x_new_deaths_test)

        # total number of covid cases over time
        plt.figure(figsize=(20,12))

        plt.scatter(adjusted_dates, new_cases_reshaped, color="blue")
        plt.scatter(adjusted_dates, new_deaths_reshaped, color="green")

        plt.plot(future_forecast, svm_pred, linestyle="dashed", color="purple")
        plt.plot(future_forecast, svm_pred2, linestyle="dashed", color="pink")
        
        plt.title('Número de casos de Covid a lo largo del tiempo', size=30)
        plt.xlabel('Días desde 31/12/2019', size=30)
        plt.ylabel('Número de casos', size=30)        
        plt.legend(['Predicciones de casos', 'Predicciones de muertes', 'Casos confirmados', 'Muertes confirmadas'])
        plt.xticks(size=15)
        plt.yticks(size=15)
        
       
        plt.show()        
