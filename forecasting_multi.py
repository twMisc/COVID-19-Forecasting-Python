#%% load packages
import xgboost as xgb
from load_data import get_all_time_series
from load_data import to_float_vec
import os
import random
import numpy as np 
import tensorflow as tf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy.matlib
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from tabulate import tabulate
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#%% initial settings
#observe_days = 30
#predict_days = 10
#keepdays = 150
#training_countries = ['US','Spain','Belgium','China','France','Germany','United Kingdom','Italy']
from forecasting_setup import observe_days, predict_days, keepdays, training_countries
tabulate_format = 'grid'
#%% load datset from ccse github using written function
days = observe_days + predict_days
mypath = r'../COVID-19/'
subpath = r'csse_covid_19_data/csse_covid_19_time_series'
the_path = os.path.join(mypath,subpath)
[df_infected,df_confirmed,df_recovered,df_deaths] = get_all_time_series(the_path) #%% create df of needed information for each country
#from information import df_information
#print(df_information)

#%% normalize 
def to_normal(x,mean,std):
    if std==0:
        std=1
    return (x - mean)/(std)
def to_nnormal(x,mean,std):
    for i in range(len(std)):
        if std[i]==0:
            std[i]==1
    return x *  np.matlib.repmat(std,predict_days,1).transpose()+ np.matlib.repmat(mean,predict_days,1).transpose() 
# %% create the dataset 
S = training_countries
# initialize a matrix for taining sets
x_test = []
y_test = []
x_train = []
y_train = []
#test_size = 0.33
#S_test = random.sample(list(S),int(test_size*len(S)))
#S_train = list(set(S) - set(S_test))

for s in S:
    country_code = s    
    confirmed = to_float_vec(df_confirmed[country_code].values)
    infected = to_float_vec(df_infected[country_code].values)
    deaths = to_float_vec(df_deaths[country_code].values)
    recovered = to_float_vec(df_recovered[country_code].values)
    #population = df_information[df_information.country==country_code].population.values[0] 

    confirmed_o = confirmed 
    deaths_o  = deaths
    recovered_o = recovered

    eps = 0.000000000000000001

    #%% set time vairables
    num_times = len(infected)
    # %% generate
    for i in range(num_times-days+1 -keepdays):  
        std = [np.std(confirmed[i:observe_days+i]),np.std(deaths[i:observe_days+i]),np.std(recovered[i:observe_days+i])]
        mean = [np.mean(confirmed[i:observe_days+i]),np.mean(deaths[i:observe_days+i]),np.mean(recovered[i:observe_days+i])]


        x_train.append(np.concatenate([to_normal(confirmed[i:observe_days+i],mean[0],std[0]),to_normal(deaths[i:observe_days+i],mean[1],std[1])]))
        

        y_train.append(np.concatenate([to_normal(confirmed_o[observe_days+i:days+i],mean[0],std[0]),to_normal(deaths_o[observe_days+i:days+i],mean[1],std[1])]))
# %% the train and test
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = []
y_test = []


#%% set of all countries
S_all = df_confirmed.keys()
S2 = np.sort(list(set(df_confirmed.keys())-set(S)))
for s in S_all:
    country_code = s
    confirmed = to_float_vec(df_confirmed[country_code].values)
    infected = to_float_vec(df_infected[country_code].values)
    deaths = to_float_vec(df_deaths[country_code].values)
    recovered = to_float_vec(df_recovered[country_code].values)
    #population = df_information[df_information.country==country_code].population.values[0]

    confirmed_o = confirmed 
    deaths_o  = deaths

    num_times = len(infected)
    if s in S2:
        for i in range(num_times - days +1):
            std = [np.std(confirmed[i:observe_days+i]),np.std(deaths[i:observe_days+i]),np.std(recovered[i:observe_days+i])]
            mean = [np.mean(confirmed[i:observe_days+i]),np.mean(deaths[i:observe_days+i]),np.mean(recovered[i:observe_days+i])]


            x_test.append(np.concatenate([to_normal(confirmed[i:observe_days+i],mean[0],std[0]),to_normal(deaths[i:observe_days+i],mean[1],std[1])]))
            

            y_test.append(np.concatenate([to_normal(confirmed_o[observe_days+i:days+i],mean[0],std[0]),to_normal(deaths_o[observe_days+i:days+i],mean[1],std[1])]))
    else:
        for i in range((num_times - days -keepdays+1),num_times - days +1):
            std = [np.std(confirmed[i:observe_days+i]),np.std(deaths[i:observe_days+i]),np.std(recovered[i:observe_days+i])]
            mean = [np.mean(confirmed[i:observe_days+i]),np.mean(deaths[i:observe_days+i]),np.mean(recovered[i:observe_days+i])]


            x_test.append(np.concatenate([to_normal(confirmed[i:observe_days+i],mean[0],std[0]),to_normal(deaths[i:observe_days+i],mean[1],std[1])]))
            

            y_test.append(np.concatenate([to_normal(confirmed_o[observe_days+i:days+i],mean[0],std[0]),to_normal(deaths_o[observe_days+i:days+i],mean[1],std[1])]))

x_test = np.array(x_test)
y_test = np.array(y_test)
y1 = y_train[:,0:predict_days]
y2 = y_train[:,predict_days:2*predict_days]
#%% tensorflow
tf.keras.backend.set_floatx('float64')
confirmed_regressor_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=predict_days)
])
confirmed_regressor_tf.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
deaths_regressor_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=predict_days)
])
deaths_regressor_tf.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

confirmed_regressor_tf.fit(x_train, y1,epochs=20,verbose=0)
deaths_regressor_tf.fit(x_train, y2,epochs=20,verbose=0)


# %%
def confirmed_score(x,y):
    y_true = y
    y_pred = (confirmed_regressor_tf(x).numpy())
    return 1-(np.sum((y_true - y_pred) ** 2))/np.sum((y_true - y_true.mean()) ** 2)
def deaths_score(x,y):
    y_true = y
    y_pred = (deaths_regressor_tf(x).numpy())
    return 1-(np.sum((y_true - y_pred) ** 2))/np.sum((y_true - y_true.mean()) ** 2)


# %% train linear regression and xgboost
confirmed_regressor = MultiOutputRegressor(xgb.XGBRegressor())
confirmed_regressor.fit(x_train, y1)
deaths_regressor = MultiOutputRegressor(xgb.XGBRegressor())
deaths_regressor.fit(x_train, y2)
confirmed_regressor_l = MultiOutputRegressor(LinearRegression())
confirmed_regressor_l.fit(x_train, y1)
deaths_regressor_l = MultiOutputRegressor(LinearRegression())
deaths_regressor_l.fit(x_train, y2)

# %% score
print('Data length:\t',num_times)
print(tabulate([['Confirmed', confirmed_regressor.score(x_train,y1),confirmed_regressor_l.score(x_train,y1),confirmed_score(x_train,y1)],['Deaths',deaths_regressor.score(x_train,y2),deaths_regressor_l.score(x_train,y2),deaths_score(x_train,y2)]],headers=['Training score', 'xgboost','linear regression','dense network'], tablefmt=tabulate_format))
print('\n')
print(tabulate([['Confirmed', confirmed_regressor.score(x_test,y_test[:,0:predict_days]),confirmed_regressor_l.score(x_test,y_test[:,0:predict_days]),confirmed_score(x_test,y_test[:,0:predict_days])],['Deaths',deaths_regressor.score(x_test,y_test[:,predict_days:2*predict_days]),deaths_regressor_l.score(x_test,y_test[:,predict_days:2*predict_days]),deaths_score(x_test,y_test[:,predict_days:2*predict_days])]],headers=['Testing score', 'xgboost','linear regression','dense network'], tablefmt=tabulate_format))


# %%
S2 = df_confirmed.keys()
#S2 = ['Japan','Taiwan*','India']
#random_list = random.sample(range(0, len(S2)), 15)
def print_and_draw(s):
    country_code = s
    confirmed = to_float_vec(df_confirmed[country_code].values)
    infected = to_float_vec(df_infected[country_code].values)
    deaths = to_float_vec(df_deaths[country_code].values)
    recovered = to_float_vec(df_recovered[country_code].values)
    #population = df_information[df_information.country==country_code].population.values[0]

    confirmed_o = confirmed 
    deaths_o  = deaths


    num_times = len(infected)


    x = []
    #y = []
    confirmed_p =[]
    confirmed_p2 = []
    confirmed_p3 = []

    deaths_p = []
    deaths_p2 = []
    deaths_p3 = []

    stds = []
    means =[]
    #eps = 0.000000000000000001
    for i in range(num_times-days+predict_days+1):
        std = [np.std(confirmed[i:observe_days+i]),np.std(deaths[i:observe_days+i]),np.std(recovered[i:observe_days+i])]
        stds.append(std)
        mean = [np.mean(confirmed[i:observe_days+i]),np.mean(deaths[i:observe_days+i]),np.mean(recovered[i:observe_days+i])]
        means.append(mean)

        x.append(np.concatenate([to_normal(confirmed[i:observe_days+i],mean[0],std[0]),to_normal(deaths[i:observe_days+i],mean[1],std[1])]))
    x = np.array(x)
    stds = np.array(stds)
    means = np.array(means)

    confirmed_p = to_nnormal(confirmed_regressor.predict(x),means[:,0],stds[:,0])
    confirmed_p2 = to_nnormal(confirmed_regressor_l.predict(x),means[:,0],stds[:,0])
    confirmed_p3 = confirmed_regressor_tf.predict(x)
    confirmed_p3 = to_nnormal(confirmed_p3,means[:,0],stds[:,0])

    deaths_p = to_nnormal(deaths_regressor.predict(x),means[:,1],stds[:,1])
    deaths_p2 = to_nnormal(deaths_regressor_l.predict(x),means[:,1],stds[:,1])
    deaths_p3 = deaths_regressor_tf.predict(x)
    deaths_p3 = to_nnormal(deaths_p3,means[:,1],stds[:,1])

    #%% create interval data
    confirmed_p_int = []
    confirmed_p2_int = []
    confirmed_p3_int = []
    deaths_p_int = []
    deaths_p2_int = []
    deaths_p3_int = []
    rest = (num_times-days+predict_days)%predict_days
    for i in range(int((num_times-days+predict_days)/predict_days)+1):
        confirmed_p_int.append(confirmed_p[i*predict_days+rest])
        confirmed_p2_int.append(confirmed_p2[i*predict_days+rest])
        confirmed_p3_int.append(confirmed_p3[i*predict_days+rest])

        deaths_p_int.append(deaths_p[i*predict_days+rest])
        deaths_p2_int.append(deaths_p2[i*predict_days+rest])
        deaths_p3_int.append(deaths_p3[i*predict_days+rest])

    confirmed_p_int = np.stack(confirmed_p_int).flatten()
    confirmed_p2_int = np.stack(confirmed_p2_int).flatten()
    confirmed_p3_int = np.stack(confirmed_p3_int).flatten()

    deaths_p_int = np.stack(deaths_p_int).flatten()
    deaths_p2_int = np.stack(deaths_p2_int).flatten()
    deaths_p3_int = np.stack(deaths_p3_int).flatten()

    #%%
    #from tabulate import tabulate
    print(country_code)
    print(tabulate([['Confirmed', np.max(np.abs(confirmed_o[observe_days+rest:]-confirmed_p_int[:-predict_days])),np.max(np.abs(confirmed_o[observe_days+rest:]-confirmed_p2_int[:-predict_days])),np.max(np.abs(confirmed_o[observe_days+rest:]-confirmed_p3_int[:-predict_days]))],['Deaths', np.max(np.abs(deaths_o[observe_days+rest:]-deaths_p_int[:-predict_days])),np.max(np.abs(deaths_o[observe_days+rest:]-deaths_p2_int[:-predict_days])),np.max(np.abs(deaths_o[observe_days+rest:]-deaths_p3_int[:-predict_days]))]],headers=['Max error', 'xgboost','linear regression','dense network'], tablefmt=tabulate_format))
    print('\n')

    #%%
    plt.figure()
    plt.title(country_code+' (confirmed)')
    plt.plot(np.arange(len(confirmed)),confirmed_o,label='true confirmed')
    plt.plot(np.arange(len(confirmed_p_int))+ observe_days + rest ,confirmed_p_int,marker = 'x',markersize = '3',linewidth = 0.5,label='xgboost')
    plt.plot(np.arange(len(confirmed_p_int))+ observe_days+ rest ,confirmed_p2_int,marker = 'x',markersize = '3',linewidth = 0.5,label='linear regression')
    plt.plot(np.arange(len(confirmed_p_int))+ observe_days+ rest ,confirmed_p3_int,marker = 'x',markersize = '3',linewidth = 0.5,label='dense network')
    plt.legend()
    plt.show()

    #%%
    plt.figure()
    plt.title(country_code+' (deaths)')
    plt.plot(np.arange(len(confirmed)),deaths_o,label='true deaths')
    plt.plot(np.arange(len(confirmed_p_int)) + observe_days+ rest ,deaths_p_int,marker = 'x',markersize = '3',linewidth = 0.5,label='xgboost')
    plt.plot(np.arange(len(confirmed_p_int))+ observe_days+ rest ,deaths_p2_int,marker = 'x',markersize = '3',linewidth = 0.5,label='linear regression')
    plt.plot(np.arange(len(confirmed_p_int))+ observe_days+ rest ,deaths_p3_int,marker = 'x',markersize = '3',linewidth = 0.5,label='dense network')
    plt.legend()
    plt.show()
    
    #%%
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(confirmed)), y=confirmed_o,
                    mode='lines+markers',
                    name='true confirmed'))
    fig.add_trace(go.Scatter(x=np.arange(len(confirmed_p_int))+ observe_days+ rest, y=confirmed_p_int,
                    mode='lines+markers',
                    name='xgboost'))
    fig.add_trace(go.Scatter(x=np.arange(len(confirmed_p_int))+ observe_days+ rest, y=confirmed_p2_int,
                    mode='lines+markers',
                    name='linear regression'))
    fig.add_trace(go.Scatter(x=np.arange(len(confirmed_p_int))+ observe_days+ rest, y=confirmed_p3_int,
                    mode='lines+markers',
                    name='dense network'))
    fig.write_html('forecasting_confirmed.html')


    #import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(confirmed)), y=deaths_o,
                    mode='lines+markers',
                    name='true deaths'))
    fig.add_trace(go.Scatter(x=np.arange(len(confirmed_p_int))+ observe_days+ rest, y=deaths_p_int,
                    mode='lines+markers',
                    name='xgboost'))
    fig.add_trace(go.Scatter(x=np.arange(len(confirmed_p_int))+ observe_days+ rest, y=deaths_p2_int,
                    mode='lines+markers',
                    name='linear regression'))
    fig.add_trace(go.Scatter(x=np.arange(len(confirmed_p_int))+ observe_days+ rest, y=deaths_p3_int,
                    mode='lines+markers',
                    name='dense network'))
    fig.write_html('forecasting_deaths.html')