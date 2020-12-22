#%% setup file for forecasting_multi
observe_days = 30
predict_days = 1
keepdays = 150
training_countries = ['US','Spain','Belgium','China','France','Germany','United Kingdom','Italy']

# %%
def print_setups():
    print('Observe days:\t' , observe_days)
    print('Predict days:\t', predict_days)
    print('Keep days:\t', keepdays)
    print('Training Countries:')
    print(*training_countries, sep=", ")