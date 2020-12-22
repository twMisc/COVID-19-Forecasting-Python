# COVID-19-Forecasting-Python
Predict the covid-19 confirmed and deaths using collected datas and simple models.

# How to run
Here we demonstrate how to run our program in UNIX environment. Suppose one has python3, pip, git installed. First download the dataset and our program. 
```bash
$ git clone https://github.com/CSSEGISandData/COVID-19/
$ git clone https://github.com/twMisc/COVID-19-Forecasting-Python
$ cd COVID-19-Forecasting-Python
```

Install the required packages in Python.
```bash
$ pip install xgboost numpy tensorflow sklearn tabulate matplotlib plotly pandas
```

Then modify the file `forecasting_setup.py` as required
```python
observe_days = 30
predict_days = 1
keepdays = 150
training_countries = ['US','Spain','Belgium','China','France','Germany','United Kingdom','Italy']
```

* observe_days: input data length
* predict_days: output data length 
* keepdays: the length of data to NOT use in training
* training_countries: the countries to use in training, can be found in `country_list.txt`

## Run in console
Run `forecasting_run.py` in console
```bash
$ python forecasting_run.py
```

## Run in Visual Studio Code interactive
Simply open `forecasting_run.py` in Vscode. Which also includes some explanation.

## Run in Jupyter notebook
Open `forecasting_run.ipynb` in Jupyter lab or the classic Jupyter notebook.

## Run it in python interpreter
After setting up your `forecasting_setup.py`, in python run
```python
>>> from forecasting_multi import print_and_draw
>>> print_and_draw('Japan')
```
Replace Japan with any country in `country_list.txt`. Note that two pyplot html files should also be generated.

# More details
Read `./pdf/main.pdf`.

# The dataset
The dataset used is JHU CSSE COVID-19 Data, from https://github.com/CSSEGISandData/COVID-19.