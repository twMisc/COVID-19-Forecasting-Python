# COVID-19-Forecasting-Python
Predict the covid-19 confirmed and deaths using collected datas and simple models.

# How to run
Here we demonstrate how to run our program in UNIX environment. If you want to run using [Docker](https://www.docker.com/), skip to the [Docker section](#run-in-docker-with-jupyter-lab). Suppose one has python3, pip, git installed. The following installation is tested with python3.8.

## Download the data and program
First download the dataset and our program. 
```bash
$ git clone https://github.com/CSSEGISandData/COVID-19/
$ git clone https://github.com/twMisc/COVID-19-Forecasting-Python
$ cd COVID-19-Forecasting-Python
```

## Create a virtual environment (recommended)
This section is not needed but recommended. Create a virtual environment using
```bash
$ python -m venv forecasting
```
On UNIX or MacOS, run:
```bash
$ source forecasting/bin/activate
```
On Windows, run:
```shell
> forecasting\Scripts\activate.bat
```
Alternatively, you can do this using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
```bash
$ conda create --name forecasting python=3.8
$ conda activate forecasting
```

## Install the requirements
Upgrade pip and install the required packages using pip
```bash
$ python -m pip install --upgrade pip
$ pip install xgboost numpy tensorflow scikit-learn tabulate matplotlib plotly pandas
```
Alternatively, install the requirements using `requirements.txt` if you encountered into any troubles.
```bash
$ pip install -r requirements.txt
```

## Modify the parameters
Modify the file `forecasting_setup.py` as required
```python
observe_days = 30
predict_days = 1
keepdays = 150
training_countries = ['US','Spain','Belgium','China','France','Germany','United Kingdom','Italy']
```
* observe_days: input data length
* predict_days: output data length 
* keepdays: the length of data to NOT use in training
* training_countries: the countries to be used in training, can be found in `country_list.txt`

## Run it in Python interpreter
Use the function yourself in your python console. After setting up your `forecasting_setup.py`, open up your python console, 
```bash
$ python
```
In python run
```python
>>> from forecasting_multi import print_and_draw
>>> print_and_draw('Japan')
```
Replace Japan with any country in `country_list.txt`. Two files `forecasting_confirmed.png` and `forecasting_deaths.png` will be generated after running `print_and_draw(country_name)`, where `country_name` is your input. Note that two html plot files `forecasting_confirmed.html` and `forecasting_deaths.html` generated by Plotly should also be available.

## Using the pre-written run file
Run `forecasting_run.py` in console to see the results if you want a quick glance with multiple outputs:
```bash
$ python forecasting_run.py
```

## Run in Visual Studio Code interactive window
To use this, you must have a Python environment where you have installed the [Jupyter Package](https://pypi.org/project/jupyter/). You should also install the  [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) in VScode. Check out VScode's [doc](https://code.visualstudio.com/docs/python/jupyter-support-py) if you want to learn more.

Open `forecasting_run.py` in VScode, which also includes some explanation.

## Run in Jupyter Notebook
If you have [Jupyter Notebook or JupyterLab](https://jupyter.org/install.html) installed, you can use them as well.

Open `forecasting_run.ipynb` in JupyterLab or the classic Jupyter Notebook.

## Run in Docker (with Jupyter-lab)
You can ignore all steps above and run in Docker. Suppose you have docker installed where you have accessibility with non-root user, in console run 
```bash
$ docker build --pull --rm -f "Dockerfile" -t covid19forecastingpython:latest "." 
```
This should take a while.

After the build is finished, run
```bash
$ docker run -it --rm -p 8888:8888 covid19forecastingpython:latest 
```
You should see a url in your console as this form (the token will be different)
```
http://127.0.0.1:8888/lab?token=fb2007c7ca877df17d8c72ccfd4c37d94a668bff2d40be4d
```
Open this url in your web browser to use Jupyter-lab. Click on `forecasting_run.ipynb` on the left. Click the `▶▶` symbol to run. You can open `forecasting_setup.py` to modify the parameters, take a look at the [Modify the parameters](#modify-the-parameters) section.

# More details
Read https://github.com/twMisc/COVID-19-Forecasting-Python/blob/main/docs/main.pdf.

# The dataset
The dataset used is JHU CSSE COVID-19 Data, from https://github.com/CSSEGISandData/COVID-19.