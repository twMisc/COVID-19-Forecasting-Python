FROM python:3.8.7
COPY . /COVID-19-Forecasting-Python
RUN apt-get update \
&& apt-get install git 
RUN git clone https://github.com/CSSEGISandData/COVID-19/
WORKDIR /COVID-19-Forecasting-Python
RUN pip install -r requirements.txt
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]