FROM python:2.7

WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install -r requirements.txt

COPY app.py /usr/src/app/
COPY index.html /usr/src/app/static/
COPY css/* /usr/src/app/static/css/
COPY js/* /usr/src/app/static/js/
COPY views/* /usr/src/app/static/views/

CMD ["env", "python", "/usr/src/app/app.py"]
