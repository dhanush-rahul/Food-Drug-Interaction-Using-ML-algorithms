FROM python:3.8
ADD requirements.txt /
ADD food-drug-ssp-all.csv.csv /
RUN pip install -r /requirements.txt
ADD mlfdi.py /
CMD ["python", "./mlfdi.py"]

