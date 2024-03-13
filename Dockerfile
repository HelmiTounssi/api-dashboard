# Utiliser une image Docker basée sur Python
FROM python:3.8

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install --ignore-installed -r requirements.txt
RUN pip install seaborn
RUN pip install phik
RUN pip install prettytable
RUN pip install bayesian-optimization
RUN pip install lime
RUN pip install pydantic-settings


# Exposez le port 5000 (le port sur lequel votre application écoute)
EXPOSE 5000

# Commande pour exécuter votre application FastAPI avec Uvicorn
CMD ["uvicorn", "app.main:appfast", "--reload", "--host", "0.0.0.0", "--port", "5000"]