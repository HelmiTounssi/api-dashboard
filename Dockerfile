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
# Installer les dépendances de l'application
RUN pip install --no-cache-dir gunicorn flask
# Exposer le port 5000
EXPOSE 5000

# Commande pour exécuter l'application avec Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "run:app"]