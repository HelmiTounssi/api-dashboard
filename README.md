# Flask API

This folder ('/api/') contains the Flask API (with a configuration file `config.py`)

## API calls

(Currently using custom api endpoints)

| HTTP method | API endpoint                            | Description                                              |
| ----------- | --------------------------------------- | -------------------------------------------------------- |
| GET         | clients                                 | Get a list of clients                                    |
| GET         | clients/<id>                            | Get a single client                                      |
| GET         | clients/<id>?predict=True               | Get a single client and predict risk (default threshold) |
| GET         | clients/<id>?predict=True&threshold=0.7 | Get a single client and predict risk (custom threshold)  |
| GET         | clients/<id>?explain=True               | Get a single client and explain risk (default threshold) |


## Reading data from another source

For Proof of Concept (POC), this application reads the data locally on Heroku.

For production, the code should be modified to read from an external source (with authentification, encryption and add a library for in-memory cache):
## test en local
docker build -t api-dashboard .
docker run -p 5000:5000 api-dashboard:latest
## github actions  fait le ci/cd en google cloud
 1-creer un cluster kubernetes autopilot-cluster-1  
 2- creer Artifact registery  home-credit-repo
 3- creer cloud storage Buckets  : data-model-home-credit 
 4- faire un push et github action build.xml creer le livrable docker ,push dans le registry puis instance avec ressources.yaml dans kubernetes .
 5-ajouter le token google .json et le id de projet  dans secert action dans github.

 uvicorn app.main:appfast --reload --port 5000