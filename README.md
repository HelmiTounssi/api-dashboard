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

docker build -t api-dashboard .
docker run -p 5000:5000 api-dashboard:latest
