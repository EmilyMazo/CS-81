# CS-81

//To run stream.py, first run `pip install -r requirements.txt` to install the `requests` library.

//Then create a file with API keys, `keys.json`. Everything else will work from there; just change the query parameters.

First, run `python parse_crowdflower_JSON_tweets.py `
Next, `./runTagger.sh tweets2.txt > parsed_tweets.txt`
Next, `python political_data_train_model.py`.
