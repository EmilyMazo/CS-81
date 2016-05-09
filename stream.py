import json
from urllib import urlencode
import requests
import base64
import os
import time
import datetime

API_JSON = "keys.json"
BT_JSON = "bearer_token.json"

def filter_status(status):
    return {
        "id": status["id_str"],
        "text": status["text"],
        "hashtags": status["entities"]["hashtags"],
        "symbols": status["entities"]["symbols"],
        "urls": status["entities"]["urls"],
        "username": status["user"]["screen_name"]
    }

def convert_result(result):
    statuses = map(filter_status, result["statuses"])
    return {
        "metadata": result["search_metadata"],
        "statuses": statuses
    }

def get_bearer_token(api):
    token_credentials = base64.b64encode("%s:%s" % (api["consumer_key"], api["consumer_secret"]))

    headers = {
                "Content-type": "application/x-www-form-urlencoded;charset=UTF-8",
                'Connection': 'keep-alive',
                "Authorization": "Basic %s" % token_credentials,
                "Host": "api.twitter.com",
                "User-Agent": "Caltech CS81 Twitter Research"
            }
    req = requests.post("https://api.twitter.com/oauth2/token", data={"grant_type":"client_credentials"}, headers=headers)

    return req.json()

def search_tweets(bt, query, count=50):
    url = "https://api.twitter.com/1.1/search/tweets.json"

    data = urlencode({
        "count": count,
        "q": query + " -RT"
    })
    headers = {
        'Connection': 'keep-alive',
        "Host": "api.twitter.com",
        "User-Agent": "Caltech CS81 Twitter Research",
        "Authorization": "Bearer %s" % bt["access_token"]
    }

    req = requests.get(url, params=data, headers=headers)

    return req.json()

def get_next_results(bt, next_results):
    url = "https://api.twitter.com/1.1/search/tweets.json" + next_results
    headers = {
        'Connection': 'keep-alive',
        "Host": "api.twitter.com",
        "User-Agent": "Caltech CS81 Twitter Research",
        "Authorization": "Bearer %s" % bt["access_token"]
    }

    req = requests.get(url, headers=headers)

    return req.json()

def get_tweets(bt, query, per_page=5):
    i = 1
    cur_result = convert_result(search_tweets(bt, "trump", count=per_page))
    print "Got page %d..." % i

    all_statuses = cur_result["statuses"]
    
    while True:
        try:
            next_results = cur_result["metadata"]["next_results"]
        except:
            print "Ran out of statuses."
            break

        cur_result = convert_result(get_next_results(bt, next_results))

        i += 1
        print "Got page %d..." % i

        all_statuses.extend(cur_result["statuses"])

    return all_statuses

if __name__ == '__main__':
    if not os.path.isdir("results"):
        os.mkdir("results")

    if not os.path.isfile("keys.json"):
        print "No API keys located in keys.json."
        exit()
    API = json.load(open("keys.json"))

    bt = None
    print "Loading bearer token..."
    if not os.path.isfile(BT_JSON):
        print "Bearer token not found in %s file. Getting from Twitter API..." % BT_JSON
        bt = get_bearer_token(API)
        if bt is not None:
            with open(BT_JSON, "w") as f:
                json.dump(bt, f)
    else:
        with open(BT_JSON) as f:
            bt = json.load(f)

    ts = str(long(1000*time.time()))

    results = get_tweets(bt, "donald trump OR hillary clinton OR bernie sanders", per_page=100)

    with open("results/results-%s.json" % ts, "w") as f:
        json.dump(results, f)