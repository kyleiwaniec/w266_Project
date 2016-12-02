from flask import Flask, url_for, request, send_from_directory

import json
import pandas as pd
import math



app = Flask(__name__)

df = pd.read_csv("/usr/data/iQ4-Activity.csv")

# Make list of unique related_to_item (topic ids) ids
topic_ids = [int(i) for i in df['related_item_id'].unique() if math.isnan(i) == False]
json_result = df.to_json()

@app.route('/')
@app.route('/dashboard')
@app.route('/criteria')
def root():
    return app.send_static_file('index.html')

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('/usr/src/app/static/js', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('/usr/src/app/static/css', path)

@app.route('/views/<path:path>')
def send_views(path):
    return send_from_directory('/usr/src/app/static/views', path)

@app.route("/api/hello")
def helloWorld():
  return "Hello, cross-origin-world!"

@app.route('/api/')
def api_root():
  return json_result

# Get all topic ids
@app.route('/api/topics')
def api_topics():
  return json.dumps(sorted(topic_ids))

# get a topic given the id
@app.route('/api/topic/<topic_id>', methods=['GET', 'POST'])
def api_topic(topic_id):
  topic = df[df['item_id'] == float(topic_id)]
  return topic.to_json(orient="records")

# get all comments for topic
@app.route('/api/comments/<topic_id>')
def api_comments(topic_id):
  comments = df.loc[df['related_item_id'] == float(topic_id)]
  return comments.to_json(orient="records")



if __name__ == '__main__':
  app.run(port='8085', host="0.0.0.0")
  # app.run(host='ec2-23-21-150-163.compute-1.amazonaws.com',port='8080')

'''
NOTES:

I got as far as installing and running the flask app, and the angular front-end in AWS on our company instance.
To access the app one has to log in through iQ4 which stores a session cookie.

My last remaining issue is with the fact that iQ4 forces an Https redirect, but the flask API is only accessible
via http. It listens on port 8080. I tried adding port 8080 to our load balancer on AWS (which is required for the free SSL cert)
but I don't really understand how it works. The request ends up timing out.

I can't call the API over http while the front-end is served over https because it throws an Access-Origin error.

In my local environment I have everything on http and it works as expected.

'''