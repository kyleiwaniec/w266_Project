from flask import Flask, url_for, request
from flask_cors import CORS, cross_origin

import json
import pandas as pd
import MySQLdb
import math



app = Flask(__name__)
CORS(app)


db = MySQLdb.connect(host="xxx",
                     user="xxx",
                     passwd="xxx",
                     db="xxx")

# you must create a Cursor object. It will let
# you execute all the queries you need
cur = db.cursor()

# Use all the SQL you like
cur.execute('''
			SELECT a.*, m.role_id 
            FROM iq4.activity a 
            join membership m on m.user_id=a.user_id 
            WHERE (a.grouping_id=354 or a.grouping_id=355 or a.grouping_id=342 or a.grouping_id=341)
            and a.item_type != 'L' and a.item_type != 'F'
            and (m.role_id=51 or m.role_id=52 or m.role_id=53 or m.role_id=54)
            GROUP BY a.id;
            '''
            )

results = []


num_fields = len(cur.description)
field_names = [i[0] for i in cur.description]
# print field_names

for row in cur.fetchall():
    results.append(row)
    # query_result = json.loads(unicode(row[3], errors='ignore'))

db.close()
df = pd.DataFrame(results)
df.columns = field_names


# Make list of unique related_to_item (topic ids) ids
topic_ids = [int(i) for i in df['related_item_id'].unique() if math.isnan(i) == False]
print sorted(topic_ids)


json_result = df.to_json()


@app.route("/hello")
@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"

@app.route('/')
@cross_origin()
def api_root():
	return json_result

# Get all topic ids (related_item_id)
@app.route('/topics')
@cross_origin()
def api_topics():
	return json.dumps(sorted(topic_ids))

# get a topic given the id (discussion)
@app.route('/topic/<topic_id>', methods=['GET', 'POST'])
@cross_origin()
def api_topic(topic_id):
	topic = df[df['item_id'] == float(topic_id)]
	return topic.to_json(orient="records")

# get all comments for topic (all comments for the discussion)
@app.route('/comments/<topic_id>')
@cross_origin()
def api_comments(topic_id):
	comments = df.loc[df['related_item_id'] == float(topic_id)]
	return comments.to_json(orient="records")



if __name__ == '__main__':
	app.run(host='scoring.local',port='8080')
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
