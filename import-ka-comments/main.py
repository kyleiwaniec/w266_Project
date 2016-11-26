import csv
import hashlib
import json
import sys
import urllib2

FIELDS_LIST = ["video", "id", "content", "authorKaid", "answerCount", "replyCount", "date", "sumVotesIncremented", "qualityKind", "replyTo"]

wrote_header = False
wrote_videos = set()
row_count = 0

csv.field_size_limit(sys.maxsize)

try:
    with open("/usr/data/ka-comments.csv", "rb") as csvfile:
        reader = csv.reader(csvfile)
        cnt = 0
        for row in reader:
            cnt += 1
            if cnt == 1:
                wrote_header = True
                continue
            wrote_videos.add(row[0])
            row_count += 1
except Exception as e:
    print "Exception: %s" % e

print "%s" % ("california-academy-of-sciences" in wrote_videos)

print "%d videos already fetched." % len(wrote_videos)

# Download topic tree to get list of videos
f = urllib2.urlopen("http://www.khanacademy.org/api/v1/topictree")
topictree = json.loads(f.read())

def extract_videos(node, videos_set):
    if node["kind"] == "Video":
        videos_set.add(node["readable_id"])

    if "children" in node:
        [extract_videos(child, videos_set) for child in node["children"]]

    return videos_set

all_videos = extract_videos(topictree, set())
videos = list(all_videos - wrote_videos)
video_count = len(wrote_videos)

if not wrote_header:
    with open("/usr/data/ka-comments.csv", "ab") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(FIELDS_LIST)

for video in videos:
    with open("/usr/data/ka-comments.csv", "ab") as csvfile:
        writer = csv.writer(csvfile)
        cursor = None
        for page in xrange(50):
            url = ("https://www.khanacademy.org/api/internal/discussions/video/%s/"
                "questions?casing=camel&sort=1&subject=all&limit=40&page=0&"
                "lang=en" % video)
            if cursor:
                url += "&cursor=%s" % cursor

            f = urllib2.urlopen(url)
            questions = json.loads(f.read())
            print "Video %s (%d/%d) page %d: %d items." % (video, video_count, len(all_videos), page, len(questions["feedback"]))
            if len(questions["feedback"]) == 0:
                break

            for item in questions["feedback"]:
                item["video"] = video
                item["id"] = hashlib.sha1(item["expandKey"]).hexdigest()
                item["replyTo"] = ""
                writer.writerow([unicode(item[key]).encode('utf-8') for key in FIELDS_LIST])
                row_count += 1

                for subitem in item["answers"]:
                    subitem["video"] = video
                    subitem["id"] = hashlib.sha1(subitem["expandKey"]).hexdigest()
                    subitem["replyTo"] = item["id"]
                    writer.writerow([unicode(subitem[key]).encode('utf-8') for key in FIELDS_LIST])
                    row_count += 1

            cursor = questions["cursor"]

        video_count += 1

print "Done. Processed %d videos and wrote %d rows." % (video_count, row_count)
