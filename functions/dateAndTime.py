import time
from daytime import datetime
import numpy as np

string = '2022-04-04'
timeStamp = (time.mktime(datetime.datetime.strptime(string, "%Y-%m-%d").timetuple()))
readable = datetime.datetime.fromtimestamp(timeStamp).isoweekday()

print(readable)

timeStamp = (time.mktime(datetime.datetime.strptime(testvalue, "%Y-%m-%d").timetuple()))
readable = datetime.datetime.fromtimestamp(timeStamp).isoweekday()
print(readable)

