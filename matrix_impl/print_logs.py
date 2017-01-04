import os
import sys
from os import listdir
from os.path import isfile, join

job = sys.argv[1]
jobid = job.split("_")[1] + job.split("_")[2]
# path = "/app/hadoop-1.2.1/tmp/hduser/mapred/local/userlogs/"
path = "/usr/local/hadoop-1.2.1/logs/userlogs/"
# path = path + job + "/attempt_" + jobid
path = path + job

directories = [x[0] for x in os.walk(path)]
print(directories)
for directory in directories:
    try:
        print("opening", directory)
        with open(directory + "/stdout") as f:
            data = f.read()
            print(data)
    except IOError:
        print("IOError")
