import json
import datetime
import csv

# Load data from JSON files
with open('nepseindexvalue.json') as f:
    datavalue = json.load(f)

with open('nepsetime.json') as f:
    nepsetime = json.load(f)

# Open the CSV file for writing
with open('NepseData.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['Time', 'Value'])

    # Write the data rows
    for i in range(len(datavalue)):
        date = datetime.datetime.utcfromtimestamp(nepsetime[i]).strftime('%Y-%m-%d %H:%M:%S')
        print("Time : " + date + " Value : " + str(datavalue[i]))
        writer.writerow([date, datavalue[i]])

print("Data has been successfully written to NepseData.csv")
