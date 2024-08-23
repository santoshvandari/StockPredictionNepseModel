import json,datetime



with open('nepseindexvalue.json') as f:
    datavalue = json.load(f)


with open('nepsetime.json') as f:
    nepsetime = json.load(f)


with open('NepseData.csv', 'w') as f:
    f.write('Time,Value\n')
    for i in range(10):
        date = datetime.datetime.utcfromtimestamp(nepsetime[i]).strftime('%Y-%m-%d %H:%M:%S')
        f.write(str(date) + ',' + str(datavalue[i]) + '\n')