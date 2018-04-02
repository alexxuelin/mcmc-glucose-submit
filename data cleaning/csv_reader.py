import csv
import re
import datetime

results = []
john = [0,0]
# yo = []
count1 = 0
count2 = 0
count3 = 0
count = 300
min_datapoints = 288
counter = 288
results.append(['Event Subtype','Timestamp (YYYY-MM-DDThh:mm:ss)','Glucose Value (mg/dL)', 'Glucose Rate of Change (mg/dL/min)', 'Day of the Week', 'Offset', 'Indicator'])
with open("CLARITY_Export_Menelly_Steven_2018-03-28_003504.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ') # change contents to floats
    for row in reader: # each row is a list
    	row = (row[0]).split(",")
    	temp = re.split('T|:|-',row[1])
    	# count = len(row)
    	# john = row
    	# count = count + 1
    	if len(temp) > 2 and john != [temp[1],temp[2]] and counter == 288:
    		counter = 1
    	if len(temp) > 2 and (len(row) == 14) and datetime.date(int(temp[0]),int(temp[1]),int(temp[2])).weekday() in range(0,4) and counter < min_datapoints:
    		if john != [temp[1],temp[2]]:
    			john = [temp[1],temp[2]]
    			counter = 1
    		else:
    			counter = counter + 1
    		day_of_week = "N/A"
    		if datetime.date(int(temp[0]),int(temp[1]),int(temp[2])).weekday() == 0:
    			day_of_week = "Monday"
    		elif datetime.date(int(temp[0]),int(temp[1]),int(temp[2])).weekday() == 1:
    			day_of_week = "Tuesday"
    		elif datetime.date(int(temp[0]),int(temp[1]),int(temp[2])).weekday() == 2:
    			day_of_week = "Wednesday"
    		elif datetime.date(int(temp[0]),int(temp[1]),int(temp[2])).weekday() == 3:
    			day_of_week = "Thursday"
    		# if [temp[1], temp[2]] not in yo:
    		# 	yo.append([temp[1], temp[2]])
    		if int(row[7]) >= 200:
    			indicator = 1
    		else:
    			indicator = 0
        	results.append([row[3],row[1],row[7],row[11],day_of_week, counter, indicator])
        	# if int(temp[1]) == 3 and int(temp[2]) == 26:
        	# 	count1 = count1 + 1
        	# if int(temp[1]) == 3 and int(temp[2]) == 19:
        	# 	count2 = count2 + 1
        	# # if int(temp[1]) == 3 and int(temp[2]) == 12:
        	# # 	count3 = count3 + 1
        	# if int(temp[1]) == 2 and int(temp[2]) == 14:
        	# 	count3 = count3 + 1

# bob = []
# for i in yo:
# 	poss_total = len([x for x in results if [x[1][1],x[1][2]] == i])
# 	if poss_total < count:
# 		bob = i
# 		count = poss_total

# print count1
# print count2
# print count3
# print results[120]
# print john
# print count
# print bob


with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(results)
