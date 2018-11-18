import csv
import pdb 
from itertools import islice
import pandas as pd

csv.field_size_limit(2**30)


inputfile = '/Users/z/Desktop/execs_email.txt'
outputfile_106 = r'/Users/z/Desktop/execs_email_t106.txt'
outputfile_107 = r'/Users/z/Desktop/execs_email_t107.txt'
outputfile_108 = r'/Users/z/Desktop/execs_email_t108.txt'
outputfile_109 = r'/Users/z/Desktop/execs_email_t109.txt'
outputfile_110 = r'/Users/z/Desktop/execs_email_t110.txt'


start_time_108 = 975556140 #Nov 30, 2000
end_time_108 = 976130880 

start_time_109 = 976156440 #Dec 7, 2000 
end_time_109 = 976746720


start_time_107 = 974937600
end_time_107 = 975542400 #not include this


start_time_106 = 974332800
end_time_106 = 974937600 #not include this

start_time_110 = 976746720
end_time_110 =  977351520


seconds_in_week = 604800


outputfile_129 = r'/Users/z/Desktop/execs_email_t129.txt'
outputfile_130 = r'/Users/z/Desktop/execs_email_t130.txt'
outputfile_131 = r'/Users/z/Desktop/execs_email_t131.txt'
outputfile_132 = r'/Users/z/Desktop/execs_email_t132.txt'




start_time_129 = start_time_109 + (129-109)*seconds_in_week
start_time_130 = start_time_109 + (130-109)*seconds_in_week
start_time_131 = start_time_109 + (131-109)*seconds_in_week
start_time_132 = start_time_109 + (132-109)*seconds_in_week 
start_time_133 = start_time_109 + (133-109)*seconds_in_week 


end_time_129 = start_time_130
end_time_130 = start_time_131
end_time_131 = start_time_132
end_time_132 = start_time_133





df = pd.read_csv(inputfile, sep=' ')

df.Time = pd.to_numeric(df.Time, errors='coerce')

df_108 = df.loc[(df['Time'] >= start_time_108) & (df['Time'] <= end_time_108)]
df_109 = df.loc[(df['Time'] >= start_time_109) & (df['Time'] <= end_time_109)]

df_106 = df.loc[(df['Time'] >= start_time_106) & (df['Time'] <= end_time_106)]
df_107 = df.loc[(df['Time'] >= start_time_107) & (df['Time'] <= end_time_107)]

df_110 = df.loc[(df['Time'] >= start_time_110) & (df['Time'] <= end_time_110)]


df_129 = df.loc[(df['Time'] >= start_time_129) & (df['Time'] <= end_time_129)]
df_130 = df.loc[(df['Time'] >= start_time_130) & (df['Time'] <= end_time_130)]
df_131 = df.loc[(df['Time'] >= start_time_131) & (df['Time'] <= end_time_131)]
df_132 = df.loc[(df['Time'] >= start_time_132) & (df['Time'] <= end_time_132)]



df_129.to_csv(outputfile_129, header=None, index=None, sep=' ', columns = ['Source', 'Target'], mode='w')
df_130.to_csv(outputfile_130, header=None, index=None, sep=' ', columns = ['Source', 'Target'], mode='w')
df_131.to_csv(outputfile_131, header=None, index=None, sep=' ', columns = ['Source', 'Target'], mode='w')
df_132.to_csv(outputfile_132, header=None, index=None, sep=' ', columns = ['Source', 'Target'], mode='w')


#pdb.set_trace()

#df_108.to_csv(outputfile_108, header=None, index=None, sep=' ', columns = ['Source', 'Target'], mode='w')
#df_109.to_csv(outputfile_109, header=None, index=None, sep=' ', columns = ['Source', 'Target'], mode='w')
 
#df_106.to_csv(outputfile_106, header=None, index=None, sep=' ', columns = ['Source', 'Target'], mode='w')
#df_107.to_csv(outputfile_107, header=None, index=None, sep=' ', columns = ['Source', 'Target'], mode='w')
 

#df_110.to_csv(outputfile_110, header=None, index=None, sep=' ', columns = ['Source', 'Target'], mode='w')




#
#
#
#
##
#

# start_time = 910948020
# seconds_in_year = 3.154e7
# seconds_in_month = 2.628e6


# time_interval = seconds_in_week #seconds_in_month
# end_time = start_time + time_interval
# interval_end_stamps = []
# interval_start_stamps = []

# while end_time <= 1024688419:
# 	interval_start_stamps.append(start_time)
# 	end_time = int(start_time + time_interval)
# 	interval_end_stamps.append(end_time)
# 	start_time = end_time + 1

# print (interval_end_stamps)
# print (interval_start_stamps)

# pdb.set_trace()




# with open(inputfile) as ifile:
#     for row in islice(csv.reader(ifile), 5, 10):
#         print(row)

# with open(inputfile) as ifile:
# 	reader = csv.DictReader(ifile, delimiter=' ')
# 	for row in reader:
# 		print (row)


# with open(inputfile, newline='', encoding='utf-8') as ifile:
#     reader = csv.reader(ifile, delimiter='\t')
#     rows = [row for row in reader if int(row[0]) <= 315522000]

# for row in rows:
#     print (row)




 
# with open(inputfile) as fd:

# 	start_time = 910948020



#     for row in islice(csv.reader(fd), start_time, end_time):
#         print(row)

#     start_time = start_time + 1 

# with open(inputfile, newline='', encoding='utf-8') as ifile:
#     reader = csv.reader(ifile, delimiter='\t')


#     #next(reader, None) #skip line1

#     header = [u'source', u'target']
#     with open(outputfile, 'w', newline='', encoding='utf-8') as ofile:

#         writer = csv.writer(ofile)
#         writer.writerow(header)
#         for row in reader:
#                 source = row[1]
#                 target = row[2]       
#                 writer.writerow([source, target])