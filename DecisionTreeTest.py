from MLFrame.DecisionTree import DecisionTree as DT
from MLFrame.StandardImport import *

Outlook = 0
Temperature =1
Humidity = 2
Wind = 3
yset=[0,0,1,1,1,0,1,0,1,1,1,1,1,0]
xput = [['Sunny',	'Hot',	'High',		'Weak'],\
	['Sunny',	'Hot',	'High',		'Strong'],\
	['Overcast',	'Hot',	'High',		'Weak'],\
	['Rain',		'Mild',	'High',		'Weak'],\
	['Rain',		'Cool',	'Normal',	'Weak'],\
	['Rain',		'Cool',	'Normal',	'Strong'],\
	['Overcast',	'Cool',	'Normal',	'Strong'],\
	['Sunny',	'Mild',	'High',		'Weak'],\
	['Sunny',	'Cool',	'Normal',	'Weak'],\
	['Rain',		'Mild',	'Normal',	'Weak'],\
	['Sunny',	'Mild',	'Normal',	'Strong'],\
	['Overcast',	'Mild',	'High',		'Strong'],\
	['Overcast',	'Hot',	'Normal',	'Weak'],\
	['Rain',		'Mild',	'High',		'Strong']]


AttrDic = {}

AttrDic[Outlook] = {1:'Sunny',3:'Overcast',5:'Rain'}
AttrDic[-1*Outlook-1] = {'Sunny':1,'Overcast':3,'Rain':5}

AttrDic[Temperature] = {2:'Hot',7:'Mild',3:'Cool'}
AttrDic[-1*Temperature-1] = {'Hot':2,'Mild':7,'Cool':3}

AttrDic[Humidity] = {9:'High',8:'Mild',5:'Normal'}
AttrDic[-1*Humidity-1] = {'High':9,'Mild':8,'Normal':5}

AttrDic[Wind] = {2:'Weak',9:'Normal',0:'Strong'}
AttrDic[-1*Wind-1] = {'Weak':2,'Normal':9,'Strong':0}

OutPatternDic = {0:'No',1:'YES'}


XData = []
for row in  xput:
	appendRow = [AttrDic[-1*index-1][row[index]] for index in range(len(row))]
	XData.append(appendRow)
	print appendRow
	print [AttrDic[index][appendRow[index]] for index in range(len(appendRow))]


dataSet = dSet.ds_2_np_patter(np.array(XData), np.array(yset))
#print dataSet.patternDic

'''............................ set up the taining situation'''
tree = DT.DTree()
treeFC = DT.ID3(dataSet)
it = treeFC(tree)
DT.cal_entropy(dataSet)
it(debug = True)
print [tree(row) for row in dataSet.x]
