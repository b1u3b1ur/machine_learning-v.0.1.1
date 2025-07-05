import pandas as pd 
df=pd.read_csv('train.csv')
df2=pd.read_csv('test.csv')
ID= df2['id']

df.drop(['id','bdate','last_seen',
    'city','occupation_name',
    'career_start','career_end',
    'langs',], axis=1, inplace= True)

df['education_form'].fillna('Full-time', inplace= True)
df['occupation_type'].fillna('university', inplace = True)

def quaso(data):
    if data=='False':
        return int(9)
    else:
        return int(data)

def papus(data):
    if data=='False':
        return int(7) 
    else:
        return int(data)

df['life_main'] = df['life_main'].apply(quaso)
df['people_main'] = df['people_main'].apply(papus)

def change_educ(data):
    if data == 'Full-time':
        return 0
    elif data == 'Distance Learning':
        return 1
    elif data == 'Part-time':
        return 2
#Probar con variables de dummies

def change_ocup(data):
    if data == 'university':
        return 0
    elif data == 'work':
        return 1

def change_stat(data):
    if data == 'Alumnus (Specialist)':
        return 0
    elif data == 'Student (Specialist)':
        return 1
    elif data == "Student (Bachelor's)":
        return 2
    elif data == "Alumnus (Bachelor's)":
        return 3
    elif data == "Alumnus (Master's)":
        return 4
    elif data == "PhD":
        return 5 
    elif data == "Student (Master's)":
        return 6
    elif data == "Undergraduate applicant":
        return 7
    elif data == "Candidate of Sciences":
        return 8

df['education_form'] = df['education_form'].apply(change_educ)
df['occupation_type'] = df['occupation_type'].apply(change_ocup)
df['education_status'] = df['education_status'].apply(change_stat)
#print(df['education_form'].value_counts())

#No Borrar es importante XD
#education_form
#Full-time            0
#Distance Learning    1
#Part-time            2

#occupation_type
#university    0
#work          1

#####################################################################
df2.drop(['id','bdate','last_seen',
    'city','occupation_name',
    'career_start','career_end',
    'langs',], axis=1, inplace= True)

df2['education_form'].fillna('Full-time', inplace= True)
df2['occupation_type'].fillna('university', inplace = True)

df2['has_mobile'] = df2['has_mobile'].apply(int)
df2['followers_count'] = df2['followers_count'].apply(int)
df2['graduation'] = df2['graduation'].apply(int)
df2['relation'] = df2['relation'].apply(int)

def quaso(data):
    if data=='False':
        return int(9)
    else:
        return int(data)

def papus(data):
    if data=='False':
        return int(7) 
    else:
        return int(data)

df2['life_main'] = df2['life_main'].apply(quaso)
df2['people_main'] = df2['people_main'].apply(papus)

def change_educ(data):
    if data == 'Full-time':
        return 0
    elif data == 'Distance Learning':
        return 1
    elif data == 'Part-time':
        return 2

def change_ocup(data):
    if data == 'university':
        return 0
    elif data == 'work':
        return 1

def change_stat(data):
    if data == 'Alumnus (Specialist)':
        return 0
    elif data == 'Student (Specialist)':
        return 1
    elif data == "Student (Bachelor's)":
        return 2
    elif data == "Alumnus (Bachelor's)":
        return 3
    elif data == "Alumnus (Master's)":
        return 4
    elif data == "PhD":
        return 5 
    elif data == "Student (Master's)":
        return 6
    elif data == "Undergraduate applicant":
        return 7
    elif data == "Candidate of Sciences":
        return 8

df2['education_form'] = df2['education_form'].apply(change_educ)
df2['occupation_type'] = df2['occupation_type'].apply(change_ocup)
df2['education_status'] = df2['education_status'].apply(change_stat)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x_train= df.drop('result', axis= 1)
y_train= df['result']
x_test = df2

# x_train,x_test,y_train,y_test = train_test_split(x , y , test_size = 0.40)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors= 17)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

# percent= accuracy_score(y_test,y_pred) * 100
# print(percent)
print(df2.info())
print(y_pred)

result = pd.DataFrame({'id':ID, 'result':y_pred})
result.to_csv('pablo.csv', index = False)


