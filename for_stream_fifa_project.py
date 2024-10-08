import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint


df=pd.read_csv('international_matches.csv', encoding='utf-8')
df.drop(['home_team_goalkeeper_score'], axis=1, inplace=True)
df.drop(['away_team_goalkeeper_score'], axis=1, inplace=True)
df.drop(['home_team_mean_defense_score'], axis=1, inplace=True)
df.drop(['home_team_mean_offense_score'], axis=1, inplace=True)
df.drop(['home_team_mean_midfield_score'], axis=1, inplace=True)
df.drop(['away_team_mean_defense_score'], axis=1, inplace=True)
df.drop(['away_team_mean_offense_score'], axis=1, inplace=True)
df.drop(['away_team_mean_midfield_score'], axis=1, inplace=True)
df.drop(['country'], axis = 1, inplace=True)
df.drop(['city'], axis = 1, inplace=True)
df = df.drop(df[df['tournament'] == 'Friendly'].index)
df.drop(['neutral_location'], axis=1, inplace=True)
idx= df.isnull().index
# df.drop(idx)
df.dropna(inplace=True)




label_encoder = LabelEncoder()

#Honduras	away에만 있음
#Benin	away에만 있음
#Moldova home에만 있음
#Togo home에만 있음
df['home_team_numeric'] = label_encoder.fit_transform(df['home_team'])
df['away_team_numeric'] = label_encoder.fit_transform(df['away_team'])
df['home_team_continent_numeric'] = label_encoder.fit_transform(df['home_team_continent'])
df['away_team_continent_numeric'] = label_encoder.fit_transform(df['away_team_continent'])
df['tournament_numeric'] = label_encoder.fit_transform(df['tournament'])
df = df.replace({'shoot_out': {'Yes': True, 'No': False}})
df[df['away_team']=='Iceland']

data=df.drop(['home_team_result','date','home_team','away_team','home_team_continent','away_team_continent','tournament','home_team_score','away_team_score','tournament_numeric','shoot_out'], axis =1)
target=df['home_team_result']
data[data['home_team_numeric'] == 29]

kn = KNeighborsClassifier()
kn.fit(data, target)


kr_jp=pd.DataFrame({'home_team_fifa_rank':[23],'away_team_fifa_rank':[16],'home_team_total_fifa_points':[1572],'away_team_total_fifa_points':[1639],'home_team_numeric':[102],'away_team_numeric':[97],'home_team_continent_numeric':[1],'away_team_continent_numeric':[1]})
Ger_hun=pd.DataFrame({'home_team_fifa_rank':[13],'away_team_fifa_rank':[32],'home_team_total_fifa_points':[1692],'away_team_total_fifa_points':[1511],'home_team_numeric':[75],'away_team_numeric':[88],'home_team_continent_numeric':[2],'away_team_continent_numeric':[2]})
Fr_It=pd.DataFrame({'home_team_fifa_rank':[2],'away_team_fifa_rank':[10],'home_team_total_fifa_points':[1851],'away_team_total_fifa_points':[1726],'home_team_numeric':[71],'away_team_numeric':[95],'home_team_continent_numeric':[2],'away_team_continent_numeric':[2]})
Den_Ser=pd.DataFrame({'home_team_fifa_rank':[20],'away_team_fifa_rank':[35],'home_team_total_fifa_points':[1621],'away_team_total_fifa_points':[1505],'home_team_numeric':[71],'away_team_numeric':[164],'home_team_continent_numeric':[2],'away_team_continent_numeric':[2]})
Tur_Ice=pd.DataFrame({'home_team_fifa_rank':[26],'away_team_fifa_rank':[71],'home_team_total_fifa_points':[1538],'away_team_total_fifa_points':[1355],'home_team_numeric':[194],'away_team_numeric':[90],'home_team_continent_numeric':[2],'away_team_continent_numeric':[2]})
print("한국vs일본:한국",kn.predict(kr_jp))
print("독일vs헝가리:독일",kn.predict(Ger_hun))
print("프랑스vs이탈리아:프랑스",kn.predict(Fr_It))
print("덴마크vs세르비아:덴마크",kn.predict(Den_Ser))
print("터키vs아이슬란드:터키",kn.predict(Tur_Ice))
kn.score(data,target)

rf = RandomForestClassifier(random_state=40)
rf.fit(data,target)

print("한국vs일본:한국",rf.predict(kr_jp))
print("독일vs헝가리:독일",rf.predict(Ger_hun))
print("프랑스vs이탈리아:프랑스",rf.predict(Fr_It))
print("덴마크vs세르비아:덴마크",rf.predict(Den_Ser))
print("터키vs아이슬란드:터키",rf.predict(Tur_Ice))

rf.score(data,target)

checkpoint = ModelCheckpoint('best_model_match_prediction.keras', save_best_only=True, monitor='val_loss',mode='min')
encoder=LabelEncoder()
target_encoded = encoder.fit_transform(target)
model=Sequential([
    Dense(128, activation='relu', input_shape=(data.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation = 'softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(data, target_encoded, epochs=200, batch_size=20, validation_split=0.2, callbacks=[checkpoint])

print(model.predict(kr_jp))
print(model.predict(Den_Ser))
print(model.predict(Ger_hun))
print(model.predict(Fr_It))
print("Encoded Target:", target_encoded)