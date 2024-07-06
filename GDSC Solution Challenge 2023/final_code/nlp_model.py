import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import csv
import os, glob
import time


df=pd.read_csv('C:/Users/User/Desktop/hatespeech_data/labeled_data.csv')
## df=pd.read_csv('labeled_data.csv')
#C:/Users/User/Desktop/labeled_data.csv
print('총 샘플의 수 :',len(df))

data=df[['hate_speech','tweet']].copy()
data.rename(columns={'hate_speech':'v1','tweet':'v2'},inplace=True)
data['v1']=np.where(data['v1']>0,1,0)  #v1 1,2,3...로 있어서 v1이 1 이상이면 1 값으로 
data['v1'].value_counts().plot(kind='bar');

print(data.groupby('v1').size().reset_index(name='count'))


#data섞기
data = data.sample(frac=1).reset_index(drop=True)


X_data = data['v2']
y_data = data['v1']

df_len=[ len(i) for i in X_data]
max_len=max(df_len)

# max_len=max_len*2


print('메일 본문의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))


#전체 단어의 개수를 1,000개로 제한하고 정수 인코딩을 진행합니다.
vocab_size = 1000
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X_data) # 5169개의 행을 가진 X의 각 행에 토큰화를 수행
sequences = tokenizer.texts_to_sequences(X_data) # 단어를 숫자값, 인덱스로 변환하여 저장

# 상위 5개의 샘플을 출력해봅시다.

print(sequences[:5])

##################################################
#훈련 데이터와 테스트 데이터의 분리 비율 결정 (8:2)
n_of_train = int(len(sequences)*0.8)
n_of_test = int(len(sequences)-n_of_train)
print('훈련 데이터의 개수:',n_of_train)
print('테스트 데이터의 개수:',n_of_test)

#전체 데이터에서 가장 길이가 긴 메일과, 전체 메일 데이터의 길이 분포 알아보기
X_data = sequences
print('data 최대 길이:%d' %max(len(I) for I in X_data))
print('data 평균 길이:%f' %(sum(map(len,X_data))/len(X_data)))
plt.hist([len(s) for s in X_data],bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show

#전체 데이터셋의 길이를 max_len으로 맞춘다.
# max_len=172
data = pad_sequences(X_data, maxlen = max_len)
print("훈련 데이터의 크기(shape):",data.shape)

#maxlen에는 가장 긴 메일의 길이였던 172라는 숫자를 넣음.
#이는 5169개의 X_data길이를 전부 172로 바꿈.
#172보다 길이가 짧은 메일 샘플은 전부 숫자 0이 패딩되어 172의 길이를 가짐.

#이제 X_data는 5169 * 172 의 크기를 가지게 됨.
#이제 X_train과 X_test를 분리(??)


X_train = data[:int(len(data)*0.7)] #X_data 데이터 중에서 앞의 4135개의 데이터, 즉 70%만 저장
y_train = np.array(y_data[:int(len(data)*0.7)])

X_val = data[int(len(data)*0.7):int(len(data)*0.9)]
y_val = np.array(y_data[int(len(data)*0.7):int(len(data)*0.9)])

X_test = data[int(len(data)*0.9):]
y_test = np.array(y_data[int(len(data)*0.9):])


from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


cnn_lstm_model = Sequential()
# cnn_lstm_model.add(Embedding(1000, 128, input_length=172))
cnn_lstm_model.add(Embedding(1000, 128, input_length=max_len))
cnn_lstm_model.add(Dropout(0.2))
cnn_lstm_model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
cnn_lstm_model.add(MaxPooling1D(pool_size=4))
cnn_lstm_model.add(LSTM(128))
cnn_lstm_model.add(Dense(1, activation='sigmoid'))
cnn_lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
es = EarlyStopping(monitor = 'val_acc', mode = 'min', verbose = 1, patience = 3)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

# .fit: 학습시키기
history = cnn_lstm_model.fit(X_train, y_train,
            epochs = 1,
            batch_size = 64,
            validation_data = (X_val, y_val),
            callbacks = [es,mc])

print("\n 테스트 정확도: %.4f" %(cnn_lstm_model.evaluate(X_test, y_test)[1]))

pred = cnn_lstm_model.predict(X_test)


	###################################################
	#tensorflow로부터 파일 올때까지 대기
	# 해당 .py파일 위치
    #path=os.path.dirname(os.path.realpath(__file__))

	# 파일 있을때까지 대기
    #count = 1
    #while not glob.glob(os.path.join(path,"whoareyou.csv")):
        #    print(count)
        #    count+=1
        #    time.sleep(1)

	###################################################

#4. 학습시킨 모델에 새로운 csv 집어넣기
#df2=pd.read_csv('C:/Users/User/Desktop/new_data/whoareyoou2.csv')
df2=pd.read_csv('whoareyou.csv')
#df2의 복사본
df2_copy=df2.copy()

print('총 샘플의 수 :',len(df2))

data2=df2[['tweet']].copy()
data2.rename(columns={'tweet':'v2'},inplace=True)

#data섞기
data2 = data2.sample(frac=1).reset_index(drop=True)


X_data2 = data2['v2']


df2_len=[ len(i) for i in X_data2]
max_len2=max(df2_len)

# max_len2=max_len2*2


print('메일 본문의 개수: {}'.format(len(X_data2)))

#전체 단어의 개수를 1,000개로 제한하고 정수 인코딩을 진행합니다.
vocab_size = 1000
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X_data2) # 5169개의 행을 가진 X의 각 행에 토큰화를 수행
sequences2 = tokenizer.texts_to_sequences(X_data2) # 단어를 숫자값, 인덱스로 변환하여 저장

# 상위 5개의 샘플을 출력해봅시다.

print(sequences2[:5])

##############################################

#전체 데이터셋의 길이를 max_len으로 맞춘다.
# max_len= 000


X_data2 = sequences2
data2 = pad_sequences(X_data2, maxlen = max_len2)
print("훈련 데이터의 크기(shape):",data2.shape)

X_test2 = data2

pred2 = cnn_lstm_model.predict(X_test2)

#pred2의 복사본
pred3 = pred2.copy()

##################################################
#5. 결과값 넣어서 반환

df2_copy['hate speech'] = pred3
#df2_copy.to_csv('C://Users//User//Desktop//hate_results//result.csv', sep=',', na_rep='NaN')
df2_copy.to_csv('result.csv', sep=',', na_rep='NaN')
