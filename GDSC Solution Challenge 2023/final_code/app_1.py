from flask import Flask, render_template, request, send_file, redirect, url_for
import sys
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
import pandas as pd
import csv
import os, glob

app = Flask(__name__)

@app.route("/")
def hello():
	return render_template('input.html')

@app.route("/post",methods=['POST']) #input.html로부터 다운로드
def post():
	value = request.form['input'].split("=")
	if(value[0]!="https://www.youtube.com/watch?v"):
		return "그거 아님다"
	else:
		srt = YouTubeTranscriptApi.get_transcript(value[1], languages = ['en'])
	
	###################################################################################

	# hate_speech = []
	tweet = []

	for i in range(len(srt)): #csv 파일로 변환
		tweet.append(srt[i]['text'])
	# 	hate_speech.append(0)

	df = pd.DataFrame(tweet, columns=['tweet'])
	
	# df['hate_speech'] = hate_speech

	df.to_csv("{}.csv".format(value[1]), index = False)
	
	

	####################################################################################


	hate_speech = []
	text = ''
	color = '#ffffff'
	criteria = 0.4
	def time(y): # youtube_transcript_api에 있는 시간 정보를 .srt 포맷에 맞게 변환
		for i in range(len(srt)):
			startHour = int(y // 3600)
			startMin = int(y //60 - startHour*60)
			startSec = int(y % 60 //1)
			startMili = int(y % 1 * 1000)
			if startHour < 10 : 
				startHour = str(0) + str(startHour)
			if startMin < 10 : 
				startMin = str(0) + str(startMin)
			if startSec < 10 : 
				startSec = str(0) + str(startSec)
		return str(startHour) + ":" + str(startMin) + ":" + str(startSec) + "," + str(startMili)
	# for i in srt:
	# 	print(i)
	# sys.stdout = open('subtitle.txt', 'w')
	# {}.csv".format(value[1])

	###################################################
	#tensorflow로부터 파일 올때까지 대기
	# 해당 .py파일 위치
	path=os.path.dirname(os.path.realpath(__file__))

	# 파일 있을때까지 대기
	count = 1
	while not glob.glob(os.path.join(path,"result.csv")):
		print(count)
		count+=1
		time.sleep(1)

	###################################################

	f = open("result.csv", 'r') # 결과 csv 파일을 읽고 배열로 저장
	rdr = csv.reader(f)

	for line in rdr:
		hate_speech.append(line[2])
			
	

	

	f.close()
	
	with open("{}.srt".format(value[1]), "w", encoding='utf-8') as SRTFormatter: # .srt 포맷으로 변환
		
		
		
		for i in range(len(srt)) :
			
			SRTFormatter.write("{}\n".format(i))
			SRTFormatter.write("{}\n".format(time(srt[i]['start']) + " --> " + time(srt[i]['start']+srt[i]['duration'])))
			# print(hate_speech[i+1])
			if float(hate_speech[i+1]) >criteria:
				color = '#fc9d9d'
			else:
				color = '#ccddff'
			SRTFormatter.write("{}".format('<font color={}>'.format(color)))
			SRTFormatter.write("{}".format(srt[i]['text']))
			SRTFormatter.write("{}\n\n".format('</font>'))
			# print(i, end='\n')
			# print(time(srt[i]['start']) + " --> " + time(srt[i]['start']+srt[i]['duration']), end='\n')
			# print('<font color={}>'.format(color), end="" )
			# print(srt[i], end='\n')
			# print('</font>', end="\n")
			# print('', end='\n')

	# sys.stdout.close()
	for i in range(len(srt)):
		text += str(i) + ' ' + srt[i]['text'] + '\r'
	# 	text += str(i) + '\n'
	# 	text += str(srt[i]['start']) + " --> " + str(srt[i]['duration'] + '\n'	
	PATH = "{}.srt".format(value[1])
	print(url_for('post'))

	
	# send_file("whoareyoou.csv", as_attachment=True)	
	# send_file(PATH, as_attachment=True)

	##################################################################################
		
	return send_file(PATH, as_attachment=True)
@app.route("/download")
def download():
	return render_template("download.html")
		

if __name__ == "__main__":
		app.run(debug=True)

#