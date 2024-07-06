# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, send_file, redirect, url_for
import sys


from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
app = Flask(__name__)

@app.route("/")
def hello():
   return render_template('input.html')

@app.route("/post",methods=['POST'])
def post():

   done = False
   value = request.form['input'].split("=")
   if(value[0]!="https://www.youtube.com/watch?v"):
      return "그거 아님다"
   else:
      srt = YouTubeTranscriptApi.get_transcript(value[1], languages = ['en'])
   text = ''
   color = '#01f01f'
   def time(y):
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
   #    print(i)
   # sys.stdout = open('subtitle.txt', 'w')
   with open("{}.srt".format(value[1]), "w", encoding='utf-8') as SRTFormatter:
      for i in range(len(srt)) :
         
         SRTFormatter.write("{}\n".format(i))
         SRTFormatter.write("{}\n".format(time(srt[i]['start']) + " --> " + time(srt[i]['start']+srt[i]['duration'])))
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
   #    text += str(i) + '\n'
   #    text += str(srt[i]['start']) + " --> " + str(srt[i]['duration'] + '\n'   
   PATH = "{}.srt".format(value[1])
   print(url_for('post'))
   
   
   
   return send_file(PATH, as_attachment=True)  
   
@app.route("/download")
def download():
   return render_template("download.html")
      

if __name__ == "__main__":
      app.run(debug=True)

#