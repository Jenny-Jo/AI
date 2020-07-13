from flask import Flask

app = Flask(__name__)

@app.route('/') # 웹주소 뒤에 / 치고 추가 입력 가능
def hello333():
    return '<h1>Hello youngsun world</h1>'

@app.route('/bit') # http://127.0.0.1:8888/bit
def hello334():
    return '<h1>Hello bit computer world</h1>'

@app.route('/Jenny')
def hello335():
    return "<h1>Hello Jenny's World</h1>"

@app.route('/bit/bitcamp') # http://127.0.0.1:8888/bit
def hello336():
    return '<h1>Hello bitcamp  world</h1>'

if __name__ =='__main__':
    app.run(host='127.0.0.1', port = 8888, debug=True)
'''
 * Serving Flask app "fk01_hello" (lazy loading)       
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Restarting with windowsapi reloader
 * Debugger is active!
 * Debugger PIN: 260-159-139
 * Running on http://127.0.0.1:8888/ (Press CTRL+C to quit)
127.0.0.1 - - [13/Jul/2020 14:27:34] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [13/Jul/2020 14:27:34] "GET /favicon.ico HTTP/1.1" 404 -
127.0.0.1 - - [13/Jul/2020 14:27:40] "GET / HTTP/1.1" 200 

200 나오면 정상적으로 돌아간다
8888번 포트로 웹 사이트를 가동시켜라
'''