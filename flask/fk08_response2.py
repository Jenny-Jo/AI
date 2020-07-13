from flask import Flask, Response, make_response
app = Flask(__name__)

@app.route('/')
def response_test():
    custom_response = Response('[★] Custom Response', 200, {'Program' : 'Flask Web Application'})
    return make_response(custom_response)

@app.before_first_request
def before_first_request():
    print('[1] 앱이 가동되고 나서 첫번째 http 요청에만 응답합니다.')

# 웹서버 실행시키고 나서 프린트문 가동/ route  돌아가기 전에 먼저 돌아간다?
# before first request에 미리 저장해서 응대할 수 있다
# 내 서버에만 보임
# 첫번째 한번만 된다

@app.before_request
def before_request():
    print('[2] 매 http요청이 처리되기 전에 실행됩니다')
# 웹서버 열리고 새로고침하면 프린트됨

@app.after_request
def after_requeset(response):
    print('[3] 매 http 요청이 처리되고 나서 실행됩니다')
    return response

@app.teardown_request
def teardown_request(exception):
    print('[4] 매 http 요청의 결과가 브라우저에 응답하고 나서 호출된다')
    return exception

@app.teardown_appcontext
def teardown_appcontext(exception):
    print('[5] http요청의 application context가 종료될 때 실행된다')


if __name__ == '__main__':
    app.run(host='127.0.0.1')
# 웹서버 열자마자 프린트됨

# ms sql, sql과 flak엮어
# db 열었을 때 // 속도도 빨라/ 넘파이가 더 빨라/ 넘파이로 변환해서 하면 돼/ 디비에선 수정까지 된다
# 난 값 찾는 것 껌이다 
# sql이 빨라
###############목표가 내일 두가지 연결해서 땡겨쓰고... 끝내는게 목표다아ㅓㅓㅓㅓ라ㅓ랑러ㅏ어라얼
# 세션이나 쿠키 정리해서 과제로 쏜다!!!!!!!!!!!!!!!!
# 21번 파일
# 리스폰스와 리퀘스트
# 메일에다가!!!!!!!!!!
# 과제가 아니지만 과제같은 느낌적인 느낌  하.... 화장실...
# ms sql server download
# my sql? 유료- 오라클이 샀어
# ms sql 무료