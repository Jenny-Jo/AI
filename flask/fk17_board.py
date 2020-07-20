# 'F:/Study/AI/data/wanggun/wanggun.db'

from flask import Flask, render_template, request
import sqlite3
from requests.api import request

app = Flask (__name__)

# database

conn = sqlite3.connect('F:/Study/AI/data/wanggun/wanggun.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM general')
print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect('F:/Study/AI/data/wanggun/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general')
    rows = c.fetchall() # one이면 loop문 써야해
    return render_template('board_index.html', rows=rows)


@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('F:/Study/AI/data/wanggun/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id = '+str(id))
    rows = c.fetchall()
    return render_template('board_modi.html', rows=rows)

'''# /modi
@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('F:/Study/AI/data/wanggun/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id ='+str(id))   # id 를 설정해주면 설정한 id에 대한 값만 출력. 
    rows = c.fetchall();
    return render_template('board_modi.html', rows = rows)'''


app.run(host='127.0.0.1', port=5001, debug=False)


'''
@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.methods == 'POST':
        try: 
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect('F:/Study/AI/data/wanggun/wanggun.db') as conn:
                cur = conn.cursor()
                cur.execute(" UPDATE general  SET  war=" + str(war) + "WHERE id ="+str(id))
                conn.commit()
                msg = ' 정상 입력됨'
        except: 
            conn.rollback()
            msg = '입력과정에서 에러발생'
        finally:
            return render_template("board_result.html", msg = msg)
            conn.close()
            '''
                