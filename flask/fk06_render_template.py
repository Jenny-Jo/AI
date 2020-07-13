from flask import Flask, render_template
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

# html에는 index, user 필요
# http://127.0.0.1:5000/user/아무거나 < 이게 user>name 에 들어간다