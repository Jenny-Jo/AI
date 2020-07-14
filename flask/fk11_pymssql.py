import pymssql as ms
# print('잘 접속 되었니?')

# conn = ms.connect(server='127.0.0.1:60075', user='bit2', password='1234', database='bitdb')
conn = ms.connect(server='127.0.0.1',port=60075, user='bit2', password='1234', database='bitdb')

cursor = conn.cursor() # 정보지정
# cursor.execute('SELECT * FROM iris;')
# cursor.execute('SELECT * FROM wine;')
cursor.execute('SELECT * FROM sonar;')


row = cursor.fetchone()

while row :
    print('첫칼럼 : %s, 둘째 컬럼: %s' %(row[0], row[1]))
    row = cursor.fetchone() # 150개 열중에 한개 가져온다
    
conn.close()

print('The End')