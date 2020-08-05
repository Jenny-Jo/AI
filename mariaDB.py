# -*- coding:utf-8 -*-

import mysql.connector
# mysql connector를 사용하기 위한 모듈 선언



# mysql connection을 선언한다. 
# 파라미터는 host는 접속 주소, user는 ID, passwd는 패스워드, database는 접속할 데이터 베이스이다.
dbconn = mysql.connector.connect(host="127.0.0.1", user="root", passwd="dkdlxl", database="test");


# 검색을 할 경우 사용되는 함수.
def select(query, bufferd=True):

    # 전역에 선언되어 있는 connection을 가져온다.
    global dbconn;

    # 커서를 취득한다. (bufferd는 내부의 검색 버퍼를 사용하는데(connection 리소스를 아끼기 위한 값)
    # 검색되어 있는 값은 메모리에 두고 재 요청이 올 경우, 디비에 검색을 하지 않고 메모리의 값이 리턴 됨, 
    # 특히 대용량 페이징을 사용할 때 사용하면 좋음.
    cursor = dbconn.cursor(buffered=bufferd);

    # 쿼리를 실행한다.
    cursor.execute(query);


    # 검색 결과를 확인하기 위해서는 커서를 리턴해야 한다.
    # cursor.fetchall()로 결과를 리스트로 내보낼 수도 있다.
    # 그러나 결과가 대용량일 경우 fetchall로 대량의 값을 메모리에 넣으면 느려질 수 있다.
    return cursor;


# DML(Data Manipulation Language)의 insert, update, delete를 처리하는 함수
def merge(query, values, bufferd=True):


    # 전역에 선언되어 있는 connection을 가져온다.
    global dbconn;

    try:
        # 커서를 취득한다.
        cursor = dbconn.cursor(buffered=bufferd);


        # 쿼리를 실행한다. values는 query 값에 있는 sql query식의 바인딩 값이다.
        # 문자열 포멧팅으로 설정된다. values는 튜플 값으로 입력된다.
        cursor.execute(query, values);


        # 쿼리를 커밋한다.
        dbconn.commit();
    except Exception as e:


        # 에러가 발생하면 쿼리를 롤백한다.
        dbconn.rollback();
        raise e;

# DML(Data Manipulation Language)의 insert, update, delete를 대랑 처리하는 함수
def merge_bulk(query, values, bufferd=True):


    # 전역에 선언되어 있는 connection을 가져온다.
    global dbconn;

    try:
        # 커서를 취득한다.
        cursor = dbconn.cursor(buffered=bufferd);


        # 쿼리를 실행한다. values는 query 값에 있는 sql query식의 바인딩 값이다.
        # 문자열 포멧팅으로 설정된다. values는 리스트 튜플 값으로 입력된다.
        cursor.executemany(query, values);

        # 쿼리를 커밋한다.
        dbconn.commit();

    except Exception as e:
        # 에러가 발생하면 쿼리를 롤백한다.
        dbconn.rollback();
        raise e;


# DML이외의 쿼리를 실행하는 함수.
def execute(query, bufferd=True):


    # 전역에 선언되어 있는 connection을 가져온다.
    global dbconn;
    try:

        # 커서를 취득한다.
        cursor = dbconn.cursor(buffered=bufferd);

        # 쿼리를 실행한다.
        cursor.execute(query);

        # 쿼리를 커밋한다.
        dbconn.commit();


    except Exception as e:
        # 에러가 발생하면 쿼리를 롤백한다.
        dbconn.rollback();
        raise e;
try:
    # 테이블 PythonTable를 삭제한다.(이전 실행 중 에러가 발생하면 테이블을 지우고 시작한다. 다음 CREATE에서 에러난다.)
    #execute("DROP TABLE PythonTable");
    # 테이블 PythonTable를 생성한다.
    execute("""
CREATE TABLE PythonTable (
idx int auto_increment primary key,
data1 varchar(255),
data2 varchar(255)
)
""");


    # 테이블 값을 대량 insert하기 위한 리스트 튜플 값
    values = [('data1', 'test1'),
    ('data2', 'test2'),
    ('data3', 'test3'),
    ('data4', 'test4'),
    ('data5', 'test5')];


    # 데이터를 대량 입력한다.
    merge_bulk("INSERT INTO PythonTable (data1, data2) VALUES (%s, %s)", values);

    # PythonTable를 출력한다.
    for row in select("SELECT * FROM PythonTable"):
        print(row);

    # data1의 데이터 값을 update1로 수정한다.
    merge("UPDATE PythonTable set data2=%s where data1=%s", ('update1','data1'));

    # PythonTable를 출력한다.
    for row in select("SELECT * FROM PythonTable"):
        print(row);


    # 테이블 PythonTable를 삭제한다.
    execute("DROP TABLE PythonTable");
except Exception as e:
    print(e);
finally:

    # connection을 다 사용하면 반드시 connection 리소스를 닫는다.
    dbconn.close();



