import sqlite3

conn = sqlite3.connect('test.db')

cursor = conn.cursor()

# a4로 뽑아 보면서 외워야 됨
cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT, FoodName TEXT, 
               Company TEXT, Price INTEGER)""")

sql = "DELETE FROM supermarket"
cursor.execute(sql)
# 얘네가 있어야 돌릴때마다 중복해서 추가 안됨


# data 넣자
sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (1, '과일', '자몽', '마트',1500))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (2, '음료수', '망고주스', '편의점',1000))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (3, '고기', '소고기', '하나로마트',10000))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (4, '참깨라면', '컵라면', 'cu',1200))

sql = "SELECT * FROM supermarket"
sql = "SELECT Itemno, Category, FoodName, Company, Price  FROM supermarket"

cursor.execute(sql)

rows = cursor.fetchall()

for row in rows:
    print(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + ' ' +
              str(row[3]) + ' ' + str(row[4]))

conn.commit()
conn.close()

# 내일은 플라스크와 연결해서 웹게시판 배우겠다