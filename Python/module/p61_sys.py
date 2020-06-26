import sys
# 이 파일과 연결된 경로
print(sys.path)
# ['f:\\Study\\Python\\module', 'C:\\Users\\bitcamp\\anaconda\\python37.zip', 'C:\\Users\\bitcamp\\anaconda\\DLLs', 'C:\\Users\\bitcamp\\anaconda\\lib', 'C:\\Users\\bitcamp\\anaconda', 'C:\\Users\\bitcamp\\anaconda\\lib\\site-packages', 'C:\\Users\\bitcamp\\anaconda\\lib\\site-packages\\win32', 'C:\\Users\\bitcamp\\anaconda\\lib\\site-packages\\win32\\lib', 'C:\\Users\\bitcamp\\anaconda\\lib\\site-packages\\Pythonwin']
# PS F:\Study>

from test_import import p62_import
p62_import.sum2()

from test_import.p62_import import sum2
sum2()