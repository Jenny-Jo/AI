file = None
try:
    file = open('file_not_found_exception','r')
except FileNotFoundError as ioe:
    print('파일 찾을 수 없음')
finally:
    file.close()