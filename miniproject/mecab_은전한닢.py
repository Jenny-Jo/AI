import MeCab
m = MeCab.Tagger()
out = m.parse('미캅이 잘 설치되었는지 확인중입니다.')
print(out)

out = m.parse('문재인')
print(out)


