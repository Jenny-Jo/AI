# 필수 : 10430번 나머지------------------------

A, B, C = map(int, input().split())

print((A+B) % C)
print(((A % C) + (B % C)) % C)
print((A * B) % C)
print(((A % C) * (B % C)) % C)

# 처음엔 divmod()로 일일이 몫, 나머지를 나누어서 할려고 했는데 너무 번거로웠음
# 어제 배운 함수를 토대로 A,B,C를 나눠주고 그대로 연산에 적용함