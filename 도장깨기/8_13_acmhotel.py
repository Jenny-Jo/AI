'''
https://www.acmicpc.net/problem/10250
'''
# 호텔 방 배정 문제 -미완성
import sys

# xx = n//h 
# yy = n%h
t,h,w,n = map(int,sys.stdin.readline().split())

def find_room(h,w,n):
    
    xx, yy = divmod(n, h)
    if xx < 1 :
        xx = '01'
        room_num = str(yy)+str(xx)
    elif xx >=1 :
        if w < 10:
            xx = '0'+str(xx+1)
            room_num = str(yy)+str(xx)
        else:
            room_num = str(yy)+str(xx)
    return print(room_num)


for i in range(1,t+1):
    find_room(h,w,n)
    


