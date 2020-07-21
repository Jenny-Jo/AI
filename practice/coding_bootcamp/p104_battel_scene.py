my_hp = 15
slime_hp = 8
index = 0
import random

while slime_hp > 0 and my_hp >0:
    attack = random.randint(1,7)
    if index % 2 == 0 :
        print('몬스터에게'+str(attack)+'의 피해를 입혔다')
        slime_hp -= attack
    else:
        print('주인공에게'+str(attack)+'의 피해를 입혔다')
        my_hp -= attack
    index +=1
    
if my_hp > 0 :
    print('몬스터를 격파하였다')
else:
    print('주인공이 죽었다')