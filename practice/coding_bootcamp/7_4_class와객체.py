class Human:
    age = 0
    last_name = ''
    first_name=''
    height = 0.0
    weight=0.0
    
    def get_bmi(self):
        return (self.weight) / (self.height**2)

younghee = Human()
younghee.age=35
younghee.last_name='이'
younghee.first_name='영희'
younghee.height= 1.7
younghee.weight= 68.2

bmi = younghee.get_bmi()
print(bmi)

if (younghee.age>=35 and younghee.first_name=='영희'):
    print('선택된 사람은 ' + younghee.last_name+ younghee.first_name)