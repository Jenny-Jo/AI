# kg/m**2

weight = input('체중 kg')
height = input('신장 m')
bmi = float(weight)/ (float(height)**2)

print(bmi)
if bmi < 18.5 :
    print('skinny')
elif 18.5 <= bmi <25:
    print('normal')
elif 25 <= bmi < 35:
    print('a bit fat')
elif 35 <= bmi:
    print('very fat')