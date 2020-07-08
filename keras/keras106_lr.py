weight = 0.5           # 초기 가중치
input = 0.5            # x
goal_prediction = 0.8  # y

lr = 0.001

for iteration in range(1101):                           # 0.5를 넣어서 0.8을 찾아가는 과정
    prediction = input*weight                           # y = w*x 
    error = (prediction - goal_prediction)**2           # loss

    print('Error : ' + str(error)+'\tPrediction : '+str(prediction))

    up_prediction = input *(weight + lr)                # weight = gradient : -경사 올림
    up_error = (goal_prediction - up_prediction)**2     # loss

    down_predicrion = input*(weight - lr)               # weight = gradient : +경사 내림
    down_error = (goal_prediction - down_predicrion)**2 # loss

    if(down_error < up_error):                          
        weight = weight - lr                            

    if(down_error > up_error):                          
        weight = weight + lr                           