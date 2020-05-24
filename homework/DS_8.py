# 8.1 경사하강법
from scratch.linear_algebra import Vector, dot

def sum_of_squares(v: Vector) -> float:
    return dot(v,v)

# 8.2 gradient  계산하기
from typing import Callable
def difference_quotient(f: Callable[[float], float],   
                        x: float,
                        h:float) -> float:
    return(f(x + h) - f(x)) / h 

def square(x: float) -> float:
    return x*x

def derivative(x: float) -> float :
    return 2 *x

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h= 0.001) for x in xs]

import matplotlib.pyplot as plt
plt.title("Actual derivatives vs. Estimates")
plt.plot(xs, actuals, 'rx', label='Actual')
plt.plot(xs, estimates, 'b+', label = 'Estimate')
plt.legend(loc=9)
plt.show()

def partial_difference_quotient(f: callable[[Vector],float],
                                v: Vector,
                                i: int,
                                h: float) ->float:
    w = [v_j + (h if j ==i else 0)
        for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
                      v : Vector ,
                      h : float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]

# 8.3 gradient 적용하기
import random
from scratch.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size : float) ->Vecotr:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v,step)

def sum_of_squares_gradient(v:Vector) -> Vector:
    return [2*v_i for v_i in v]

v = [random.uniform(-10,10) for i in range(3)]

for epoch in range(1000) :
    grad = sum_of_squares_gradient(v)
    v = gradient_step(v, grad, -0.01)
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001

#8.5  경사하강법으로 모델 학습
inputs = [(x, 20*x + 5) for x in range(-50, 50)]

def linear_gradient(x: float, y:float, theta : Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (predicted - y)
    squared_error = error **2
    grad = [2 * error *x, 2 *b error]
    return grad

from scratch.linear_algebra import vector_mean

theta = [random.uniform(-1,1), random.uniform(-1, 1)]
learning_rate = 0.001

for epoch in range(5000):
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1, "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"

# 8.6  미니배치와 SGD

from typing import TypeVar, List, Iterator
T = TypeVar('T')

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool= True) -> Iterator[List[T]] :
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle: random.shuffle(batch_starts)

    for start in batch_size:
        end = start + batch_sixe
        yield dataset[start:end]

theta = [random.uniform (-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)
slope, intercept = theta
assert 19.9  < slope < 20.1,
assert 4.9 <  intercept < 5.1

theta = [radnom.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(100) :
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)
slope, intercept = theta

assert 19.9 < slope < 20.1, "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
