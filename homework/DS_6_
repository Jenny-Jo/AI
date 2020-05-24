#6.2 조건부 확률
import enum, random
class Kid(enum.Enum):
    Boy = 0
    Girl = 1

def random_kid() -> Kid:
    return random.choice([Kid.Boy, Kid.Girl])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == Kid.Girl:
        older_girl += 1
    if older == Kid.Girl and younger == Kid.Girl:
        both_girls += 1
    if older == Kid.Girl or younger == Kid.Girl:
        either_girl += 1
print("P(both | older):", both_girls / older_girl)
print("P(both | either): ", both_girls / either_girl)

#6.5 연속분포
def uniform_cdf(x: float) ->float:
    if x <0: return 0
    elif x <1: return x
    else: return 1

# 6.6 정규분포
import math
SQRT_TWO_PI = math.sqrt(2*math.pi)

def normal_pdf(x: float, mu:float = 0, sigma: float = 1) -> float: 
    return (math.exp(-(x-mu) **2/2/sigma**2)/(SQRT_TWO_PI * sigma))

import matplotlib.pyplot as plt
xs = [x/10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_pdf(x,sigma=1) for x in xs], '-', label = 'mu=0, sigma=1')
plt.plot(xs, [normal_pdf(x,sigma=2) for x in xs], '--', label= 'mu=0, sigma=2')
plt.plot(xs, [normal_pdf(x,sigma=0.5) for x in xs], ':', label= 'mu=0, sigma=0.5')
plt.plot(xs, [normal_pdf(x,mu=-1) for x in xs], '-', label= 'mu=1, sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()

# 누적 분포 함수
def normal_cdf(x: float, mu: float=0, sigma: float = 1) ->float:
    return (1+ math.erf((x- mu) / math.sqrt(2) / sigma))/2

xs = [x / 10.0 for x in range(-50,50)]
plt.plot(xs,[normal_cdf(x, sigma =1) for x in xs], '-', label = 'mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x, sigma =2) for x in xs], '--', label = 'mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x, sigma =0.5) for x in xs], '-', label = 'mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x, sigma =1) for x in xs], '-', label = 'mu=-1,sigma=1')
plt.legend(loc=4) 
plt.title("Various Normal cdfs")
plt.show()

def inverse_normal_cdf( p : float,
                        mu: float = 0,
                        sigma: float = 1,
                        tolerance: float = 0.00001) -> float:

if mu != 0 or sigma != 1 :
    return mu+ sigma * inverse_normal_cdf(p, tolerance = tolerance)

low_z = -10.0
hi_z = 10.0
while hi_z -low_z > tolerance :
    mid_z = (low_z + hi_z) /2
    mid_p = normal_cdf(mid_z)
    if mid_p < p:
        low_z = mid_z
    else:
        hi_z = mid_z
return mid_z

def bernoulli_trial(p:float) -> int:
    return 1 if random.random() < p else 0
def binomial(n: int, p:float) ->int:
    return sum (bernoulli_trial(p) for _ in range(n))

from collections import Counter
def binomial_histogram(p: float, n: int, num_points: int) ->None:
    data = [binomial(n,p) for _ in range(num_points)]

    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()]),
            [v / num_points for v in histogram.values()],
            0.8,
            color = '0.75'
mu = p * n 
sigma = math.sqrt(n* p * (1 - p))

xs = range(min(date), max(data) + 1)
ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)]
    for i in xs]
plt.plot(xs, ys)
plt.title("Binomial Distribution vs. Normal Approximation")
plt.show()
