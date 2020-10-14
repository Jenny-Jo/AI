# 1. base case- solve simplest case of your problem
# 2. assumtion - that your function already exists and works for simpler problems.
# 3. use the solved simpler problems to solve your current problem.

def reverse(str):
  # base case
  if len(str) <2 :
    return str
  # recursive case
  else:
    return reverse(str[1:]) + str[0]

reverse("I'am cool")