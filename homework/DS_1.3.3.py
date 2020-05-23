salaries_and_tenures = [(83000,8.7), (88000, 9.1),
                        (48000,0.7), (76000,6)
                        (69000, 6.5), (76000,7.5)
                        (60000, 2.5), (83000,10)
                        (48000,1.9), (63000, 4.2)

Salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_teures :
  salary_by_tenure[tenure].append(salary)
  
# average_salary_by_tenure = {
#   tenure: sum(salaries) / len(salaries)
#   for tenure, salaries in salary_by_tenure.items()

def tenure_bucket(tenure): 
  if tenure < 2:
    return "less than two"
  elif tenure < 5:
    return "between two and five"
  else:
    return " more than five"

salary_by_tenure_bucket = defaultdict(list)
 
for salary, tenure in salaries_and_tenures:
  bucket = tenure_bucket(tenure)
  salary_by_tenure_bucket[bucket].append(salary)

average_salary_by_bucket = {
  tenure_bucket : sum(salaries)/ len(salaries)
  for tenure_bucket, salaries in salary_by_tenure_bucket.items()
  
  
