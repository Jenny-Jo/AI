salaries_and_tenures = [(83000,8.7), (88000, 9.1),
                        (48000,0.7), (76000,6)
                        (69000, 6.5), (76000,7.5)
                        (60000, 2.5), (83000,10)
                        (48000,1.9), (63000, 4.2)

Salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_teures :
  salary_by_tenure[tenure].append(salary)
  
average_salary_by_tenure = {
  tenure: sum(salaries) / len(salaries)
  for tenure, salaries in salary_by_tenure.itesm()
  
  
