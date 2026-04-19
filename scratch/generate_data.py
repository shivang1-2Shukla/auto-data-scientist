import pandas as pd
import numpy as np
import os

np.random.seed(42)

n_samples = 1000

# Features
age = np.random.randint(22, 60, n_samples)
years_at_company = np.random.randint(0, 15, n_samples)
salary = np.random.randint(40000, 150000, n_samples)
job_satisfaction = np.random.randint(1, 11, n_samples) # 1 to 10
departments = np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing'], n_samples)

# Logic for Will_Quit
# High chance of quitting if: satisfaction < 5 OR (salary < 60k and years_at_company > 3)
will_quit = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    prob_quit = 0.1 # base probability
    
    if job_satisfaction[i] < 4:
        prob_quit += 0.5
    if salary[i] < 60000 and years_at_company[i] > 3:
        prob_quit += 0.3
    if age[i] < 28:
        prob_quit += 0.1 # younger people jump jobs more
        
    will_quit[i] = 1 if np.random.rand() < prob_quit else 0

df = pd.DataFrame({
    'Age': age,
    'Years_At_Company': years_at_company,
    'Salary': salary,
    'Job_Satisfaction_Score': job_satisfaction,
    'Department': departments,
    'Will_Quit': will_quit
})

os.makedirs('d:/auto-data-scientist/data', exist_ok=True)
df.to_csv('d:/auto-data-scientist/data/employee_attrition.csv', index=False)
print("Dataset generated at data/employee_attrition.csv")
