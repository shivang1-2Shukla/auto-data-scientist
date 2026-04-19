import pandas as pd
import numpy as np
import os

np.random.seed(99)

n_samples = 100

# Features (Same as before, but no Will_Quit column)
age = np.random.randint(22, 60, n_samples)
years_at_company = np.random.randint(0, 15, n_samples)
salary = np.random.randint(40000, 150000, n_samples)
job_satisfaction = np.random.randint(1, 11, n_samples)
departments = np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing'], n_samples)

df = pd.DataFrame({
    'Age': age,
    'Years_At_Company': years_at_company,
    'Salary': salary,
    'Job_Satisfaction_Score': job_satisfaction,
    'Department': departments
})

os.makedirs('d:/auto-data-scientist/data', exist_ok=True)
df.to_csv('d:/auto-data-scientist/data/new_employees_untested.csv', index=False)
print("Untested Dataset generated at data/new_employees_untested.csv")
