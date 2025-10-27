import pandas as pd
df = pd.read_csv(r'c:\Users\harsh\student_risk_management\data\processed\cleaned_student_data.csv')
print('Dropout distribution:')
print(df['dropout'].value_counts())
print(f'\nTotal: {len(df)}')
print(f'At Risk (1): {df["dropout"].sum()} ({df["dropout"].mean()*100:.1f}%)')
print(f'Safe (0): {(~df["dropout"].astype(bool)).sum()} ({(1-df["dropout"].mean())*100:.1f}%)')
