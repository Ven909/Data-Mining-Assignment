import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
early = pd.read_csv('S19_All_Release_2_10_22/early.csv')
subjects = pd.read_csv('S19_All_Release_2_10_22/Data/LinkTables/Subject.csv')

# 2. Aggregate student performance from the early problems
student_stats = early.groupby('SubjectID').agg(
    Avg_Attempts=('Attempts', 'mean'),
    Early_Success_Rate=('CorrectEventually', 'mean'),
    Total_Struggles=('Label', lambda x: (x == False).sum())
).reset_index()

# 3. Merge with final grades
df_final = student_stats.merge(subjects[['SubjectID', 'X-Grade']], on='SubjectID')

# 4. Define the success quadrants
med_attempts = df_final['Avg_Attempts'].median()
med_grade = df_final['X-Grade'].median()

def categorize(row):
    if row['X-Grade'] >= med_grade:
        return 'High Achiever (Naturals)' if row['Avg_Attempts'] < med_attempts else 'High Achiever (Grinders)'
    else:
        return 'At Risk (Disengaged)' if row['Avg_Attempts'] < med_attempts else 'At Risk (Struggling)'

df_final['Quadrant'] = df_final.apply(categorize, axis=1)

# 5. Visualizations
plt.figure(figsize=(12, 8))

# Scatter plot
sns.scatterplot(
    data=df_final, 
    x='Avg_Attempts', 
    y='X-Grade', 
    hue='Quadrant', 
    style='Quadrant',
    s=100,
    palette='viridis'
)

# Add thj quadrant dividers
plt.axvline(med_attempts, color='grey', linestyle='--', alpha=0.5)
plt.axhline(med_grade, color='grey', linestyle='--', alpha=0.5)

plt.title('How does Early Persistence affect Final Grade')
plt.xlabel('Average Attempts (Early Problems)')
plt.ylabel('Final Exam Grade (X-Grade)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Data-Mining/task2-artifacts/success_quadrant.png')

# 6. Correlation Summary
print("--- Correlation Analysis ---")
print(df_final[['Avg_Attempts', 'Early_Success_Rate', 'Total_Struggles', 'X-Grade']].corr()['X-Grade'])

# 7. Let's see if struggling in the early problems (Label=False) correlates with later problems
late = pd.read_csv('S19_All_Release_2_10_22/late.csv')
late_struggle = late.groupby('SubjectID')['Label'].mean().reset_index()
late_struggle.rename(columns={'Label': 'Late_Success_Rate'}, inplace=True)

eval_df = df_final.merge(late_struggle, on='SubjectID')
late_corr = eval_df['Early_Success_Rate'].corr(eval_df['Late_Success_Rate'])
print(f"\n How well does Early success correlate to Late success: {late_corr:.3f}")