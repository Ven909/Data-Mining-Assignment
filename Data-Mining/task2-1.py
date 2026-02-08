import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def perform_resilience_analysis():
    print("Loading datasets...")

    main_df = pd.read_csv('S19_All_Release_2_10_22/Data/MainTable.csv')
    code_states = pd.read_csv('/Users/venkata/Documents/CAL_POLY_CSC/WNTR26/CSC313/Assignments/Data-Mining-Assignment/S19_All_Release_2_10_22/Data/CodeStates/CodeStates.csv')
    early_df = pd.read_csv('/Users/venkata/Documents/CAL_POLY_CSC/WNTR26/CSC313/Assignments/Data-Mining-Assignment/S19_All_Release_2_10_22/early.csv')
    late_df = pd.read_csv('/Users/venkata/Documents/CAL_POLY_CSC/WNTR26/CSC313/Assignments/Data-Mining-Assignment/S19_All_Release_2_10_22/late.csv')
    subjects = pd.read_csv('/Users/venkata/Documents/CAL_POLY_CSC/WNTR26/CSC313/Assignments/Data-Mining-Assignment/S19_All_Release_2_10_22/Data/LinkTables/Subject.csv')

    # 2. Map Code to Events
    # We merge the actual source code text into the process data
    df = main_df.merge(code_states, on='CodeStateID', how='left')
    df['CodeLength'] = df['Code'].fillna('').str.len()

    # 3. Calculate "Code Churn"
    # Churn is the change in characters between consecutive submissions
    df = df.sort_values(by=['SubjectID', 'ProblemID', 'ServerTimestamp'])
    df['Churn'] = df.groupby(['SubjectID', 'ProblemID'])['CodeLength'].diff().abs()

    # 4. Define "Sinkhole" Events
    # These are compiler errors where the student changed < 5 characters (minimal effort)
    df['Is_Sinkhole'] = (df['EventType'] == 'Compile.Error') & (df['Churn'] < 5)

    # 5. Aggregate Metrics for Early Problems (First 30)
    # Filter only for records pertaining to early assignments
    early_subjects = early_df['SubjectID'].unique()
    early_process = df[df['SubjectID'].isin(early_subjects)]
    
    student_early_stats = early_process.groupby('SubjectID').agg(
        total_errors=('EventType', lambda x: (x == 'Compile.Error').sum()),
        sinkhole_errors=('Is_Sinkhole', 'sum'),
        avg_score=('Score', 'mean')
    ).reset_index()

    # Calculate Frustration Index (FI)
    # FI = 1.0 means every error was a "sinkhole" (micro-edit)
    student_early_stats['FI'] = (
        student_early_stats['sinkhole_errors'] / 
        (student_early_stats['total_errors'] + 1)
    )

    # 6. Merge with Outcomes (Final Grade and Late Struggle)
    # Get the final exam grade (X-Grade)
    analysis_df = student_early_stats.merge(subjects[['SubjectID', 'X-Grade']], on='SubjectID')
    
    # Calculate late term success rate (Label is TRUE if successful)
    late_perf = late_df.groupby('SubjectID')['Label'].mean().reset_index()
    late_perf.rename(columns={'Label': 'Late_Success_Rate'}, inplace=True)
    analysis_df = analysis_df.merge(late_perf, on='SubjectID')

    # 7. Visualization: Early Frustration vs. Final Grade
    plt.figure(figsize=(10, 6))
    plt.scatter(analysis_df['FI'], analysis_df['X-Grade'], color='#3498db', alpha=0.7)
    
    # Linear regression trend line
    m, b = np.polyfit(analysis_df['FI'], analysis_df['X-Grade'], 1)
    plt.plot(analysis_df['FI'], m*analysis_df['FI'] + b, color='#e74c3c', linestyle='--')
    
    plt.title('Early-Term Frustration Index vs. Final Exam Performance')
    plt.xlabel('Frustration Index (Ratio of Low-Effort Errors)')
    plt.ylabel('Final Exam Score (X-Grade)')
    plt.grid(True, alpha=0.3)
    plt.savefig('frustration_analysis.png')
    
    # 8. Evaluation & Output
    correlation = analysis_df['FI'].corr(analysis_df['X-Grade'])
    print(f"Correlation (FI vs Final Grade): {correlation:.3f}")
    
    # Export for reporting
    analysis_df.to_csv('student_behavioral_risk.csv', index=False)
    print("Report generated: student_behavioral_risk.csv")

if __name__ == "__main__":
    perform_resilience_analysis()