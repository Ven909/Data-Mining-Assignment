import pandas as pd

def task_one():

    main_df = pd.read_csv('S19_All_Release_2_10_22/Data/MainTable.csv')
    subject_df = pd.read_csv('S19_All_Release_2_10_22/Data/LinkTables/Subject.csv')
    early_df = pd.read_csv('S19_All_Release_2_10_22/early.csv')
    late_df = pd.read_csv('S19_All_Release_2_10_22/late.csv')

    # --- Task 1 ---
    main_unique = main_df['SubjectID'].nunique()
    sub_unique = subject_df['SubjectID'].nunique()
    
    print(f"Task 1")
    print(f"   - Students in MainTable: {main_unique}")
    print(f"   - Students in SubjectTable : {sub_unique}")
    print(f"   - Discrepancy: {abs(main_unique - sub_unique)} students")
    '''
    But why though?
    I'm guessing instructors used test accounts to verify problems in MainTable.
    Then, these accounts were excluded in Subject because they aren't real students.

    Or, maybe some students dropped the class halfway.
    '''

    # --- Task 2 ---
    all_perf_df = pd.concat([early_df, late_df])
    avg_attempts = all_perf_df.groupby('ProblemID')['Attempts'].mean().reset_index()
    top_attempt_prob = avg_attempts.loc[avg_attempts['Attempts'].idxmax()]

    print(f"Task 2")
    print(f"   - Problem ID: {top_attempt_prob['ProblemID']}")
    print(f"   - Avg Attempts per Student: {top_attempt_prob['Attempts']:.2f}")

    # --- Task 3 ---
    # filter for errors, then count per student per problem
    error_events = main_df[main_df['EventType'] == 'Compile.Error']
    error_counts = error_events.groupby(['ProblemID', 'SubjectID']).size().reset_index(name='Count')
    
    # get average errors per problem 
    avg_errors = error_counts.groupby('ProblemID')['Count'].mean().reset_index()    
    top_error_prob = avg_errors.loc[avg_errors['Count'].idxmax()]

    print(f"Task 3")
    print(f"   - Problem ID: {top_error_prob['ProblemID']}")    # most error prob
    print(f"   - Avg Erros per Student: {top_error_prob['Count']:.2f}")     # avg # of errors on that prob
    '''
    The average is better because, in most college courses, the participation
    drops towards the end of the term.     
    '''

if __name__ == "__main__":
    task_one()