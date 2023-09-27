import os
import pandas as pd

def calculate_average_results(metrics_dir, excels_dir, excel_name, i_start, i_end):
    # Create an empty DataFrame for the summary
    summary_df = pd.DataFrame(columns=[
        'Leave-One-Out Horse',
        'RF Accuracy',
        'RF Precision',
        'RF Recall',
        'RF F1 Score',
        'RF Confusion Matrix',
    ])

    # Iterate over each horse
    for i in range(i_start, i_end + 1):
        # Read Naive Bayes Excel file for the current horse
        nb_filename = os.path.join(metrics_dir, f"metrics_{i}.xlsx")
        nb_df = pd.read_excel(nb_filename)

        # Get the performance metrics for Naive Bayes
        nb_accuracy = nb_df.loc[0, 'Accuracy']
        nb_precision = nb_df.loc[0, 'Precision']
        nb_recall = nb_df.loc[0, 'Recall']
        nb_f1_score = nb_df.loc[0, 'F1 Score']
        nb_confusion_matrix = nb_df.loc[0, 'Confusion Matrix']

        # Add a row to the summary DataFrame
        summary_df.loc[i-1] = [
            i,
            nb_accuracy,
            nb_precision,
            nb_recall,
            nb_f1_score,
            nb_confusion_matrix
        ]

    # Calculate the average of each column
    average_row = summary_df.mean(numeric_only=True)
    average_row['Leave-One-Out Horse'] = 'Average'

    # Create an empty DataFrame for the blank row
    blank_row = pd.DataFrame()

    # Append the blank row to the summary DataFrame
    summary_df = summary_df._append(blank_row, ignore_index=True)

    # Append the average row to the summary DataFrame
    summary_df = summary_df._append(average_row, ignore_index=True)

    # Save the summary to the specified path and file name
    summary_filename = os.path.join(excels_dir, excel_name + ".xlsx")
    summary_df.to_excel(summary_filename, index=False)

    return average_row[1]