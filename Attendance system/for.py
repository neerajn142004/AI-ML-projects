import pandas as pd
import os
from datetime import datetime

# --- Configuration (Must match attendance.py) ---
CSV_FILE = 'attendance_log.csv'
EXCEL_REPORT_PREFIX = 'Attendance_Summary_Report_'

def generate_excel_report():
    """
    Reads the raw CSV log, processes it to calculate First Entry, 
    Last Exit, Duration, and daily counts, and saves the formatted 
    data to a multi-sheet Excel file.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found at {CSV_FILE}")
        return

    try:
        # 1. Load and Prepare Data
        df = pd.read_csv(CSV_FILE, dtype={'Person_ID': str, 'Action': str})
        
        if df.empty or 'Timestamp' not in df.columns:
            print("Error: CSV file is empty or missing the 'Timestamp' column.")
            return

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        
        # Extract the Date component
        df['Date'] = df['Timestamp'].dt.date
        
        # 2. Group and Aggregate Data
        
        # Group by Date and Person_ID
        grouped = df.groupby(['Date', 'Person_ID'])
        
        # NOTE: FIX 1 - Added include_groups=False to silence DeprecationWarning.
        # NOTE: FIX 2 - Used .dt.time immediately after min()/max() on the Series,
        #               but will ensure conversion works below.

        # Calculate First Entry Time (Timestamp object)
        first_entry = grouped.apply(lambda x: x[x['Action'] == 'ENTRY']['Timestamp'].min(), 
                                    include_groups=False) 
        first_entry.name = 'First_Entry_Time'

        # Calculate Last Exit Time (Timestamp object)
        last_exit = grouped.apply(lambda x: x[x['Action'] == 'EXIT']['Timestamp'].max(), 
                                  include_groups=False)
        last_exit.name = 'Last_Exit_Time'

        # Calculate Entry Count
        entry_count = grouped.apply(lambda x: (x['Action'] == 'ENTRY').sum(), 
                                    include_groups=False)
        entry_count.name = 'Entry_Count'
        
        # Calculate Exit Count
        exit_count = grouped.apply(lambda x: (x['Action'] == 'EXIT').sum(), 
                                   include_groups=False)
        exit_count.name = 'Exit_Count'

        # Combine all aggregations into a single DataFrame
        summary_df = pd.concat([first_entry, last_exit, entry_count, exit_count], axis=1)
        summary_df.reset_index(inplace=True)
        
        # 3. Calculate Duration
        
        # Fix the "Can only use .dt accessor" error by directly accessing the time component 
        # using the .dt accessor on the Timestamp objects, which are now correctly created.
        summary_df['First_Entry_Time_Only'] = summary_df['First_Entry_Time'].dt.time
        summary_df['Last_Exit_Time_Only'] = summary_df['Last_Exit_Time'].dt.time

        # Calculate Duration (Last Exit - First Entry). 
        summary_df['Duration'] = summary_df['Last_Exit_Time'] - summary_df['First_Entry_Time']

        # Format Duration as HH:MM:SS string
        def format_duration(td):
            if pd.isna(td):
                return None
            total_seconds = int(td.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

        summary_df['Duration_H:M:S'] = summary_df['Duration'].apply(format_duration)
        
        # 4. Final Formatting
        
        final_columns = [
            'Date', 
            'Person_ID', 
            'First_Entry_Time_Only', 
            'Last_Exit_Time_Only', 
            'Duration_H:M:S', 
            'Entry_Count', 
            'Exit_Count'
        ]
        
        final_report = summary_df[final_columns].sort_values(by=['Date', 'Person_ID']).reset_index(drop=True)
        
        # Rename columns to match the desired output
        final_report.rename(columns={
            'First_Entry_Time_Only': 'First_Entry_Time',
            'Last_Exit_Time_Only': 'Last_Exit_Time'
        }, inplace=True)
        
        # Convert Time objects (which are technically strings after .dt.time) to string format for cleaner Excel display
        # Note: If they are already strings from .dt.time, this just ensures it.
        final_report['First_Entry_Time'] = final_report['First_Entry_Time'].astype(str).str.replace('NaT', '')
        final_report['Last_Exit_Time'] = final_report['Last_Exit_Time'].astype(str).str.replace('NaT', '')


        # 5. Write to Excel with multiple sheets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_excel_name = f"{EXCEL_REPORT_PREFIX}{timestamp}.xlsx"
        
        with pd.ExcelWriter(final_excel_name, engine='openpyxl') as writer:
            
            # Sheet 1: Main Summary Report (Matches your requested format)
            final_report.to_excel(writer, sheet_name='Daily Summary', index=False)
            
            # Sheet 2: Raw Data (for inspection/backup)
            df[['Timestamp', 'Person_ID', 'Action']].to_excel(writer, sheet_name='Raw Log Entries', index=False)
            
        print(f"\n--- SUCCESS: Detailed Excel report generated: {final_excel_name} ---")
        print("You can now run 'python report_tool.py' to query the data.")

    except Exception as e:
        print(f"An unexpected error occurred during report generation: {e}")

if __name__ == '__main__':
    generate_excel_report()