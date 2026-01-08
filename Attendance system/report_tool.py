from itertools import count
import pandas as pd
from datetime import datetime, timedelta # Import timedelta
import argparse
from dateutil.parser import parse, ParserError
from typing import Optional, List, Tuple
import sys

# --- Configuration ---
LOG_FILE = 'attendance_log.csv'
UNKNOWN_VISITOR_ID = "Unknown_Visitor" # Must match the ID in attendance_app.py!
# IMPORTANT: Add the UNKNOWN_VISITOR_ID so the parser recognizes it
KNOWN_NAMES = ["Neeraj", "Yashith", UNKNOWN_VISITOR_ID] 

# --- Helper Functions ---

def load_and_prepare_log() -> Optional[pd.DataFrame]:
    """Loads the CSV log and converts Timestamps to datetime objects."""
    try:
        # Suppress DtypeWarning for mixed types if columns are empty
        df = pd.read_csv(LOG_FILE, dtype={'Person_ID': str, 'Action': str})
        
        # Check if the log is empty (only headers exist)
        if df.empty or len(df) == 0:
            return pd.DataFrame(columns=['Person_ID', 'Action', 'Timestamp'])

        # Convert Timestamp column
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except FileNotFoundError:
        print(f"Error: Log file '{LOG_FILE}' not found. Run attendance_app.py first.")
        return None
    except Exception as e:
        print(f"An error occurred loading the log: {e}")
        return None

def parse_natural_query(query: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Parses a natural language query string to extract Person, Date, and Action.
    Returns (person, date_str, action)
    """
    query = query.lower()
    
    # Initialize all variables
    person: Optional[str] = None
    date_str: Optional[str] = None
    action: str = 'EXIT' # Default action
    
    # 1. Extract Person ID
    for name in KNOWN_NAMES:
        if name.lower() in query:
            person = name
            break
            
    # CRITICAL: If the user queries "unknown" or "visitor", map it to the ID.
    if person is None and ("unknown" in query or "visitor" in query):
        person = UNKNOWN_VISITOR_ID


    # 2. Extract Date
    # Try to parse the date from the whole string
    try:
        # dateutil tries to find a date (e.g., 'today', 'Oct 14th', '2025-10-14')
        parsed_date = parse(query, fuzzy=True, dayfirst=False).date()
        date_str = parsed_date.strftime('%Y-%m-%d')
    except ParserError:
        pass # date_str remains None
    except Exception:
        pass # Ignore other parsing issues

    # --- UPDATED: Check for "today" and "yesterday" specifically ---
    if date_str is None:
        if "yesterday" in query:
            # Calculate and set date_str for yesterday
            yesterday = datetime.now() - timedelta(days=1)
            date_str = yesterday.strftime('%Y-%m-%d')
        elif "today" in query:
            # Set date_str for today
            date_str = datetime.now().strftime('%Y-%m-%d')
    # -----------------------------------------------------------

    # 3. Determine Action/Command (Expanded Vocabulary)
    if any(keyword in query for keyword in ["total count", "total events", "summary"]):
        action = 'SUMMARY' # <-- NEW SUMMARY ACTION
        
    elif any(keyword in query for keyword in ["what time", "when", "show logs", "list entries", "list exits"]):
        action = 'LOGS'
        
    elif "status" in query or "in or out" in query or "who is where" in query or "where is" in query:
        action = 'STATUS'
        
    elif any(keyword in query for keyword in ["enter", "entry", "come in", "came", "arrive", "arrival", "check in"]):
        action = 'ENTRY' # Action for COUNT
        
    elif any(keyword in query for keyword in ["left", "exit", "went out", "gone", "check out", "leave"]):
        action = 'EXIT' # Action for COUNT
    
    return person, date_str, action

def query_current_status(df: pd.DataFrame, date_str: Optional[str] = None):
    """Answers: Who is currently IN or OUT? Filters by date if provided."""
    
    # --- NEW: Filter DataFrame by Date if date_str is provided ---
    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            # Filter the main DataFrame to only include events from the target date
            df_filtered = df[df['Timestamp'].dt.date == target_date].copy()
            time_context = f" on {date_str}"
        except ValueError:
            print("Error: Invalid date format. Please use YYYY-MM-DD.")
            return
    else:
        df_filtered = df.copy()
        time_context = " (All Time)"
    # -----------------------------------------------------------
    
    print(f"\n✅ Query Result: Current Status of People{time_context}:")
    
    if df_filtered.empty:
        print("No relevant log data found.")
        return

    # Get the last action for each person in the (date-filtered) log
    latest_actions = df_filtered.sort_values(by='Timestamp').groupby('Person_ID').last()
    
    for index, row in latest_actions.iterrows():
        status = row['Action']
        current_location = "IN" if status == "ENTRY" else "OUT"
        print(f"- {index}: Currently {current_location} (Last seen: {row['Timestamp']})")

def query_total_employees(df: pd.DataFrame):
    """Answers: How many unique people are in the log?"""
    unique_count = df['Person_ID'].nunique()
    print(f"\n✅ Query Result: The attendance log contains data for {unique_count} unique person/entity(s).")
    
    if unique_count > 0:
        names = df['Person_ID'].unique()
        print(f"Names found: {', '.join(names)}")

def query_status_filter(df: pd.DataFrame, required_status: str, date_str: Optional[str] = None):
    """Answers: Who is IN or WHO is OUT. Filters by date if provided."""
    
    # --- NEW: Filter DataFrame by Date if date_str is provided ---
    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            # Filter the main DataFrame to only include events from the target date
            df_filtered = df[df['Timestamp'].dt.date == target_date].copy()
            time_context = f" on {date_str}"
        except ValueError:
            print("Error: Invalid date format. Please use YYYY-MM-DD.")
            return
    else:
        df_filtered = df.copy()
        time_context = " (All Time)"
    # -----------------------------------------------------------
    
    if df_filtered.empty:
        print(f"\n✅ Query Result: No log data found{time_context}.")
        return

    latest_actions = df_filtered.sort_values(by='Timestamp').groupby('Person_ID').last()
    required_action = "ENTRY" if required_status == "IN" else "EXIT"
    
    filtered_people = latest_actions[latest_actions['Action'] == required_action]
    
    names_list = filtered_people.index.tolist()
    count = len(names_list)
    
    print(f"\n✅ Query Result: Currently, {count} person(s) are {required_status.upper()}{time_context}:")
    
    if count > 0:
        print(f"People {required_status.upper()}: {', '.join(names_list)}")
    else:
        print(f"No people are currently {required_status.upper()}{time_context}.")


def query_action_count(df: pd.DataFrame, person_id: str, action_type: str, date_str: Optional[str] = None):
    """
    Generalized function to count ENTRY or EXIT events and format the output dynamically.
    """
    
    # 1. Filter the DataFrame based on Person and Action
    filtered_df = df[(df['Person_ID'] == person_id) & (df['Action'] == action_type)].copy()
    
    # 2. Filter by Date if provided
    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            filtered_df = filtered_df[filtered_df['Timestamp'].dt.date == target_date]
        except ValueError:
            print("Error: Invalid date format. Please use YYYY-MM-DD.")
            return
    
    count = len(filtered_df)
    
    # 3. Determine the natural date context for the sentence
    if date_str:
        date_context = f" on {date_str}"
    else:
        date_context = " in total"

    # 4. Determine the natural action phrase and create the dynamic response
    action_phrase = "came to the office" if action_type == "ENTRY" else "left the office"
    
    print(f"✅ Query Result: {person_id} {action_phrase} {count} time(s){date_context}.")

def query_specific_logs(df: pd.DataFrame, person_id: str, date_str: Optional[str] = None):
    """
    Retrieves and displays the actual timestamps/logs for a person.
    """
    
    # 1. Filter by Person
    filtered_df = df[df['Person_ID'] == person_id].copy()
    
    # 2. Filter by Date if provided
    date_context = " (All Time)"
    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            filtered_df['Date'] = filtered_df['Timestamp'].dt.date
            filtered_df = filtered_df[filtered_df['Date'] == target_date]
            date_context = f" on {date_str}"
        except ValueError:
            print("Error: Invalid date format.")
            return

    if filtered_df.empty:
        print(f"\n✅ Query Result: No log entries found for {person_id}{date_context}.")
        return

    print(f"\n✅ Query Result: Log entries found for {person_id}{date_context}:")
    
    # Sort and display the results
    filtered_df = filtered_df.sort_values(by='Timestamp')
    
    for index, row in filtered_df.iterrows():
        action = row['Action']
        time_only = row['Timestamp'].strftime('%H:%M:%S')
        date_only = row['Timestamp'].strftime('%Y-%m-%d')
        
        print(f"- {date_only} | {action:<7} recorded at {time_only}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    df_log = load_and_prepare_log()
    
    if df_log is None:
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Query the office attendance log using natural language or commands.")
    
    # Add new specific command-line flags for summary queries
    parser.add_argument('--total-employees', action='store_true', help='Show the total count and list of unique employees in the log.')
    parser.add_argument('--who-is-in', action='store_true', help='Show the list of employees currently IN the office.')
    parser.add_argument('--who-is-out', action='store_true', help='Show the list of employees currently OUT of the office.')
    
    # Argument for Natural Queries
    parser.add_argument('query_string', nargs='*', help='The full query, e.g., "status", "Neeraj left today", or "who is in?".')

    args = parser.parse_args()
    
    # --- HANDLE NEW COMMAND-LINE QUERIES FIRST ---
    if args.total_employees:
        query_total_employees(df_log)
    elif args.who_is_in:
        # Pass today's date if no natural language query is provided, or rely on future logic
        query_status_filter(df_log, "IN")
    elif args.who_is_out:
        # Pass today's date if no natural language query is provided, or rely on future logic
        query_status_filter(df_log, "OUT")
    
    # --- FALLBACK TO NATURAL LANGUAGE QUERY ---
    else:
        full_query = " ".join(args.query_string).strip()
        
        if not full_query:
            print("Please provide a query or use a command-line flag (e.g., 'status', 'Neeraj left today', or '--who-is-in').")
            parser.print_help()
            sys.exit(1)
        
        # Parse the natural query
        person, date_str, action = parse_natural_query(full_query)
        
        print(f"Interpreting query: Person={person}, Action={action}, Date={date_str if date_str else 'All Time'}")
        
        # --- EXECUTE PARSED NATURAL QUERY ---
        if action == 'STATUS':
            # Pass the extracted date_str (which will be today's date if "today" or "yesterday" was in the query)
            query_current_status(df_log, date_str) 
        elif action == 'LOGS':
            if person:
                query_specific_logs(df_log, person, date_str)
            else:
                print("\nError: Please specify the person's name for this query (e.g., 'show logs for Neeraj').")
        
        elif action == 'SUMMARY':
            if person:
                # Determine the date context for the header
                date_context_header = f" on {date_str}" if date_str else " in total"
                
                print(f"\n--- Attendance Summary for {person}{date_context_header} ---")
                
                # Call query_action_count for ENTRY
                query_action_count(df_log, person, 'ENTRY', date_str)
                
                # Call query_action_count for EXIT
                query_action_count(df_log, person, 'EXIT', date_str)
                print("-------------------------------------------------------------------")
            else:
                print("\nError: Please specify the person's name for a summary query.")
                
        elif person and action in ['ENTRY', 'EXIT']:
            # This is the original, specific COUNT query
            query_action_count(df_log, person, action, date_str)
        else:
            print(f"\nCould not fully understand query or missing person/action: '{full_query}'")
            print("Please ensure the query contains a person's name or a recognized status command.")