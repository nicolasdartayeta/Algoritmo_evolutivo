import sqlite3
import re  # Using regex for potentially easier value extraction
import sys # To get filename from command line argument
import os  # To check if file exists

def parse_benchmark_results(filename):
    """
    Parses the benchmark results text file.

    Args:
        filename (str): The path to the benchmark results text file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the parsed data for one configuration run.
        str: The TSP problem name found in the header, or None.
    """
    if not os.path.exists(filename):
        print(f"Error: Input file '{filename}' not found.")
        return [], None

    results_list = []
    current_config = {}
    tsp_problem = None

    # Regex patterns to extract values more reliably
    patterns = {
        # Config section
        "population_size": re.compile(r"Population Size:\s*(\d+)", re.IGNORECASE),
        "crossover_rate": re.compile(r"Crossover Rate:\s*([\d.]+)", re.IGNORECASE),
        "mutation_rate": re.compile(r"Mutation Rate:\s*([\d.]+)", re.IGNORECASE),
        "generations": re.compile(r"Generations:\s*(\d+)", re.IGNORECASE),
        "selection_operator": re.compile(r"Selection:\s*(.+)", re.IGNORECASE),
        "crossover_operator": re.compile(r"Crossover:\s*(.+)", re.IGNORECASE),
        "mutation_operator": re.compile(r"Mutation:\s*(.+)", re.IGNORECASE),
        "replacement_operator": re.compile(r"Replacement:\s*(.+)", re.IGNORECASE),
        # Results section
        "num_executions": re.compile(r"Results \(Avg over (\d+) runs\):", re.IGNORECASE),
        "avg_execution_time": re.compile(r"Avg Execution Time:\s*([\d.]+) seconds", re.IGNORECASE),
        "avg_best_cost": re.compile(r"Avg Best Cost:\s*([\d.]+)", re.IGNORECASE),
        "std_dev_best_cost": re.compile(r"Std Dev Best Cost:\s*([\d.]+)", re.IGNORECASE),
        # Header info
        "tsp_problem_header": re.compile(r"TSP Problem:\s*(.+)", re.IGNORECASE),
    }

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                print(line)
                # Check for header info (like TSP problem) before configs start
                if not current_config and not results_list: # Only check at the very beginning
                     match = patterns["tsp_problem_header"].search(line)
                     if match:
                          tsp_problem = match.group(1).strip()
                          continue # Move to next line after finding header

                if line.startswith("Configuration:"):
                    # Starting a new block, store the previous one if valid
                    if current_config and "avg_execution_time" in current_config: # Check if results were parsed
                        results_list.append(current_config)
                    current_config = {} # Reset for the new block
                    if tsp_problem: # Add problem name if found
                         current_config['tsp_problem'] = tsp_problem
                    continue

                if not current_config and not line.startswith("Config:"):
                    # Skip lines before the first "Config:" block (or empty lines)
                    # unless it's header info handled above
                    continue

                # Try matching patterns within a block
                matched = False
                for key, pattern in patterns.items():
                    if key == "tsp_problem_header": continue # Already handled

                    match = pattern.search(line)
                    if match:
                        value = match.group(1).strip()
                        # Handle specific keys and type conversions
                        if key in ["population_size", "generations", "num_executions"]:
                            current_config[key] = int(value)
                        elif key in ["crossover_rate", "mutation_rate",
                                     "avg_execution_time", "avg_best_cost", "std_dev_best_cost"]:
                            current_config[key] = float(value)
                        else: # Operator names (strings)
                            current_config[key] = value
                        matched = True
                        break # Stop checking patterns for this line

                # Add last config block after the loop finishes if it's valid
            if current_config and "avg_execution_time" in current_config:
                results_list.append(current_config)

    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        return [], None
    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
        return [], None # Return empty list on error

    return results_list, tsp_problem

def store_results_in_db(db_filename, results_data):
    """
    Stores the parsed benchmark results into an SQLite database.

    Args:
        db_filename (str): The path to the SQLite database file.
        results_data (list): A list of dictionaries from parse_benchmark_results.
    """
    if not results_data:
        print("No results data to store.")
        return

    conn = None # Initialize connection to None
    try:
        conn = sqlite3.connect(db_filename)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        # Using TEXT for operators, REAL for floats, INTEGER for ints
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tsp_problem TEXT,
                population_size INTEGER,
                crossover_rate REAL,
                mutation_rate REAL,
                generations INTEGER,
                selection_operator TEXT,
                crossover_operator TEXT,
                mutation_operator TEXT,
                replacement_operator TEXT,
                num_executions INTEGER,
                avg_execution_time REAL,
                avg_best_cost REAL,
                std_dev_best_cost REAL
            )
        ''')

        # Insert data using parameterized queries
        sql_insert = '''
            INSERT INTO benchmark (
                tsp_problem, population_size, crossover_rate, mutation_rate, generations,
                selection_operator, crossover_operator, mutation_operator, replacement_operator,
                num_executions, avg_execution_time, avg_best_cost, std_dev_best_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        rows_inserted = 0
        for record in results_data:
            # Ensure all expected keys exist, provide defaults (None) if missing
            data_tuple = (
                record.get('tsp_problem'), # Get from parsed data
                record.get('population_size'),
                record.get('crossover_rate'),
                record.get('mutation_rate'),
                record.get('generations'),
                record.get('selection_operator'),
                record.get('crossover_operator'),
                record.get('mutation_operator'),
                record.get('replacement_operator'),
                record.get('num_executions'),
                record.get('avg_execution_time'),
                record.get('avg_best_cost'),
                record.get('std_dev_best_cost')
            )
            # Basic check if essential data is present before inserting
            if record.get('avg_execution_time') is not None:
                 try:
                      cursor.execute(sql_insert, data_tuple)
                      rows_inserted += 1
                 except sqlite3.Error as e:
                      print(f"Error inserting record: {e}")
                      print(f"Problematic data: {data_tuple}")


        conn.commit()
        print(f"Successfully inserted {rows_inserted} records into '{db_filename}'.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during database operation: {e}")
    finally:
        if conn:
            conn.close()

# --- Main Script Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # Get input filename from command line argument or use a default
    if len(sys.argv) > 1:
        input_txt_file = sys.argv[1]
    else:
        # Find the latest benchmark results file if no argument given
        try:
             benchmark_files = [f for f in os.listdir('.') if f.startswith('benchmark_results_') and f.endswith('.txt')]
             if benchmark_files:
                  input_txt_file = max(benchmark_files, key=os.path.getctime) # Get latest file
                  print(f"No input file specified. Using latest found: '{input_txt_file}'")
             else:
                  print("Error: No benchmark results file found in current directory and no filename provided.")
                  sys.exit(1)

        except Exception as e:
             print(f"Error finding latest benchmark file: {e}")
             print("Please provide the results filename as a command line argument.")
             sys.exit(1)


    db_file = "ga_benchmark_data_p43.db" # Name for the SQLite database file
    # --------------------

    print(f"Parsing benchmark data from '{input_txt_file}'...")
    parsed_data, problem_name = parse_benchmark_results(input_txt_file)

    if parsed_data:
        print(f"Parsed {len(parsed_data)} configuration results.")
        if problem_name:
             print(f"Detected TSP Problem: {problem_name}")
        print(f"Storing results into database '{db_file}'...")
        store_results_in_db(db_file, parsed_data)
    else:
        print("No data parsed from the file.")

    print("Script finished.")