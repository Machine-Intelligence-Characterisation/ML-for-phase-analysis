from ase.db import connect
import sys

def count_entries(db_path):
    try:
        # Connect to the database
        db = connect(db_path)
        
        # Count the total number of entries
        total_entries = db.count()
        
        print(f"Database: {db_path}")
        print(f"Total number of entries: {total_entries}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    count_entries("training_data/simXRD_full_data/new/ILtrain_combined_1.db")