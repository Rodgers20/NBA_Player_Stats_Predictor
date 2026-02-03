# scripts/fetch_team_data.py

import os
import sys
import pandas as pd
from utils.data_fetch import get_team_data  # Adjust path if needed


# Add the project root to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetch import get_team_data  # Now it should find utils

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Fetch and display team data
team_data = get_team_data()
print(team_data.head())  # Show the first few rows of the team data

# Save the team data to a CSV file
team_data.to_csv("data/teams_list.csv", index=False)
print("Team data saved to data/teams_list.csv")
