import pandas as pd

# Read the CSV file
df = pd.read_csv('breast-cancer.csv')

# Display the first few rows of the DataFrame
print(df)

'''doesnt show all the output by default, to do so , 
change set option "pd.set_option('display.max_columns', None)" '''