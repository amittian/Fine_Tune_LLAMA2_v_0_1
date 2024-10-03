import pandas as pd
import re

class IssueParser:
    def __init__(self, df):
        """
        Initialize IssueParser with a DataFrame.

        :param df: DataFrame containing the description column.
        """
        self.df = df

    def extract_app_and_issue(self, description):
        """
        Extracts Application and the entire Primary Issue from the description text.

        :param description: String containing the issue description.
        :return: Tuple of application name and primary issue.
        """
        # Regex pattern for extracting application name
        app_match = re.search(r'Application:\s*\[(.*?)\]', description)
        app_name = app_match.group(1) if app_match else None
        
        # Regex pattern for extracting the full primary issue including the label
        issue_match = re.search(r'(Primary Issue:\s*.*?);', description)
        primary_issue_name = issue_match.group(1) if issue_match else None

        return app_name, primary_issue_name

    def add_parsed_columns(self):
        """
        Adds two new columns 'app_name' and 'primary_issue_name' to the DataFrame
        by parsing the 'description' column.
        """
        # Apply the extraction function to each description
        self.df['app_name'], self.df['primary_issue_name'] = zip(*self.df['description'].apply(self.extract_app_and_issue))

    def get_dataframe(self):
        """
        Returns the modified DataFrame with the new parsed columns.

        :return: DataFrame with 'app_name' and 'primary_issue_name' columns.
        """
        return self.df

# Sample DataFrame with 5 records
data = {'description': [
    '''version 1 data not pulling into e911 task for filing
    Additional Info:  
    Application: [FUZE - Site Management];  
    Primary Issue: interface_issues;''',
    
    '''Customer data not syncing correctly. 
    Additional Info:  
    Application: [Connect - Order Management];  
    Primary Issue: data_sync_error;''',

    '''Unable to generate reports from the dashboard. 
    Additional Info:  
    Application: [ivapp - Reporting];  
    Primary Issue: report_generation_failure;''',

    '''API call is failing intermittently. 
    Additional Info:  
    Application: [FUZE - API Services];  
    Primary Issue: api_failure;''',

    '''Delay in processing customer requests. 
    Additional Info:  
    Application: [Connect - Customer Portal];  
    Primary Issue: request_processing_delay;'''
]}

# Creating the DataFrame
df = pd.DataFrame(data)

# Initialize IssueParser with the DataFrame
parser = IssueParser(df)

# Add parsed columns
parser.add_parsed_columns()

# Retrieve and display the updated DataFrame
updated_df = parser.get_dataframe()
print(updated_df[['description', 'app_name', 'primary_issue_name']])
