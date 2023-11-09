import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them into a single dataframe.
    
    Parameters:
    messages_filepath (str): File path for the messages dataset.
    categories_filepath (str): File path for the categories dataset.
    
    Returns:
    DataFrame: Merged dataframe of messages and categories.
    """
    messages = pd.read_csv(messages_filepath)  # Load messages dataset
    categories = pd.read_csv(categories_filepath)  # Load categories dataset

    df = pd.merge(messages, categories, on='id', how='inner')  # Merge datasets on id

    return df

def clean_data(df):
    """
    Clean the merged dataframe by splitting categories, converting category values, and removing duplicates.
    
    Parameters:
    df (DataFrame): Merged dataframe of messages and categories.
    
    Returns:
    DataFrame: Cleaned dataframe.
    """
    categories = df.categories.str.split(';', expand=True)  # Split categories into separate columns

    # Extract a list of new column names for categories
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames

    for column in categories:
        # Extract last char of each string (the number) and convert it to string
        categories[column] = categories[column].apply(lambda x: x[-1]).astype(str)

    df.drop(columns='categories', axis=1, inplace=True)  # Remove the original categories column

    df = pd.concat([df, categories], axis=1)  # Concatenate original dataframe with new categories dataframe

    df = df.drop_duplicates()  # Drop duplicate rows

    # Convert category values to int and filter rows with values other than 0 and 1
    df[category_colnames] = df[category_colnames].astype(int)
    df = df[(~df[category_colnames].isin((0, 1))).sum(axis=1)==0]

    return df

def save_data(df, database_filename):
    """
    Save the cleaned data into a SQLite database.
    
    Parameters:
    df (DataFrame): Cleaned dataframe.
    database_filename (str): Filename for the output SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  # Save dataframe to SQL table

def main():
    """
    Main function to execute the ETL pipeline.
    
    Perform the loading, cleaning, and saving of data when the script is run with the correct number of arguments.
    """
    if len(sys.argv) == 4:
        # Unpack filepaths provided as command-line arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # Load data from files
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # Clean data
        print('Cleaning data...')
        df = clean_data(df)
        
        # Save data to database
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        # Print usage instructions if correct arguments are not provided
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
