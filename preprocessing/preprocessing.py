import sys
import pandas as pd
def preprocess_data(file_path1, file_path2):
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Merge the two DataFrames based on the 'structureId' column using outer join
    merged_df = pd.merge(df1, df2, on='structureId', how='outer')

    #Drop columns that have no use to us
    dropped_df = merged_df.drop(columns=['experimentalTechnique', 'macromoleculeType_y', 'residueCount_y', 'resolution', 'crystallizationMethod', 'pdbxDetails', 'publicationYear'])

    # Drop rows with missing values
    cleaned_data = dropped_df.dropna()

    # Filter the data to only include proteins
    protein_data = cleaned_data[cleaned_data['macromoleculeType_x'] == 'Protein']

    # Filter data to only incude sequences <350 residues in length
    filtered_data = protein_data[protein_data['residueCount_x'] < 350]

    # Keep only the 30 most populous classes 
    class_counts = filtered_data['classification'].value_counts()
    top_classes = class_counts.nlargest(30).index
    filtered_data = filtered_data[filtered_data['classification'].isin(top_classes)]

    unique_data = filtered_data.drop_duplicates(subset='sequence', keep='first')

    return unique_data


if __name__ == "__main__":
    # Check if two file path arguments are provided
    if len(sys.argv) < 3:
        print("Please provide two file paths as command line arguments.")
        sys.exit(1)

    # Get the file paths from the command line arguments
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]

    # Call the preprocess_data function with the file paths
    preprocessed_data1 = preprocess_data(file_path1, file_path2)

    # Save the preprocessed data to a CSV file
    preprocessed_data1.to_csv('preprocessed_data.csv', index=False)
    