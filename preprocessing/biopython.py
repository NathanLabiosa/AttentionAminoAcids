import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import sys

def process_protein_data(sequence):
    def get_isoelectric_point(sequence):
        try:
            analyzed_seq = ProteinAnalysis(sequence)
            return analyzed_seq.isoelectric_point()
        except:
            return None

    def get_aromaticity(sequence):
        try:
            analyzed_seq = ProteinAnalysis(sequence)
            return analyzed_seq.aromaticity()
        except:
            return None

    def get_instability_index(sequence):
        try:
            analyzed_seq = ProteinAnalysis(sequence)
            return analyzed_seq.instability_index()
        except:
            return None

    def get_secondary_structure_fraction(sequence):
        try:
            analyzed_seq = ProteinAnalysis(sequence)
            return analyzed_seq.secondary_structure_fraction()  # Returns tuple (helix, turn, sheet)
        except:
            return (None, None, None)

    # Create a DataFrame with a single row containing the input sequence
    data = pd.DataFrame({'sequence': [sequence]})

    # Apply the functions
    data['Isoelectric_Point'] = data['sequence'].apply(get_isoelectric_point)
    data['Aromaticity'] = data['sequence'].apply(get_aromaticity)
    data['Instability_Index'] = data['sequence'].apply(get_instability_index)

    # For secondary structure, since it returns a tuple, you might want to separate it into different columns
    data[['Helix', 'Turn', 'Sheet']] = data['sequence'].apply(
        lambda x: pd.Series(get_secondary_structure_fraction(x))
    )

    # Check for any NaN values and decide how to handle them
    data.isna().sum()

    # Save the data to a CSV file
    data.to_csv('protein_data.csv', index=False)

# Take input from the command line
sequence_input = sys.argv[1]

# Call the function with the input sequence
process_protein_data(sequence_input)
