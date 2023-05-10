# Imports

# Compute protein descriptors
from propy import PyPro
from propy import AAComposition
from propy import CTD

# Build Sequence Object
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Grouping iterable
from itertools import chain

# Data analysis
import pandas as pd

# Feature Extraction function
def extract_feature(sequences):
    """
    Extracts 572 features from peptide sequences.

    Inputs:
    - list/series: Sequences 

    Outputs:
    - dataframe: dataframe with 572 features 
    """
    
    # if the list is empty
    if len(sequences) == 0:
        print('List/Series is empty')
        return

    allFeaturesData = []

    for seq in sequences:

        # Make sure the sequence is a string
        s = str(seq)

        # replace the unappropriate peptide sequence to A
        s = s.replace('X','A')
        s = s.replace('x','A')
        s = s.replace('U','A')
        s = s.replace('Z','A')
        s = s.replace('B','A')

        # Calculating primary features
        analysed_seq = ProteinAnalysis(s)
        wt = analysed_seq.molecular_weight()
        arm = analysed_seq.aromaticity()
        instab = analysed_seq.instability_index()
        pI = analysed_seq.isoelectric_point()

        # create a list for the primary features
        pFeaturesData = [len(seq), wt, arm, instab, pI]

        # Get Amino Acid Composition (AAC), Composition Transition Distribution (CTD) and Dipeptide Composition (DPC)
        resultAAC = AAComposition.CalculateAAComposition(s)
        resultCTD = CTD.CalculateCTD(s)
        resultDPC = AAComposition.CalculateDipeptideComposition(s)

        # Collect all the features into lists
        aacFeaturesData = [j for i,j in resultAAC.items()]
        ctdFeaturesData = [l for k,l in resultCTD.items()]
        dpcFeaturesData = [n for m,n in resultDPC.items()]

        allFeaturesData.append(pFeaturesData + aacFeaturesData + ctdFeaturesData + dpcFeaturesData)

    pFeaturesName = ['SeqLength','Weight','Aromaticity','Instability','IsoelectricPoint']
    aacFeaturesData = [i for i,j in resultAAC.items()]
    ctdFeaturesData = [k for k,l in resultCTD.items()]
    dpcFeaturesData = [m for m,n in resultDPC.items()]

    featuresName  = []
    featuresName.append(pFeaturesName+aacFeaturesData+ctdFeaturesData+dpcFeaturesData)
    featuresFlattenList = list(chain.from_iterable(featuresName))

    # create dataframe using all extracted features and the names
    allFeaturesData = pd.DataFrame(allFeaturesData, columns = featuresFlattenList)
    
    return allFeaturesData