import numpy as np
import math
import re
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- BIOLOGICAL DICTIONARIES ---

# Helix forming propensity (Pace & Scholtz)
AA_HELIX_PROPENSITY = {
    'A': 0, 'L': -0.21, 'R': -0.21, 'M': -0.24, 'K': -0.26, 'Q': -0.39, 'E': -0.40,
    'I': -0.41, 'W': -0.49, 'S': -0.50, 'Y': -0.53, 'F': -0.54, 'V': -0.61, 'H': -0.61,
    'N': -0.65, 'T': -0.66, 'C': -0.68, 'D': -0.69, 'G': -1.0, 'P': -3.16
}

# Hydrophobicity for helical moment (Eisenberg) - Optimized for secondary structures
AA_HYDROPHOBICITY_EISENBERG = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29, 'Q': -0.85, 'E': -0.74,
    'G': 0.48, 'H': -0.40, 'I': 1.38, 'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19,
    'P': 0.12, 'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

# Electrical Charge
AA_CHARGE = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}

# General Hydrophobicity (Kyte-Doolittle)
AA_HYDRO = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
    'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
    'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Energy Stability per amino acid
AA_ENERGY_STABILITY = {
    'A': -0.11, 'R': -0.15, 'N': -0.32, 'D': -0.66, 'C': -0.01,
    'Q': -0.42, 'E': -0.71, 'G': -0.11, 'H': -0.18, 'I': -0.36,
    'L': -0.47, 'K': -0.21, 'M': -0.52, 'F': -0.78, 'P': -0.35,
    'S': -0.29, 'T': -0.28, 'W': -1.18, 'Y': -0.89, 'V': -0.31
}

# Affinity for CRM1 nuclear export protein binding pockets
CRM1_POCKET_AFFINITY = {
    'L': -4.5, 'I': -4.2, 'V': -3.8, 'F': -4.0, 'M': -3.9,
    'A': -1.5, 'T': -1.0,
    'P': 2.0,  'W': 1.5 
}

# --- HELPER FUNCTIONS ---

def calculate_hydrophobic_moment(seq, angle=100):
    """
    Calculates the hydrophobic moment to check for amphipathicity.
    High values suggest one side of the helix is hydrophobic and the other is hydrophilic.
    """
    hbar_x, hbar_y = 0.0, 0.0
    for i, aa in enumerate(seq):
        h = AA_HYDROPHOBICITY_EISENBERG.get(aa, 0)
        rad = math.radians(i * angle)
        hbar_x += h * math.cos(rad)
        hbar_y += h * math.sin(rad)
    return math.sqrt(hbar_x**2 + hbar_y**2) / len(seq) if len(seq) > 0 else 0

def calculate_hydrophobic_profile(seq, window_size=3):
    """
    Calculates a sliding window hydrophobic profile.
    Returns stats (max, min, std) representing 'peaks' and 'valleys' of hydrophobicity.
    """
    profile = []
    for i in range(len(seq) - window_size + 1):
        window = seq[i:i+window_size]
        h_val = sum([AA_HYDRO.get(aa, 0) for aa in window]) / window_size
        profile.append(h_val)
    if not profile: return [0, 0, 0]
    return [np.max(profile), np.min(profile), np.std(profile)]

def calculate_energy_features(seq):
    """
    Calculates peptide stability and CRM1 binding proxy energies.
    """
    seq = str(seq).upper()
    n = len(seq)
    if n == 0: return [0] * 5
    
    total_pep_energy = sum([AA_ENERGY_STABILITY.get(aa, 0) for aa in seq])
    pep_energy_density = total_pep_energy / n
    
    binding_stability = sum([CRM1_POCKET_AFFINITY.get(aa, 0) for aa in seq if aa in 'LIVFM'])
    penalty = 2.5 if ('P' in seq or 'W' in seq) else 0
    e_bind_proxy = binding_stability + penalty
    bind_energy_density = e_bind_proxy / n
    
    return [total_pep_energy, pep_energy_density, e_bind_proxy, bind_energy_density, penalty]

# --- MAIN FEATURE EXTRACTION FUNCTION ---

def get_nes_features(sequence):
    """
    Extracts 37 physicochemical and structural features from a protein sequence.
    """
    try:
        seq = str(sequence).upper()
        if not seq or len(seq) < 5:
            return [0] * 37

        analyser = ProteinAnalysis(seq)
        n = len(seq)
        
        # 1. Basic Protein Stats (9 features)
        feats = [
            n, analyser.molecular_weight(), analyser.aromaticity(),
            analyser.instability_index(), analyser.isoelectric_point(),
            analyser.gravy(), analyser.get_amino_acids_percent().get('L', 0), 
            sum([analyser.get_amino_acids_percent().get(aa, 0) for aa in 'DE']), 
            sum([analyser.get_amino_acids_percent().get(aa, 0) for aa in 'KRH']) 
        ]
        
        # 2. Termini characteristics (4 features)
        n_term, c_term = seq[:3], seq[-3:]
        feats.extend([
            sum([AA_CHARGE.get(aa, 0) for aa in n_term]),
            sum([AA_HYDRO.get(aa, 0) for aa in n_term]),
            sum([AA_CHARGE.get(aa, 0) for aa in c_term]),
            sum([AA_HYDRO.get(aa, 0) for aa in c_term])
        ])
        
        # 3. Structural & Pattern matching (3 features)
        feats.append(calculate_hydrophobic_moment(seq)) 
        feats.append(sum([AA_HELIX_PROPENSITY.get(aa, 0) for aa in seq]) / n)
        # Regex for common NES hydrophobic spacing pattern
        feats.append(1.0 if re.search(r"[LIVFM].{2,3}[LIVFM].{2,3}[LIVFM].{1}[LIVFM]", seq) else 0.0)

        # 4. Regional distribution (Split into thirds) (6 features)
        third = n // 3
        for seg in [seq[:third], seq[third:2*third], seq[2*third:]]:
            if len(seg) > 0:
                feats.extend([sum([AA_CHARGE.get(aa, 0) for aa in seg])/len(seg), 
                              sum([AA_HYDRO.get(aa, 0) for aa in seg])/len(seg)])
            else: feats.extend([0, 0])

        # 5. Hydrophobic Profile Stats (3 features)
        feats.extend(calculate_hydrophobic_profile(seq))
        
        # 6. Leucine density/spacing (1 feature)
        l_indices = [i for i, aa in enumerate(seq) if aa == 'L']
        feats.append(np.diff(l_indices).mean() if len(l_indices) > 1 else 0)

        # 7. Hydrophobic spacing stats (3 features)
        hydro_indices = [i for i, aa in enumerate(seq) if aa in 'LIVFM']
        if len(hydro_indices) > 1:
            diffs = np.diff(hydro_indices)
            feats.extend([np.std(diffs), np.max(diffs), len(hydro_indices)/n])
        else: feats.extend([0, 0, 0])

        # 8. Split hydrophobic moment (2 features)
        mid = n // 2
        feats.append(calculate_hydrophobic_moment(seq[:mid]))
        feats.append(calculate_hydrophobic_moment(seq[mid:]))

        # 9. Charge/Hydrophobicity Ratio (1 feature)
        th = sum([AA_HYDRO.get(aa, 0) for aa in seq])
        feats.append(sum([abs(AA_CHARGE.get(aa, 0)) for aa in seq]) / th if th != 0 else 0)

        # 10. Energy Features (5 features)
        energy_feats = calculate_energy_features(seq)
        feats.extend(energy_feats)

        return feats # Total: 37 features
    except Exception:
        return [0] * 37

def process_dataframe(df, col_name="aa_sequence"):
    """
    Helper to apply feature extraction to an entire dataframe.
    """
    print("Calculating features for sequences...")
    safe_sequences = df[col_name].fillna("")
    features = np.array([get_nes_features(s) for s in safe_sequences])
    return features