"""
Seed data for protein sequences.

This module provides sample protein sequences for development and testing,
including known protein folds and synthetic sequences.
"""

import logging
from typing import List
from datetime import datetime

from ..models import ProteinModel, StructureModel, PredictionMethod
from ..repositories import ProteinRepository, StructureRepository

logger = logging.getLogger(__name__)


# Sample protein sequences from well-known proteins
SAMPLE_PROTEINS = [
    {
        "sequence": "MKFLILLFNILCLFPVLAADQADVNVIGYLKKEENKLSKAKNV",
        "metadata": {
            "name": "Mini-protein alpha helix",
            "description": "Synthetic alpha-helical peptide for testing",
            "source": "synthetic",
            "secondary_structure": "helix",
        }
    },
    {
        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "metadata": {
            "name": "Small globular protein",
            "description": "Synthetic small globular protein with mixed secondary structure",
            "source": "synthetic",
            "secondary_structure": "mixed",
        }
    },
    {
        "sequence": "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE",
        "metadata": {
            "name": "Zinc finger protein",
            "description": "DNA-binding zinc finger domain",
            "source": "natural",
            "function": "DNA binding",
            "pfam_id": "PF00096",
        }
    },
    {
        "sequence": "AEAAAKEAAAKEAAAKEAAAK",
        "metadata": {
            "name": "Ideal alpha helix",
            "description": "Designed ideal alpha helix with regular repeat",
            "source": "designed",
            "secondary_structure": "helix",
        }
    },
    {
        "sequence": "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRET",
        "metadata": {
            "name": "Albumin fragment",
            "description": "Fragment of human serum albumin",
            "source": "natural",
            "function": "transport",
            "uniprot_id": "P02768",
        }
    },
    {
        "sequence": "GSSTGSSGMKTYWGRLGPIEFGLLGSPPGYVFR",
        "metadata": {
            "name": "Beta sheet peptide",
            "description": "Designed beta sheet forming peptide",
            "source": "designed",
            "secondary_structure": "sheet",
        }
    },
    {
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        "metadata": {
            "name": "Hemoglobin alpha chain",
            "description": "Human hemoglobin alpha chain",
            "source": "natural",
            "function": "oxygen transport",
            "uniprot_id": "P69905",
        }
    },
    {
        "sequence": "PGPGPGPGPGPGPGPGPGPG",
        "metadata": {
            "name": "Flexible linker",
            "description": "Glycine-proline flexible linker",
            "source": "designed",
            "secondary_structure": "coil",
        }
    },
    {
        "sequence": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMNCKCVIS",
        "metadata": {
            "name": "K-RAS protein",
            "description": "Human KRAS proto-oncogene protein",
            "source": "natural",
            "function": "GTPase",
            "uniprot_id": "P01116",
        }
    },
    {
        "sequence": "KKYRVVLLAYLQISQVWETFGAIVGNALRIAHRYQGVMVLMKMVTLNLPSDFKEFLARLPELFLLFGKRVLGRQSVQVMVQMLQMRNCFWVEFKGGQP",
        "metadata": {
            "name": "Designed enzyme",
            "description": "Designed enzyme with novel catalytic activity",
            "source": "designed",
            "function": "catalysis",
        }
    },
]

# Sample structure data (simplified PDB-like content)
SAMPLE_STRUCTURES = [
    {
        "protein_sequence": "MKFLILLFNILCLFPVLAADQADVNVIGYLKKEENKLSKAKNV",
        "pdb_content": """HEADER    DE NOVO PROTEIN                        01-JAN-25   TEST            
ATOM      1  N   MET A   1      -8.901   4.127  -0.555  1.00 11.99           N  
ATOM      2  CA  MET A   1      -8.608   3.135  -1.618  1.00 11.99           C  
ATOM      3  C   MET A   1      -7.117   2.964  -1.897  1.00 11.99           C  
ATOM      4  O   MET A   1      -6.632   1.849  -1.897  1.00 11.99           O  
END
""",
        "confidence": 0.85,
        "structure_quality": 0.78,
        "prediction_method": PredictionMethod.ESMFOLD,
        "metadata": {
            "ramachandran_score": 0.92,
            "clash_score": 0.05,
            "compactness_score": 0.67,
        }
    },
]


def seed_proteins(connection=None) -> None:
    """Seed the database with sample proteins."""
    from ..connection import get_connection
    
    if connection is None:
        connection = get_connection()
    
    protein_repo = ProteinRepository(connection)
    structure_repo = StructureRepository(connection)
    
    logger.info("Seeding sample proteins...")
    
    created_count = 0
    for protein_data in SAMPLE_PROTEINS:
        # Create protein model
        protein = ProteinModel(
            sequence=protein_data["sequence"],
            metadata=protein_data["metadata"]
        )
        
        # Check if protein already exists
        existing = protein_repo.get_by_sequence_hash(protein.sequence_hash)
        if existing:
            logger.debug(f"Protein already exists: {protein_data['metadata'].get('name', 'Unknown')}")
            continue
        
        # Create protein
        protein_id = protein_repo.create(protein)
        protein.id = protein_id
        created_count += 1
        
        logger.debug(f"Created protein: {protein_data['metadata'].get('name', 'Unknown')} (ID: {protein_id})")
        
        # Add structure if available
        structure_data = next(
            (s for s in SAMPLE_STRUCTURES if s["protein_sequence"] == protein.sequence),
            None
        )
        
        if structure_data:
            structure = StructureModel(
                protein_id=protein_id,
                pdb_content=structure_data["pdb_content"],
                confidence=structure_data["confidence"],
                structure_quality=structure_data["structure_quality"],
                prediction_method=structure_data["prediction_method"],
                metadata=structure_data["metadata"]
            )
            
            structure_id = structure_repo.create(structure)
            logger.debug(f"Created structure for protein {protein_id} (Structure ID: {structure_id})")
    
    logger.info(f"Created {created_count} new proteins")


def create_test_proteins(count: int = 50) -> List[ProteinModel]:
    """
    Generate synthetic test proteins for development.
    
    Args:
        count: Number of test proteins to generate
        
    Returns:
        List of ProteinModel instances
    """
    import random
    import string
    
    # Common amino acid patterns
    aa_standard = "ACDEFGHIKLMNPQRSTVWY"
    aa_hydrophobic = "AILMFPWV"
    aa_hydrophilic = "DEKRNQHST"
    aa_small = "AGSTC"
    aa_aromatic = "FWY"
    
    test_proteins = []
    
    for i in range(count):
        # Generate sequence of random length
        length = random.randint(20, 200)
        
        # Create different types of sequences
        sequence_type = random.choice(["random", "hydrophobic_rich", "hydrophilic_rich", "small_rich"])
        
        if sequence_type == "random":
            sequence = ''.join(random.choices(aa_standard, k=length))
        elif sequence_type == "hydrophobic_rich":
            # 70% hydrophobic, 30% other
            hydrophobic_count = int(length * 0.7)
            other_count = length - hydrophobic_count
            sequence = ''.join(
                random.choices(aa_hydrophobic, k=hydrophobic_count) +
                random.choices(aa_standard, k=other_count)
            )
            random.shuffle(list(sequence))
            sequence = ''.join(sequence)
        elif sequence_type == "hydrophilic_rich":
            # 70% hydrophilic, 30% other
            hydrophilic_count = int(length * 0.7)
            other_count = length - hydrophilic_count
            sequence = ''.join(
                random.choices(aa_hydrophilic, k=hydrophilic_count) +
                random.choices(aa_standard, k=other_count)
            )
            random.shuffle(list(sequence))
            sequence = ''.join(sequence)
        else:  # small_rich
            # 50% small amino acids, 50% other
            small_count = int(length * 0.5)
            other_count = length - small_count
            sequence = ''.join(
                random.choices(aa_small, k=small_count) +
                random.choices(aa_standard, k=other_count)
            )
            random.shuffle(list(sequence))
            sequence = ''.join(sequence)
        
        # Create metadata
        metadata = {
            "name": f"Test protein {i+1}",
            "description": f"Synthetic test protein of type {sequence_type}",
            "source": "synthetic_test",
            "sequence_type": sequence_type,
            "generated_at": datetime.utcnow().isoformat(),
        }
        
        protein = ProteinModel(sequence=sequence, metadata=metadata)
        test_proteins.append(protein)
    
    return test_proteins


def seed_test_proteins(connection=None, count: int = 20) -> None:
    """Seed database with synthetic test proteins."""
    from ..connection import get_connection
    
    if connection is None:
        connection = get_connection()
    
    protein_repo = ProteinRepository(connection)
    
    logger.info(f"Generating {count} test proteins...")
    test_proteins = create_test_proteins(count)
    
    created_count = 0
    for protein in test_proteins:
        existing = protein_repo.get_by_sequence_hash(protein.sequence_hash)
        if not existing:
            protein_id = protein_repo.create(protein)
            created_count += 1
    
    logger.info(f"Created {created_count} test proteins")


if __name__ == "__main__":
    # Run seeding if executed directly
    logging.basicConfig(level=logging.INFO)
    seed_proteins()
    seed_test_proteins()