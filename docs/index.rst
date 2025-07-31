Protein Diffusion Design Lab Documentation
==========================================

Welcome to the Protein Diffusion Design Lab documentation!

This is a comprehensive diffusion-based protein design platform that democratizes access to state-of-the-art protein engineering tools.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_reference
   tutorials/index
   workflows/index
   development
   contributing
   changelog

Quick Start
-----------

.. code-block:: python

   from protein_diffusion import ProteinDiffuser, AffinityRanker

   # Initialize the diffusion model
   diffuser = ProteinDiffuser(
       checkpoint='weights/boltz-1b.ckpt',
       device='cuda'
   )

   # Generate protein scaffolds
   scaffolds = diffuser.generate(
       motif="HELIX_SHEET_HELIX",
       num_samples=100,
       temperature=0.8
   )

Key Features
------------

* **Pre-trained 1B Parameter Model**: State-of-the-art diffusion weights
* **SELFIES Tokenizer**: Robust molecular representation
* **FoldSeek Integration**: Comprehensive structural evaluation
* **Interactive UI**: Streamlit-based design interface

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/protein-diffusion-design-lab.git
   cd protein-diffusion-design-lab

   # Install dependencies
   pip install -r requirements.txt

   # Download pre-trained weights
   python scripts/download_weights.py --model boltz-1b

System Requirements
-------------------

* Python 3.9+
* CUDA 11.0+ GPU with 16GB+ VRAM (24GB recommended)
* 32GB system RAM
* 50GB free disk space

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`