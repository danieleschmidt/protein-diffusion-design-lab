"""
Streamlit web application for the Protein Diffusion Design Lab.

This provides an interactive web interface for protein generation,
structure prediction, and binding affinity analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.protein_diffusion import (
        ProteinDiffuser, ProteinDiffuserConfig,
        AffinityRanker, AffinityRankerConfig,
        StructurePredictor, StructurePredictorConfig
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    MODULES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Protein Diffusion Design Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'generated_sequences' not in st.session_state:
        st.session_state.generated_sequences = []
    if 'ranked_results' not in st.session_state:
        st.session_state.ranked_results = []
    if 'structure_results' not in st.session_state:
        st.session_state.structure_results = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

def create_sidebar():
    """Create the sidebar with navigation and controls."""
    st.sidebar.markdown("# 🧬 Protein Diffusion Lab")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Home", "🔬 Generate", "🏆 Rank & Analyze", "📊 Visualize", "⚙️ Settings"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Session Statistics")
    
    # Display session statistics
    if st.session_state.generated_sequences:
        st.sidebar.metric("Generated Sequences", len(st.session_state.generated_sequences))
    
    if st.session_state.ranked_results:
        st.sidebar.metric("Ranked Proteins", len(st.session_state.ranked_results))
    
    if st.session_state.structure_results:
        st.sidebar.metric("Predicted Structures", len(st.session_state.structure_results))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Quick Actions")
    
    if st.sidebar.button("🗑️ Clear All Data"):
        for key in ['generated_sequences', 'ranked_results', 'structure_results', 'analysis_results']:
            st.session_state[key] = [] if key != 'analysis_results' else {}
        st.sidebar.success("Data cleared!")
    
    return page

def show_home_page():
    """Display the home page."""
    st.markdown('<h1 class="main-header">🧬 Protein Diffusion Design Lab</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;">
    A plug-and-play diffusion pipeline for protein scaffolds that rivals commercial suites
    </div>
    """, unsafe_allow_html=True)
    
    if not MODULES_AVAILABLE:
        st.error("""
        ⚠️ **Modules not available!** 
        
        The protein diffusion modules could not be imported. This might be because:
        - Dependencies are not installed
        - The code is still being developed
        - There are import errors
        
        Please check the installation and try again.
        """)
        return
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🔬 Generate</h3>
        <p>Create novel protein scaffolds using state-of-the-art diffusion models with motif conditioning and temperature control.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🏆 Rank & Analyze</h3>
        <p>Evaluate generated proteins using binding affinity prediction, structural quality, and diversity metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Visualize</h3>
        <p>Interactive visualizations for sequence analysis, binding affinity distributions, and quality metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown('<h2 class="sub-header">🚀 Quick Start Guide</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Navigate to Generate** to create protein scaffolds with your desired motifs
    2. **Use Rank & Analyze** to evaluate and rank the generated proteins
    3. **Check Visualize** for interactive analysis and insights
    4. **Adjust Settings** to customize the generation and evaluation parameters
    """)
    
    # System status
    st.markdown('<h2 class="sub-header">🔧 System Status</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PyTorch", "✅ Available" if True else "❌ Missing")
    with col2:
        st.metric("CUDA", "✅ Available" if False else "❌ Not Available")  # Placeholder
    with col3:
        st.metric("ESM Models", "⚠️ Not Loaded")

def show_generate_page():
    """Display the protein generation page."""
    st.markdown('<h1 class="main-header">🔬 Generate Protein Scaffolds</h1>', unsafe_allow_html=True)
    
    if not MODULES_AVAILABLE:
        st.error("Modules not available. Please check the installation.")
        return
    
    # Generation parameters
    st.markdown('<h2 class="sub-header">⚙️ Generation Parameters</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        motif = st.text_input(
            "Target Motif",
            value="HELIX_SHEET_HELIX",
            help="Specify the desired protein motif (e.g., HELIX_SHEET_HELIX, SHEET_LOOP_SHEET)"
        )
        
        num_samples = st.number_input(
            "Number of Samples",
            min_value=1,
            max_value=1000,
            value=10,
            help="Number of protein scaffolds to generate"
        )
        
        max_length = st.number_input(
            "Maximum Length",
            min_value=50,
            max_value=1000,
            value=256,
            help="Maximum sequence length for generated proteins"
        )
    
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Sampling temperature (higher = more diverse)"
        )
        
        sampling_method = st.selectbox(
            "Sampling Method",
            ["ddpm", "ddim"],
            help="Choose between DDPM (higher quality) or DDIM (faster)"
        )
        
        ddim_steps = st.number_input(
            "DDIM Steps",
            min_value=10,
            max_value=200,
            value=50,
            help="Number of denoising steps for DDIM sampling"
        ) if sampling_method == "ddim" else 50
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        use_checkpoint = st.checkbox("Use Pre-trained Model", value=False)
        checkpoint_path = st.text_input("Checkpoint Path", value="") if use_checkpoint else ""
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Classifier-free guidance scale"
        )
    
    # Generate button
    if st.button("🚀 Generate Proteins", type="primary"):
        try:
            with st.spinner("Generating protein scaffolds... This may take a few minutes."):
                # Create configuration
                config = ProteinDiffuserConfig()
                config.num_samples = num_samples
                config.max_length = max_length
                config.temperature = temperature
                config.guidance_scale = guidance_scale
                
                # Initialize diffuser
                if use_checkpoint and checkpoint_path:
                    diffuser = ProteinDiffuser.from_pretrained(checkpoint_path, config)
                    st.info(f"Loaded model from {checkpoint_path}")
                else:
                    diffuser = ProteinDiffuser(config)
                    st.warning("Using randomly initialized model (demo mode)")
                
                # Generate sequences
                results = diffuser.generate(
                    motif=motif if motif else None,
                    num_samples=num_samples,
                    max_length=max_length,
                    temperature=temperature,
                    sampling_method=sampling_method,
                    ddim_steps=ddim_steps,
                    progress=False,
                )
                
                # Store results in session state
                st.session_state.generated_sequences = results
                
                st.markdown("""
                <div class="success-message">
                ✅ <strong>Generation Complete!</strong><br>
                Generated {} protein scaffolds successfully.
                </div>
                """.format(len(results)), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            logger.error(f"Generation error: {e}")
    
    # Display results
    if st.session_state.generated_sequences:
        st.markdown('<h2 class="sub-header">📋 Generated Sequences</h2>', unsafe_allow_html=True)
        
        results = st.session_state.generated_sequences
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Generated", len(results))
        with col2:
            avg_length = np.mean([len(r['sequence']) for r in results])
            st.metric("Average Length", f"{avg_length:.1f}")
        with col3:
            avg_confidence = np.mean([r['confidence'] for r in results])
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        with col4:
            high_conf = sum(1 for r in results if r['confidence'] > 0.7)
            st.metric("High Confidence", f"{high_conf}/{len(results)}")
        
        # Sequence table
        df_data = []
        for i, result in enumerate(results):
            df_data.append({
                'Index': i + 1,
                'Sequence': result['sequence'][:50] + ('...' if len(result['sequence']) > 50 else ''),
                'Length': result['length'],
                'Confidence': f"{result['confidence']:.3f}",
                'Full Sequence': result['sequence']
            })
        
        df = pd.DataFrame(df_data)
        
        # Display table with selection
        selected_indices = st.multiselect(
            "Select sequences for further analysis:",
            options=list(range(len(results))),
            format_func=lambda x: f"Sequence {x+1} (L={results[x]['length']}, C={results[x]['confidence']:.3f})"
        )
        
        st.dataframe(df[['Index', 'Sequence', 'Length', 'Confidence']], use_container_width=True)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            # FASTA download
            fasta_content = ""
            for i, result in enumerate(results):
                fasta_content += f">sequence_{i+1}_conf_{result['confidence']:.3f}\n{result['sequence']}\n"
            
            st.download_button(
                "📄 Download FASTA",
                data=fasta_content,
                file_name="generated_sequences.fasta",
                mime="text/plain"
            )
        
        with col2:
            # JSON download
            json_content = json.dumps(results, indent=2, default=str)
            st.download_button(
                "📊 Download JSON",
                data=json_content,
                file_name="generation_results.json",
                mime="application/json"
            )
        
        # Move selected sequences to ranking
        if selected_indices and st.button("🏆 Rank Selected Sequences"):
            selected_results = [results[i] for i in selected_indices]
            st.session_state.selected_for_ranking = selected_results
            st.success(f"Selected {len(selected_results)} sequences for ranking!")

def show_rank_analyze_page():
    """Display the ranking and analysis page."""
    st.markdown('<h1 class="main-header">🏆 Rank & Analyze Proteins</h1>', unsafe_allow_html=True)
    
    if not MODULES_AVAILABLE:
        st.error("Modules not available. Please check the installation.")
        return
    
    # Data source selection
    st.markdown('<h2 class="sub-header">📊 Data Source</h2>', unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choose data source:",
        ["From Generated Sequences", "Upload File", "Manual Input"]
    )
    
    sequences = []
    
    if data_source == "From Generated Sequences":
        if st.session_state.generated_sequences:
            sequences = [r['sequence'] for r in st.session_state.generated_sequences]
            st.success(f"Using {len(sequences)} generated sequences")
        else:
            st.warning("No generated sequences available. Please generate some first.")
            return
    
    elif data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload sequence file (FASTA or text)", type=['fasta', 'txt', 'fa'])
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            if content.startswith('>'):
                # FASTA format
                lines = content.split('\n')
                for line in lines:
                    if not line.startswith('>') and line.strip():
                        sequences.append(line.strip())
            else:
                # Text format
                sequences = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.success(f"Loaded {len(sequences)} sequences from file")
    
    elif data_source == "Manual Input":
        manual_sequences = st.text_area(
            "Enter sequences (one per line):",
            height=200,
            placeholder="MKLLILTCLVAVALARPKHPIPWDQAITVAYASRALGRGLVVMAQDGNRGGKFHPWTVNQGPLKDYICQAYDMGTTTEVPGTMGMLRRRSNVWSCLPRLLCERVAAPNLDPEGFVVAVPIPVYEAWDFGDPKLNLRQNTVAVTCTGVQTLAVRGRVGNLLSNGVPIGRGLPHIPSKGSGATFEFIGSDLKAELATDQAGVLQVDVQQVEACWFASQGGGVDTDYTGQPWDGGKPTVTGAMCGAFSCRHDGKRDVRVGTAAGVGGGYCSDGDGPVKPVVSNPNQALAFGLSEAGSRRLHPFTTARQGAGSM"
        )
        if manual_sequences:
            sequences = [seq.strip() for seq in manual_sequences.split('\n') if seq.strip()]
            st.success(f"Entered {len(sequences)} sequences")
    
    if not sequences:
        st.info("Please provide sequences to rank and analyze.")
        return
    
    # Ranking parameters
    st.markdown('<h2 class="sub-header">⚙️ Ranking Parameters</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_pdb = st.file_uploader("Target PDB file (optional)", type=['pdb'])
        max_results = st.number_input("Maximum Results", min_value=1, max_value=len(sequences), value=min(50, len(sequences)))
        
    with col2:
        binding_weight = st.slider("Binding Affinity Weight", 0.0, 1.0, 0.4, 0.1)
        structure_weight = st.slider("Structure Quality Weight", 0.0, 1.0, 0.3, 0.1)
        diversity_weight = st.slider("Diversity Weight", 0.0, 1.0, 0.2, 0.1)
        novelty_weight = st.slider("Novelty Weight", 0.0, 1.0, 0.1, 0.1)
    
    # Advanced ranking options
    with st.expander("🔧 Advanced Ranking Options"):
        min_confidence = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        min_structure_quality = st.slider("Minimum Structure Quality", 0.0, 1.0, 0.6, 0.05)
        max_clash_score = st.slider("Maximum Clash Score", 0.0, 1.0, 0.1, 0.05)
    
    # Rank sequences
    if st.button("🚀 Rank Sequences", type="primary"):
        try:
            with st.spinner("Ranking sequences... This may take several minutes."):
                # Save target PDB if provided
                target_pdb_path = None
                if target_pdb:
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdb', delete=False) as tmp:
                        tmp.write(target_pdb.read())
                        target_pdb_path = tmp.name
                
                # Create ranker configuration
                config = AffinityRankerConfig()
                config.binding_weight = binding_weight
                config.structure_weight = structure_weight
                config.diversity_weight = diversity_weight
                config.novelty_weight = novelty_weight
                config.min_confidence = min_confidence
                config.min_structure_quality = min_structure_quality
                config.max_clash_score = max_clash_score
                config.max_results = max_results
                
                # Initialize ranker
                ranker = AffinityRanker(config)
                
                # Rank sequences
                ranked_results = ranker.rank(
                    sequences,
                    target_pdb=target_pdb_path,
                    return_detailed=True
                )
                
                # Store results
                st.session_state.ranked_results = ranked_results
                
                # Clean up temporary file
                if target_pdb_path:
                    Path(target_pdb_path).unlink()
                
                st.markdown("""
                <div class="success-message">
                ✅ <strong>Ranking Complete!</strong><br>
                Ranked {} sequences successfully.
                </div>
                """.format(len(ranked_results)), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Ranking failed: {str(e)}")
            logger.error(f"Ranking error: {e}")
    
    # Display ranking results
    if st.session_state.ranked_results:
        st.markdown('<h2 class="sub-header">📊 Ranking Results</h2>', unsafe_allow_html=True)
        
        results = st.session_state.ranked_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Ranked", len(results))
        with col2:
            top_score = results[0]['composite_score']
            st.metric("Top Score", f"{top_score:.4f}")
        with col3:
            avg_score = np.mean([r['composite_score'] for r in results])
            st.metric("Average Score", f"{avg_score:.4f}")
        with col4:
            high_quality = sum(1 for r in results if r.get('structure_quality', 0) > 0.7)
            st.metric("High Quality", f"{high_quality}/{len(results)}")
        
        # Results table
        df_data = []
        for i, result in enumerate(results):
            df_data.append({
                'Rank': i + 1,
                'Sequence': result['sequence'][:30] + ('...' if len(result['sequence']) > 30 else ''),
                'Composite Score': f"{result['composite_score']:.4f}",
                'Binding Affinity': f"{result.get('binding_affinity', 0):.2f}",
                'Structure Quality': f"{result.get('structure_quality', 0):.3f}",
                'Confidence': f"{result.get('confidence', 0):.3f}",
                'Diversity Score': f"{result.get('diversity_score', 0):.3f}",
                'Full Sequence': result['sequence']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df.drop('Full Sequence', axis=1), use_container_width=True)
        
        # Top sequences display
        st.markdown('<h3 class="sub-header">🏅 Top 10 Sequences</h3>', unsafe_allow_html=True)
        
        for i, result in enumerate(results[:10]):
            with st.expander(f"Rank {i+1}: Score {result['composite_score']:.4f}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.text(f"Sequence: {result['sequence']}")
                with col2:
                    st.json({
                        'Composite Score': result['composite_score'],
                        'Binding Affinity': result.get('binding_affinity', 'N/A'),
                        'Structure Quality': result.get('structure_quality', 'N/A'),
                        'Confidence': result.get('confidence', 'N/A'),
                        'Length': result.get('length', len(result['sequence']))
                    })
        
        # Download results
        col1, col2 = st.columns(2)
        with col1:
            # Top sequences FASTA
            top_fasta = ""
            for i, result in enumerate(results[:20]):
                top_fasta += f">rank_{i+1}_score_{result['composite_score']:.4f}\n{result['sequence']}\n"
            
            st.download_button(
                "📄 Download Top 20 FASTA",
                data=top_fasta,
                file_name="top_sequences.fasta",
                mime="text/plain"
            )
        
        with col2:
            # Full results JSON
            json_content = json.dumps(results, indent=2, default=str)
            st.download_button(
                "📊 Download Full Results JSON",
                data=json_content,
                file_name="ranking_results.json",
                mime="application/json"
            )

def show_visualize_page():
    """Display the visualization and analysis page."""
    st.markdown('<h1 class="main-header">📊 Visualize & Analyze</h1>', unsafe_allow_html=True)
    
    # Check if we have data to visualize
    has_generated = bool(st.session_state.generated_sequences)
    has_ranked = bool(st.session_state.ranked_results)
    
    if not has_generated and not has_ranked:
        st.info("No data available for visualization. Please generate or rank some sequences first.")
        return
    
    # Data selection
    viz_data_source = st.selectbox(
        "Select data to visualize:",
        ["Generated Sequences", "Ranked Results"] if has_ranked else ["Generated Sequences"]
    )
    
    if viz_data_source == "Generated Sequences" and has_generated:
        data = st.session_state.generated_sequences
        st.success(f"Visualizing {len(data)} generated sequences")
        
        # Sequence length distribution
        st.markdown('<h2 class="sub-header">📏 Sequence Length Distribution</h2>', unsafe_allow_html=True)
        lengths = [len(d['sequence']) for d in data]
        fig_length = px.histogram(
            x=lengths,
            nbins=20,
            title="Distribution of Sequence Lengths",
            labels={'x': 'Sequence Length', 'y': 'Count'}
        )
        st.plotly_chart(fig_length, use_container_width=True)
        
        # Confidence distribution
        st.markdown('<h2 class="sub-header">🎯 Confidence Score Distribution</h2>', unsafe_allow_html=True)
        confidences = [d['confidence'] for d in data]
        fig_conf = px.histogram(
            x=confidences,
            nbins=20,
            title="Distribution of Confidence Scores",
            labels={'x': 'Confidence Score', 'y': 'Count'}
        )
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Length vs Confidence scatter
        st.markdown('<h2 class="sub-header">📈 Length vs Confidence</h2>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            x=lengths,
            y=confidences,
            title="Sequence Length vs Confidence Score",
            labels={'x': 'Sequence Length', 'y': 'Confidence Score'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    elif viz_data_source == "Ranked Results" and has_ranked:
        data = st.session_state.ranked_results
        st.success(f"Visualizing {len(data)} ranked sequences")
        
        # Score distributions
        st.markdown('<h2 class="sub-header">🏆 Score Distributions</h2>', unsafe_allow_html=True)
        
        # Composite scores
        composite_scores = [d['composite_score'] for d in data]
        fig_composite = px.histogram(
            x=composite_scores,
            nbins=20,
            title="Composite Score Distribution",
            labels={'x': 'Composite Score', 'y': 'Count'}
        )
        st.plotly_chart(fig_composite, use_container_width=True)
        
        # Multi-metric visualization
        st.markdown('<h2 class="sub-header">📊 Multi-Metric Analysis</h2>', unsafe_allow_html=True)
        
        # Prepare data for multi-metric plot
        plot_data = []
        for i, result in enumerate(data[:50]):  # Limit to top 50 for clarity
            plot_data.append({
                'Rank': i + 1,
                'Composite Score': result['composite_score'],
                'Binding Affinity': result.get('binding_affinity', 0),
                'Structure Quality': result.get('structure_quality', 0),
                'Confidence': result.get('confidence', 0),
                'Diversity Score': result.get('diversity_score', 0),
                'Sequence Length': len(result['sequence'])
            })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Correlation heatmap
        st.markdown('<h3>🔥 Score Correlations</h3>')
        numeric_cols = ['Composite Score', 'Binding Affinity', 'Structure Quality', 'Confidence', 'Diversity Score']
        corr_matrix = df_plot[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Score Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Score vs Rank
        st.markdown('<h3>📈 Scores vs Rank</h3>')
        fig_rank = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Composite Score', 'Binding Affinity', 'Structure Quality', 'Confidence')
        )
        
        fig_rank.add_trace(
            go.Scatter(x=df_plot['Rank'], y=df_plot['Composite Score'], mode='markers', name='Composite'),
            row=1, col=1
        )
        fig_rank.add_trace(
            go.Scatter(x=df_plot['Rank'], y=df_plot['Binding Affinity'], mode='markers', name='Binding'),
            row=1, col=2
        )
        fig_rank.add_trace(
            go.Scatter(x=df_plot['Rank'], y=df_plot['Structure Quality'], mode='markers', name='Structure'),
            row=2, col=1
        )
        fig_rank.add_trace(
            go.Scatter(x=df_plot['Rank'], y=df_plot['Confidence'], mode='markers', name='Confidence'),
            row=2, col=2
        )
        
        fig_rank.update_layout(height=600, showlegend=False, title_text="Score Trends by Rank")
        st.plotly_chart(fig_rank, use_container_width=True)
        
        # Top performers analysis
        st.markdown('<h2 class="sub-header">🌟 Top Performers Analysis</h2>', unsafe_allow_html=True)
        
        top_10 = data[:10]
        
        col1, col2 = st.columns(2)
        with col1:
            # Length distribution of top performers
            top_lengths = [len(d['sequence']) for d in top_10]
            fig_top_lengths = px.bar(
                x=list(range(1, 11)),
                y=top_lengths,
                title="Sequence Lengths of Top 10",
                labels={'x': 'Rank', 'y': 'Sequence Length'}
            )
            st.plotly_chart(fig_top_lengths, use_container_width=True)
        
        with col2:
            # Score breakdown for top performer
            top_result = data[0]
            scores = {
                'Composite': top_result['composite_score'],
                'Binding': top_result.get('binding_affinity', 0),
                'Structure': top_result.get('structure_quality', 0),
                'Confidence': top_result.get('confidence', 0),
                'Diversity': top_result.get('diversity_score', 0)
            }
            
            fig_top_scores = px.bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                title="Top Sequence Score Breakdown",
                labels={'x': 'Metric', 'y': 'Score'}
            )
            st.plotly_chart(fig_top_scores, use_container_width=True)
    
    # Amino acid composition analysis
    if has_generated or has_ranked:
        st.markdown('<h2 class="sub-header">🧪 Amino Acid Composition</h2>', unsafe_allow_html=True)
        
        sequences = [d['sequence'] for d in data]
        
        # Calculate AA composition
        aa_counts = {}
        total_aa = 0
        
        for seq in sequences:
            for aa in seq:
                if aa.isalpha():
                    aa_counts[aa] = aa_counts.get(aa, 0) + 1
                    total_aa += 1
        
        # Convert to percentages
        aa_percentages = {aa: (count / total_aa) * 100 for aa, count in aa_counts.items()}
        
        # Sort by frequency
        sorted_aa = sorted(aa_percentages.items(), key=lambda x: x[1], reverse=True)
        
        fig_aa = px.bar(
            x=[aa for aa, _ in sorted_aa],
            y=[pct for _, pct in sorted_aa],
            title="Amino Acid Composition (%)",
            labels={'x': 'Amino Acid', 'y': 'Percentage'}
        )
        st.plotly_chart(fig_aa, use_container_width=True)

def show_settings_page():
    """Display the settings and configuration page."""
    st.markdown('<h1 class="main-header">⚙️ Settings & Configuration</h1>', unsafe_allow_html=True)
    
    # Model settings
    st.markdown('<h2 class="sub-header">🤖 Model Settings</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Model Checkpoint Path", value="", help="Path to pre-trained model checkpoint")
        st.selectbox("Device", ["auto", "cpu", "cuda"], help="Computation device")
        st.number_input("Model Vocabulary Size", value=32, help="Size of the tokenizer vocabulary")
    
    with col2:
        st.number_input("Model Dimensions", value=1024, help="Model embedding dimensions")
        st.number_input("Number of Layers", value=24, help="Number of transformer layers")
        st.number_input("Number of Heads", value=16, help="Number of attention heads")
    
    # Generation settings
    st.markdown('<h2 class="sub-header">🔬 Generation Settings</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Default Temperature", 0.1, 2.0, 1.0, 0.1)
        st.slider("Default Guidance Scale", 1.0, 10.0, 1.0, 0.5)
        st.number_input("Default Max Length", value=256, min_value=50, max_value=1000)
    
    with col2:
        st.number_input("Default Num Samples", value=10, min_value=1, max_value=1000)
        st.selectbox("Default Sampling Method", ["ddpm", "ddim"])
        st.number_input("Default DDIM Steps", value=50, min_value=10, max_value=200)
    
    # Ranking settings
    st.markdown('<h2 class="sub-header">🏆 Ranking Settings</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Default Binding Weight", 0.0, 1.0, 0.4, 0.1)
        st.slider("Default Structure Weight", 0.0, 1.0, 0.3, 0.1)
        st.slider("Default Diversity Weight", 0.0, 1.0, 0.2, 0.1)
    
    with col2:
        st.slider("Default Novelty Weight", 0.0, 1.0, 0.1, 0.1)
        st.slider("Min Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        st.number_input("Default Max Results", value=100, min_value=1, max_value=1000)
    
    # System settings
    st.markdown('<h2 class="sub-header">💻 System Settings</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Enable Verbose Logging", value=False)
        st.checkbox("Save Intermediate Results", value=True)
        st.text_input("Output Directory", value="./output")
    
    with col2:
        st.checkbox("Enable Progress Bars", value=True)
        st.checkbox("Auto-save Results", value=True)
        st.number_input("Max Memory Usage (GB)", value=16, min_value=1, max_value=128)
    
    # Export/Import settings
    st.markdown('<h2 class="sub-header">📁 Configuration Management</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Configuration"):
            st.success("Configuration saved!")
    
    with col2:
        if st.button("📂 Load Configuration"):
            st.success("Configuration loaded!")
    
    with col3:
        if st.button("🔄 Reset to Defaults"):
            st.success("Reset to default settings!")

def main():
    """Main application function."""
    initialize_session_state()
    
    # Create sidebar and get current page
    current_page = create_sidebar()
    
    # Route to appropriate page
    if current_page == "🏠 Home":
        show_home_page()
    elif current_page == "🔬 Generate":
        show_generate_page()
    elif current_page == "🏆 Rank & Analyze":
        show_rank_analyze_page()
    elif current_page == "📊 Visualize":
        show_visualize_page()
    elif current_page == "⚙️ Settings":
        show_settings_page()

if __name__ == "__main__":
    main()