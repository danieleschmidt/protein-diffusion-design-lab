"""
Command-line interface for the Protein Diffusion Design Lab.

This module provides a comprehensive CLI for protein generation, structure
prediction, and analysis using the diffusion model pipeline.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    TORCH_AVAILABLE = False

from .diffuser import ProteinDiffuser, ProteinDiffuserConfig
from .ranker import AffinityRanker, AffinityRankerConfig
from .folding.structure_predictor import StructurePredictor, StructurePredictorConfig
from .tokenization.selfies_tokenizer import SELFIESTokenizer, TokenizerConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def generate_command(args) -> int:
    """Execute protein generation command."""
    print(f"🧬 Generating {args.num_samples} protein scaffolds...")
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ProteinDiffuserConfig(**config_dict)
    else:
        config = ProteinDiffuserConfig()
    
    # Override config with command line arguments
    if args.num_samples:
        config.num_samples = args.num_samples
    if args.max_length:
        config.max_length = args.max_length
    if args.temperature:
        config.temperature = args.temperature
    if args.device:
        config.device = args.device
    
    try:
        # Initialize diffuser
        if args.checkpoint:
            diffuser = ProteinDiffuser.from_pretrained(args.checkpoint, config)
            print(f"✅ Loaded model from {args.checkpoint}")
        else:
            diffuser = ProteinDiffuser(config)
            print("⚠️  Using randomly initialized model (no checkpoint provided)")
        
        # Generate sequences
        results = diffuser.generate(
            motif=args.motif,
            num_samples=args.num_samples,
            max_length=args.max_length,
            temperature=args.temperature,
            sampling_method=args.sampling_method,
            ddim_steps=args.ddim_steps,
            progress=not args.quiet,
        )
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sequences
        sequences_file = output_dir / "generated_sequences.txt"
        with open(sequences_file, 'w') as f:
            for i, result in enumerate(results):
                f.write(f">sequence_{i}\n{result['sequence']}\n")
        
        # Save detailed results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        json_result[key] = value.tolist()
                    else:
                        json_result[key] = value
                json_results.append(json_result)
            
            json.dump({
                "config": config.__dict__,
                "results": json_results,
                "summary": {
                    "total_sequences": len(results),
                    "average_length": sum(len(r['sequence']) for r in results) / len(results),
                    "average_confidence": sum(r['confidence'] for r in results) / len(results),
                }
            }, f, indent=2)
        
        print(f"✅ Generated {len(results)} sequences")
        print(f"📁 Results saved to {output_dir}")
        print(f"📄 Sequences: {sequences_file}")
        print(f"📊 Details: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def rank_command(args) -> int:
    """Execute protein ranking command."""
    print(f"🏆 Ranking proteins from {args.input}...")
    
    # Load sequences
    sequences = []
    input_path = Path(args.input)
    
    if input_path.suffix == '.json':
        # Load from JSON results file
        with open(input_path, 'r') as f:
            data = json.load(f)
        sequences = [result['sequence'] for result in data.get('results', [])]
    else:
        # Load from FASTA or text file
        with open(input_path, 'r') as f:
            content = f.read().strip()
            if content.startswith('>'):
                # FASTA format
                lines = content.split('\n')
                for line in lines:
                    if not line.startswith('>') and line.strip():
                        sequences.append(line.strip())
            else:
                # Simple text format
                sequences = [line.strip() for line in content.split('\n') if line.strip()]
    
    if not sequences:
        print("❌ No sequences found in input file")
        return 1
    
    print(f"📊 Found {len(sequences)} sequences to rank")
    
    try:
        # Initialize ranker
        ranker_config = AffinityRankerConfig()
        if args.max_results:
            ranker_config.max_results = args.max_results
        
        ranker = AffinityRanker(ranker_config)
        
        # Rank sequences
        ranked_results = ranker.rank(
            sequences,
            target_pdb=args.target_pdb,
            return_detailed=True,
        )
        
        # Save ranking results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save top sequences
        top_sequences_file = output_dir / "top_sequences.txt"
        with open(top_sequences_file, 'w') as f:
            for i, result in enumerate(ranked_results[:args.top_k]):
                f.write(f">rank_{i+1}_score_{result['composite_score']:.4f}\n")
                f.write(f"{result['sequence']}\n")
        
        # Save detailed rankings
        rankings_file = output_dir / "rankings.json"
        with open(rankings_file, 'w') as f:
            json.dump({
                "config": ranker_config.__dict__,
                "rankings": ranked_results,
                "statistics": ranker.get_ranking_statistics(ranked_results),
            }, f, indent=2)
        
        # Print summary
        print(f"✅ Ranked {len(ranked_results)} sequences")
        print(f"🥇 Top sequence score: {ranked_results[0]['composite_score']:.4f}")
        print(f"📁 Results saved to {output_dir}")
        print(f"🔝 Top sequences: {top_sequences_file}")
        print(f"📊 Full rankings: {rankings_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Ranking failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def fold_command(args) -> int:
    """Execute structure prediction command."""
    print(f"🧬 Predicting structures from {args.input}...")
    
    # Load sequences
    sequences = []
    input_path = Path(args.input)
    
    with open(input_path, 'r') as f:
        content = f.read().strip()
        if content.startswith('>'):
            # FASTA format
            lines = content.split('\n')
            for line in lines:
                if not line.startswith('>') and line.strip():
                    sequences.append(line.strip())
        else:
            # Simple text format
            sequences = [line.strip() for line in content.split('\n') if line.strip()]
    
    if not sequences:
        print("❌ No sequences found in input file")
        return 1
    
    print(f"📊 Found {len(sequences)} sequences for structure prediction")
    
    try:
        # Initialize structure predictor
        predictor_config = StructurePredictorConfig()
        predictor_config.method = args.method
        predictor_config.device = args.device
        predictor_config.save_structures = True
        predictor_config.output_dir = args.output
        
        predictor = StructurePredictor(predictor_config)
        
        # Predict structures
        results = predictor.batch_predict(sequences, progress=not args.quiet)
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "structure_predictions.json"
        with open(results_file, 'w') as f:
            json.dump({
                "config": predictor_config.__dict__,
                "results": results,
                "statistics": predictor.get_prediction_stats(results),
            }, f, indent=2)
        
        # Print summary
        successful_predictions = sum(1 for r in results if "error" not in r)
        print(f"✅ Predicted {successful_predictions}/{len(results)} structures successfully")
        print(f"📁 Results saved to {output_dir}")
        print(f"📊 Details: {results_file}")
        
        if successful_predictions > 0:
            avg_confidence = sum(r.get('confidence', 0) for r in results if 'error' not in r) / successful_predictions
            print(f"📈 Average confidence: {avg_confidence:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Structure prediction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def analyze_command(args) -> int:
    """Execute analysis command."""
    print(f"📊 Analyzing results from {args.input}...")
    
    try:
        # Load results
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        if 'results' in data:
            results = data['results']
        elif 'rankings' in data:
            results = data['rankings']
        else:
            print("❌ Invalid results format")
            return 1
        
        # Basic statistics
        print(f"\n📈 Analysis Summary:")
        print(f"Total sequences: {len(results)}")
        
        if results:
            # Sequence statistics
            sequences = [r.get('sequence', '') for r in results]
            lengths = [len(seq) for seq in sequences if seq]
            
            if lengths:
                print(f"Average length: {sum(lengths) / len(lengths):.1f}")
                print(f"Length range: {min(lengths)} - {max(lengths)}")
            
            # Quality statistics
            confidences = [r.get('confidence', 0) for r in results]
            if any(c > 0 for c in confidences):
                print(f"Average confidence: {sum(confidences) / len(confidences):.3f}")
                print(f"High confidence (>0.7): {sum(1 for c in confidences if c > 0.7)}")
            
            # Binding affinity statistics (if available)
            binding_affinities = [r.get('binding_affinity', 0) for r in results]
            if any(b != 0 for b in binding_affinities):
                valid_affinities = [b for b in binding_affinities if b != 0]
                print(f"Average binding affinity: {sum(valid_affinities) / len(valid_affinities):.2f} kcal/mol")
                print(f"Best binding affinity: {min(valid_affinities):.2f} kcal/mol")
        
        # Save analysis report
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = output_dir / "analysis_report.json"
            with open(report_file, 'w') as f:
                json.dump({
                    "input_file": str(args.input),
                    "total_sequences": len(results),
                    "statistics": {
                        "sequence_lengths": lengths if 'lengths' in locals() else [],
                        "confidences": confidences,
                        "binding_affinities": binding_affinities,
                    }
                }, f, indent=2)
            
            print(f"📁 Analysis report saved to {report_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def benchmark_command(args) -> int:
    """Execute benchmark command to test system performance."""
    print("🚀 Running system benchmarks...")
    
    try:
        # Initialize components
        config = ProteinDiffuserConfig()
        config.num_samples = 5  # Small benchmark
        config.max_length = 64
        
        diffuser = ProteinDiffuser(config)
        ranker = AffinityRanker()
        
        # Test generation speed
        import time
        start_time = time.time()
        
        results = diffuser.generate(
            motif="HELIX",
            num_samples=5,
            max_length=64,
            progress=False
        )
        
        gen_time = time.time() - start_time
        
        # Test ranking speed
        start_time = time.time()
        sequences = [r['sequence'] for r in results if 'sequence' in r]
        ranked = ranker.rank(sequences[:3], return_detailed=False)  # Rank subset
        rank_time = time.time() - start_time
        
        # Health check
        health = diffuser.health_check()
        
        # Print results
        print(f"\n📊 Benchmark Results:")
        print(f"Generation time: {gen_time:.2f}s ({len(results)} sequences)")
        print(f"Ranking time: {rank_time:.2f}s ({len(sequences)} sequences)")
        print(f"System health: {health['overall_status']}")
        
        if health['warnings']:
            print(f"⚠️  Warnings: {len(health['warnings'])}")
        if health['errors']:
            print(f"❌ Errors: {len(health['errors'])}")
            
        # Save benchmark results
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            benchmark_file = output_dir / "benchmark_results.json"
            with open(benchmark_file, 'w') as f:
                json.dump({
                    "generation_time": gen_time,
                    "ranking_time": rank_time,
                    "sequences_generated": len(results),
                    "sequences_ranked": len(sequences),
                    "health_check": health,
                    "system_info": {
                        "torch_available": TORCH_AVAILABLE,
                        "device": str(diffuser.device)
                    }
                }, f, indent=2)
            print(f"📁 Benchmark results saved to {benchmark_file}")
        
        return 0 if health['overall_status'] in ['healthy', 'degraded'] else 1
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def validate_command(args) -> int:
    """Execute validation command for input sequences."""
    print(f"🔍 Validating sequences from {args.input}...")
    
    try:
        # Load sequences
        sequences = []
        with open(args.input, 'r') as f:
            content = f.read().strip()
            if content.startswith('>'):
                # FASTA format
                lines = content.split('\n')
                for line in lines:
                    if not line.startswith('>') and line.strip():
                        sequences.append(line.strip())
            else:
                sequences = [line.strip() for line in content.split('\n') if line.strip()]
        
        print(f"📊 Validating {len(sequences)} sequences...")
        
        valid_sequences = 0
        issues = []
        
        for i, seq in enumerate(sequences):
            # Basic validation
            if not seq:
                issues.append(f"Sequence {i+1}: Empty sequence")
                continue
                
            if len(seq) < 5:
                issues.append(f"Sequence {i+1}: Too short ({len(seq)} residues)")
                continue
                
            # Check for valid amino acids
            valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
            invalid_aas = [aa for aa in seq.upper() if aa not in valid_aas]
            if invalid_aas:
                issues.append(f"Sequence {i+1}: Invalid amino acids: {set(invalid_aas)}")
                continue
                
            valid_sequences += 1
        
        # Print results
        print(f"✅ Valid sequences: {valid_sequences}/{len(sequences)}")
        if issues:
            print(f"⚠️  Issues found: {len(issues)}")
            if args.verbose:
                for issue in issues[:10]:  # Show first 10 issues
                    print(f"  • {issue}")
                if len(issues) > 10:
                    print(f"  ... and {len(issues) - 10} more issues")
        
        # Save validation report
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = output_dir / "validation_report.json"
            with open(report_file, 'w') as f:
                json.dump({
                    "total_sequences": len(sequences),
                    "valid_sequences": valid_sequences,
                    "invalid_sequences": len(sequences) - valid_sequences,
                    "issues": issues
                }, f, indent=2)
            print(f"📁 Validation report saved to {report_file}")
        
        return 0 if valid_sequences > 0 else 1
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def interactive_command(args) -> int:
    """Start interactive mode."""
    print("🎯 Starting interactive protein design session...")
    print("Type 'help' for commands, 'quit' to exit\n")
    
    # Initialize components
    try:
        config = ProteinDiffuserConfig()
        diffuser = ProteinDiffuser(config)
        ranker = AffinityRanker()
        print("✅ System initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return 1
    
    session_results = []
    
    while True:
        try:
            command = input("🧬 protein-diffusion> ").strip()
            
            if not command:
                continue
                
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            elif command.lower() in ['help', 'h']:
                print("""
📖 Available commands:
  generate [motif] [samples] - Generate protein scaffolds
  rank - Rank current session results
  show - Show current session results
  clear - Clear session results
  status - Show system status
  help - Show this help
  quit - Exit interactive mode
                """)
                
            elif command.lower().startswith('generate'):
                parts = command.split()
                motif = parts[1] if len(parts) > 1 else "HELIX"
                samples = int(parts[2]) if len(parts) > 2 else 5
                
                print(f"🔬 Generating {samples} sequences with motif '{motif}'...")
                results = diffuser.generate(
                    motif=motif,
                    num_samples=samples,
                    max_length=128,
                    progress=False
                )
                session_results.extend(results)
                print(f"✅ Generated {len(results)} sequences (total: {len(session_results)})")
                
            elif command.lower() == 'rank':
                if not session_results:
                    print("⚠️  No sequences to rank. Generate some first.")
                    continue
                    
                print(f"🏆 Ranking {len(session_results)} sequences...")
                sequences = [r.get('sequence', '') for r in session_results if r.get('sequence')]
                ranked = ranker.get_top_candidates(sequences, top_k=10)
                
                print("\n🥇 Top sequences:")
                for i, (seq, score) in enumerate(ranked[:5]):
                    print(f"  {i+1}. Score: {score:.4f} | {seq[:50]}...")
                    
            elif command.lower() == 'show':
                print(f"\n📊 Session summary: {len(session_results)} sequences")
                if session_results:
                    avg_length = sum(len(r.get('sequence', '')) for r in session_results) / len(session_results)
                    avg_conf = sum(r.get('confidence', 0) for r in session_results) / len(session_results)
                    print(f"Average length: {avg_length:.1f}")
                    print(f"Average confidence: {avg_conf:.3f}")
                    
            elif command.lower() == 'clear':
                session_results.clear()
                print("🗑️  Session results cleared")
                
            elif command.lower() == 'status':
                health = diffuser.health_check()
                print(f"\n🔧 System status: {health['overall_status']}")
                if health['warnings']:
                    print(f"⚠️  Warnings: {len(health['warnings'])}")
                if health['errors']:
                    print(f"❌ Errors: {len(health['errors'])}")
                    
            else:
                print(f"❓ Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Protein Diffusion Design Lab - Generate and analyze protein scaffolds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 protein scaffolds with helix motif
  protein-diffusion generate --motif HELIX_SHEET_HELIX --num-samples 100 --output ./results

  # Rank generated proteins by binding affinity
  protein-diffusion rank --input ./results/generated_sequences.txt --target-pdb target.pdb --output ./rankings

  # Predict structures for top candidates
  protein-diffusion fold --input ./rankings/top_sequences.txt --method esmfold --output ./structures

  # Analyze results
  protein-diffusion analyze --input ./results/results.json
  
  # Run system benchmarks
  protein-diffusion benchmark --output ./benchmark
  
  # Start interactive mode
  protein-diffusion interactive
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate protein scaffolds')
    gen_parser.add_argument('--checkpoint', '-c', help='Path to model checkpoint')
    gen_parser.add_argument('--config', help='Path to config JSON file')
    gen_parser.add_argument('--motif', '-m', help='Target motif (e.g., HELIX_SHEET_HELIX)')
    gen_parser.add_argument('--num-samples', '-n', type=int, default=10, help='Number of samples to generate')
    gen_parser.add_argument('--max-length', type=int, default=256, help='Maximum sequence length')
    gen_parser.add_argument('--temperature', '-t', type=float, default=1.0, help='Sampling temperature')
    gen_parser.add_argument('--sampling-method', choices=['ddpm', 'ddim'], default='ddpm', help='Sampling method')
    gen_parser.add_argument('--ddim-steps', type=int, default=50, help='Number of DDIM steps')
    gen_parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda)')
    gen_parser.add_argument('--output', '-o', default='./output', help='Output directory')
    gen_parser.set_defaults(func=generate_command)
    
    # Rank command
    rank_parser = subparsers.add_parser('rank', help='Rank protein sequences')
    rank_parser.add_argument('--input', '-i', required=True, help='Input sequences file')
    rank_parser.add_argument('--target-pdb', help='Target PDB file for binding evaluation')
    rank_parser.add_argument('--max-results', type=int, default=100, help='Maximum results to return')
    rank_parser.add_argument('--top-k', type=int, default=20, help='Number of top sequences to save')
    rank_parser.add_argument('--output', '-o', default='./rankings', help='Output directory')
    rank_parser.set_defaults(func=rank_command)
    
    # Fold command
    fold_parser = subparsers.add_parser('fold', help='Predict protein structures')
    fold_parser.add_argument('--input', '-i', required=True, help='Input sequences file')
    fold_parser.add_argument('--method', choices=['esmfold', 'colabfold'], default='esmfold', help='Prediction method')
    fold_parser.add_argument('--device', default='auto', help='Device to use')
    fold_parser.add_argument('--output', '-o', default='./structures', help='Output directory')
    fold_parser.set_defaults(func=fold_command)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('--input', '-i', required=True, help='Input results file (JSON)')
    analyze_parser.add_argument('--output', '-o', help='Output directory for analysis report')
    analyze_parser.set_defaults(func=analyze_command)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run system benchmarks')
    benchmark_parser.add_argument('--output', '-o', help='Output directory for benchmark results')
    benchmark_parser.set_defaults(func=benchmark_command)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate input sequences')
    validate_parser.add_argument('--input', '-i', required=True, help='Input sequences file')
    validate_parser.add_argument('--output', '-o', help='Output directory for validation report')
    validate_parser.set_defaults(func=validate_command)
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    interactive_parser.set_defaults(func=interactive_command)
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle device selection
    if hasattr(args, 'device') and args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Execute command
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())