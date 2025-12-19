"""
Argument parser for HH quantization.
"""
import argparse
import logging
import os

def parser_gen():
    parser = argparse.ArgumentParser()
    
    # Model args
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token')
    
    # Quantization args
    parser.add_argument('--quantize', action='store_true', help='Enable quantization')
    parser.add_argument('--w_bits', type=int, default=4, help='Weight bits')
    parser.add_argument('--a_bits', type=int, default=8, help='Activation bits')
    parser.add_argument('--a_asym', action='store_true', help='Use asymmetric activation quantization')
    
    parser.add_argument('--k_bits', type=int, default=8, help='Key bits')
    parser.add_argument('--k_asym', action='store_true', help='Use asymmetric key quantization')
    parser.add_argument('--k_groupsize', type=int, default=-1, help='Groupsize for key quantization')
    
    parser.add_argument('--v_bits', type=int, default=8, help='Value bits')
    parser.add_argument('--v_asym', action='store_true', help='Use asymmetric value quantization')
    parser.add_argument('--v_groupsize', type=int, default=-1, help='Groupsize for value quantization')
    
    # Calibration args
    parser.add_argument('--cali_dataset', type=str, default='wikitext2', help='Calibration dataset')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--cali_bsz', type=int, default=4, help='Calibration batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per layer')
    parser.add_argument('--hh_lr', type=float, default=1e-3, help='Learning rate for HH params')
    parser.add_argument('--deactive_amp', action='store_true', help='Disable AMP')
    
    # Save/Load args
    parser.add_argument('--exp_dir', type=str, default='./hh_exp', help='Experiment directory')
    parser.add_argument('--save_matrix', action='store_true', help='Save rotation matrices')
    
    # Eval args
    parser.add_argument('--eval_ppl', action='store_true', help='Evaluate perplexity')
    
    args = parser.parse_args()
    
    # Setup logger
    os.makedirs(args.exp_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.exp_dir, 'hh_quant.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    return args, logger
