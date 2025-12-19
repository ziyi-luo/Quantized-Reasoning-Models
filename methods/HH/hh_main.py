"""
Complete Full-Model Quantization Script for Householder Quantization.
Matches FlatQuant's structure but uses Householder rotations instead of Kronecker.
"""
import os
import torch
import transformers
import sys

# Add necessary paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Quantized-Reasoning-Models', 'methods'))

from HH import args_utils, log_utils, hh_model_utils, hh_train_utils, data_utils

def main():
    args, logger = args_utils.parser_gen()
    log_utils.seed_everything(seed=args.seed)
    
    # Load model
    model, apply_hh_to_model = hh_model_utils.get_model(args.model, args.seqlen, args.hf_token)
    model.eval()
    logger.info(f"Loaded model: {args.model}")
    
    # Get calibration data
    trainloader = data_utils.get_loaders(
        args.cali_dataset, nsamples=args.nsamples,
        seed=args.seed, model_path=args.model,
        seqlen=model.seqlen
    )
    logger.info(f"Loaded calibration data. First sample shape: {trainloader[0].shape}")
    
    if args.quantize:
        # Apply HH transformation to model
        model = apply_hh_to_model(args, model)
        logger.info("Applied HH transformation to model")
        
        # Calibrate and optimize rotations
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        hh_train_utils.cali_hh_quant(args, model, trainloader, device, logger=logger)
        logger.info("Finished calibration and optimization")
        
        # Save rotation matrices
        if args.save_matrix:
            os.makedirs(args.exp_dir, exist_ok=True)
            hh_params = {}
            for name, module in model.named_modules():
                if hasattr(module, 'export_wy_and_scale'):
                    W, Y, scale = module.export_wy_and_scale()
                    hh_params[name] = {'W': W, 'Y': Y, 'scale': scale}
            
            save_path = os.path.join(args.exp_dir, "hh_matrices.pth")
            torch.save(hh_params, save_path)
            logger.info(f"Saved HH matrices to {save_path}")
    
    # Evaluate perplexity
    if args.eval_ppl:
        for eval_dataset in ["wikitext2"]:
            logger.info(f"Evaluating on {eval_dataset}")
            testloader = data_utils.get_loaders(
                eval_dataset, nsamples=args.nsamples, seed=args.seed,
                model_path=args.model, seqlen=model.seqlen
            )
            # Use FlatQuant's eval_utils if available, else skip or mock
            try:
                from flatquant.flatquant import eval_utils
                ppl = eval_utils.ppl_eval(model, testloader)
                logger.info(f"{eval_dataset} PPL: {ppl}")
            except ImportError:
                logger.warning("flatquant.eval_utils not found, skipping PPL evaluation")

if __name__ == '__main__':
    main()
