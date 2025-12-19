import torch
import random
import logging
logger = logging.getLogger(__name__)

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model_path=None):
    """
    Get calibration data loaders.
    """
    logger.info(f"Loading dataset {name}...")
    random.seed(seed)
    
    # Try loading real dataset
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if name == 'wikitext2':
            traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            # Tokenize
            trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
            
            # Sample chunks
            trainloader = []
            for _ in range(nsamples):
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                trainloader.append(inp)
                
            return trainloader
    except Exception as e:
        logger.warning(f"Failed to load real dataset: {e}. Using Random Data.")
        
    # Fallback to random
    trainloader = []
    vocab_size = 32000 # Guess
    for _ in range(nsamples):
        inp = torch.randint(0, vocab_size, (1, seqlen))
        trainloader.append(inp)
        
    return trainloader
