import torch

def extract_non_default_optimizer_params(optimizer):
    # Function generated fully by an LLM
    # Optimizer class name (e.g., "Adam")
    opt_name = optimizer.__class__.__name__
    
    # Defaults defined by the optimizer
    defaults = optimizer.defaults
    
    # Parameters actually used (take from first param group)
    # Assumes same settings across groups
    current_params = optimizer.param_groups[0]
    
    # Keep only parameters that differ from defaults
    non_default = {
        k: v for k, v in current_params.items()
        if k in defaults and defaults[k] != v
    }
    
    return opt_name, non_default


import inspect

def extract_non_default_loss_params(loss):
    loss_name = loss.__class__.__name__
    
    sig = inspect.signature(loss.__class__.__init__)
    defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }
    
    # Get actual attribute values from the loss object
    current_params = {
        k: getattr(loss, k)
        for k in defaults.keys()
        if hasattr(loss, k)
    }
    
    # Keep only changed ones. Convert tensors to strings 
    non_default = {
        k: str(v) if isinstance(v, torch.Tensor) else v for k, v in current_params.items()
        if defaults[k] != v
    }
    
    return loss_name, non_default