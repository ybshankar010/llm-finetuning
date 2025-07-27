def cleanup_gpu_memory():
    """
    One-stop function to clean GPU memory in Jupyter notebooks
    Run this before loading new models or when you get OOM errors
    """
    import torch
    import gc
    import sys
    
    print("🧹 Starting GPU memory cleanup...")
    
    # Step 1: Get current frame to access notebook variables
    frame = sys._getframe(1)
    
    # Step 2: Delete common model variables from both globals and locals
    model_vars = [
        'model', 'base_model', 'finetuning_model', 'trainer', 'tokenizer', 
        'lora_model', 'peft_model', 'quantized_model', 'model_with_head',
        'xgb_base', 'xgb_ft', 'embeddings', 'outputs'
    ]
    
    deleted_vars = []
    
    # Delete from notebook globals
    for var_name in model_vars:
        if var_name in globals():
            del globals()[var_name]
            deleted_vars.append(f"global.{var_name}")
    
    # Delete from current cell locals
    if hasattr(frame, 'f_locals'):
        for var_name in model_vars:
            if var_name in frame.f_locals:
                frame.f_locals[var_name] = None
                deleted_vars.append(f"local.{var_name}")
    
    if deleted_vars:
        print(f"   ✅ Deleted variables: {', '.join(deleted_vars)}")
    else:
        print("   ℹ️  No model variables found to delete")
    
    # Step 3: Force garbage collection (multiple rounds)
    print("   🗑️  Running garbage collection...")
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            print(f"      Round {i+1}: collected {collected} objects")
    
    # Step 4: Clear CUDA cache aggressively
    if torch.cuda.is_available():
        print("   🔥 Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Clear twice for stubborn memory
        torch.cuda.ipc_collect()  # Clear inter-process memory
        
        # Step 5: Show memory status
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        free_memory = total_memory - allocated_memory
        
        print("   📊 GPU Memory Status:")
        print(f"      Total: {total_memory:.2f} GB")
        print(f"      Allocated: {allocated_memory:.2f} GB ({allocated_memory/total_memory*100:.1f}%)")
        print(f"      Reserved: {reserved_memory:.2f} GB")
        print(f"      Available: {free_memory:.2f} GB")
        
        # Step 6: Give recommendation
        if allocated_memory < 1.0:
            print("   ✅ Memory successfully cleaned! Safe to load new models.")
        elif allocated_memory < 2.0:
            print("   ⚠️  Some memory still allocated. Should be OK for most models.")
        else:
            print("   ❌ High memory usage detected. Consider restarting kernel.")
            
        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'free': free_memory,
            'success': allocated_memory < 1.0
        }
    else:
        print("   ❌ CUDA not available")
        return {'success': False}


def check_if_model_has_peft(model):
    """
    Check if model already has PEFT/LoRA adapters
    """
    # Check for PEFT attributes
    has_peft_config = hasattr(model, 'peft_config')
    has_peft_modules = hasattr(model, 'peft_modules')
    has_base_model = hasattr(model, 'base_model')
    
    # Check model class name
    is_peft_model = 'Peft' in model.__class__.__name__
    
    print(f"🔍 Model PEFT Status Check:")
    print(f"   Model class: {model.__class__.__name__}")
    print(f"   Has peft_config: {has_peft_config}")
    print(f"   Has peft_modules: {has_peft_modules}")
    print(f"   Has base_model: {has_base_model}")
    print(f"   Is PEFT model: {is_peft_model}")
    
    already_has_peft = any([has_peft_config, has_peft_modules, has_base_model, is_peft_model])
    
    if already_has_peft:
        print("   ⚠️  Model ALREADY has PEFT/LoRA adapters!")
        return True
    else:
        print("   ✅ Model is clean, ready for PEFT application")
        return False

