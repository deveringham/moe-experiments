###
# experiments_transformermodels.py
#
# Routines for MoE experiments on model architectures as defined
# in the HuggingFace transformers package.
# Dylan Everingham
# 10.02.2026
###

import time
import datetime
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, AutoModelForCausalLM
from experiments_transformermodels import *
import lm_eval
from lm_eval.models.huggingface import HFLM
import wandb
from data import*

log_dir = "./logs/"

################################################################################

def run_experiment_qwen3_moe(n_samples_train, n_samples_val,
                             hidden_size=64, intermediate_size=64, 
                             n_hidden_layers=12, n_attention_heads=4, n_kv_heads=4,
                             n_experts=8, k=2, moe_intermediate_size=64,
                             w_load_balancing=0.0,
                             batch_size=1, learning_rate=5e-4, n_epoch=1,
                             enable_wandb=False):
    """
    Executes a configurable training run of a Qwen3MoE model,
    saves the resulting model, evaluates it with lm_eval,
    and logs the results.
    enable_wandb: if true, logs run to Weights and Biases.
    """
    
    # Get tokenizer (Qwen2.5-0.5B)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Get data (FineWeb-Edu)
    tokenized_dataset = get_data_finewebedu(tokenizer, n_samples=n_samples_train, enable_wandb=False)
    
    # Configure and initialize model
    model_config = Qwen3MoeConfig(
        vocab_size = len(tokenizer),
        hidden_size = hidden_size,
        intermediate_size = intermediate_size,      # Expert embedding size
        num_hidden_layers = n_hidden_layers,        # i.e. number of expert layers / routers
        num_attention_heads = n_attention_heads,
        num_key_value_heads = n_kv_heads,
        num_experts = n_experts,                    # Experts per layer
        num_experts_per_tok = k,                    # Top-k experts activated per token
        moe_intermediate_size = moe_intermediate_size, 
        output_router_logits = True,                            # Needed for additional aux loss calculation
        router_aux_loss_coef = w_load_balancing                 # Coefficient for load balancing loss
    )
    model = Qwen3MoeForCausalLM(model_config)
    print(f"Model Parameters: {model.num_parameters():,}")
    
    # Get unique string id for the run
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    run_id = f"qwen3moe-{timestamp}"
    
    # Configure Weights and Biases
    if enable_wandb:
        wandb_id = run_id
        
        # Populate config dict with hyperparameters and metadata
        wandb_config = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'n_epoch': n_epochs,
            'n_samples_train': n_samples_train,
            'n_samples_val': n_samples_val,
            'architecture': 'qwen3moe',
            'data': 'fineweb-edu',
            'vocab_size': len(tokenizer),
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'num_hidden_layers': n_attention_heads,
            'num_attention_heads': n_attention_heads,
            'num_key_value_heads': n_kv_heads,
            'num_experts': n_experts,
            'k': k,
            'moe_intermediate_size': moe_intermediate_size, 
            'router_aux_loss_coef': w_load_balancing,
            'output_dir': log_dir+run_id,
            'per_device_train_batch_size': 1,
            'n_epoch': n_epoch,
            'n_params': model.num_parameters(),
        }
        
        wandb_run = wandb.init(
            entity="dceveringham-technical-university-of-berlin",
            project="moe-experiments",
            id=wandb_id,
            config=wandb_config,
        )
    else:
        wandb_run = None
    
    # Configure training
    if enable_wandb:
        report_to = "wandb"
        run_name = wandb_id
    else:
        report_to = "none"
        run_name = None
    training_args = TrainingArguments(
        output_dir=log_dir+run_id,
        per_device_train_batch_size=batch_size,   # Adjust based on VRAM
        gradient_accumulation_steps=min(16,batch_size),
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        max_steps=n_samples_train,
        num_train_epochs=n_epoch,
        logging_steps=100,
        save_steps=1000,
        fp16=True, # Use mixed precision
        report_to=report_to,
        run_name=run_name,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Keep time
    start_time = time.time()
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    
    # Stop timing
    end_time = time.time()
    if enable_wandb:
        wandb_run.log({'train_time': end_time-start_time})
    
    # Save the final model for evaluation
    trainer.save_model(log_dir+run_id)
                       
    # Load the model for evaluation
    loaded_model = AutoModelForCausalLM.from_pretrained(
        log_dir+run_id,
        dtype="auto",
        device_map="auto",
        trust_remote_code=False
    )
    eval_model = HFLM(
        pretrained=loaded_model, 
        #device=device,
        batch_size=8
    )

    # Choose evaluation tasks (Hellaswag for commonsense, ARC-Easy for reasoning)
    tasks = ["hellaswag", "arc_easy"]

    # Run evaluation
    val_results = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=tasks,
        num_fewshot=0,           # Zero-shot evaluation
        limit=n_samples_val      # Limit samples for quick testing
    )
    
    if enable_wandb:
        # Log evalulation metrics
        for task, metrics in val_results["results"].items():
            wandb_run.log({f"{task}": metrics})
        
        # Finish the wandb run and upload any remaining data
        wandb_run.finish()
    
    return val_results