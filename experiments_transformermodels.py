###
# experiments_transformermodels.py
#
# Routines for MoE experiments on model architectures as defined
# in the HuggingFace transformers package.
# Dylan Everingham
# 10.02.2026
###

from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from experiments_transformermodels import *
from data import*

################################################################################

def run_experiment_qwen3_moe(n_samples, enable_wandb=False):
    """
    Executes a configurable training run of a Qwen3MoE model,
    saves the resulting model, evaluates it with lm_eval,
    and logs the results.
    enable_wandb: if true, logs run to Weights and Biases.
    """
    
    # Get tokenizer (Qwen2.5-0.5B)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Get data (FineWeb-Edu)
    tokenized_dataset= get_data_finewebedu(tokenizer, enable_wandb=False)
    
    # Configure and initialize model
    model_config = Qwen3MoeConfig(
        vocab_size = len(tokenizer),
        hidden_size = 64,
        intermediate_size = 64,      # Expert embedding size
        num_hidden_layers = 12,      # i.e. number of expert layers / routers
        num_attention_heads = 16,
        num_key_value_heads = 4,
        num_experts = 8,             # Experts per layer
        num_experts_per_tok = 2,     # Top-k experts activated per token
        moe_intermediate_size = 64, 
        output_router_logits = True, # Needed for additional aux loss calculation
        router_aux_loss_coef = 0.00  # Coefficient for load balancing loss
    )
    model = Qwen3MoeForCausalLM(model_config)
    print(f"Model Parameters: {model.num_parameters():,}")
    
    # Train
    training_args = TrainingArguments(
        output_dir="./qwen3moe",
        per_device_train_batch_size=1,   # Adjust based on VRAM
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=5e-4,
        max_steps=n_samples,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        fp16=True, # Use mixed precision
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    #timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    #wandb_id = f"{model_type}-{timestamp}"

    # Start Training
    trainer.train()

    # Save the final model for evaluation
    trainer.save_model("./qwen3moe-test")