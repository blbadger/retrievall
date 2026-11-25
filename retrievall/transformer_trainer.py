import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

from datasets import load_dataset, load_from_disk
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import safetensors
import datasets
import warnings
import shutil
from dotenv import load_dotenv
from safetensors.torch import load_model
from infonce_ngram_retrieval import RetrievalTransformer, RetrievalAutoencoderTransformer

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer
from hamming_metric import compute_hamming_metric, preprocess_logits_for_metrics
warnings.filterwarnings(action='ignore')

class TransformerCLM(nn.Module):
    def __init__(self, model, vocab_size=8000, dim=512):
        super().__init__()
        self.model = model # a LlamaModel object
        self.lm_head = nn.Linear(dim, vocab_size)
        self.vocab_size = vocab_size
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input_ids, labels=None, attention_mask=None):
        output = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        labels = labels[:, 1:].contiguous()
        logits = self.lm_head(output)
        shift_logits = logits[:, :-1, :].contiguous() # b t e 
        if labels is not None:
            logits = shift_logits.view(-1, self.vocab_size)
            labels = labels.view(-1)
            loss = self.loss_fn(logits, labels)
            print (torch.sum(loss)/loss.shape[0])
            return torch.sum(loss), logits

        else:
            return logits

def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		params = parameter.numel()
		table.add_row([name, params])
		total_params += params
	print(table)
	print(f"Total Trainable Params: {total_params}")
	return total_params

def half_data(example):
    example['input_ids'] = example['input_ids'][:256]
    if 'attention_mask' in example:
        example['attention_mask'] = example['attention_mask'][:256]
    if 'labels' in example:
        example['labels'] = example['labels'][:256]

    return example

def tokenize_and_preprocess(example):
        text = example['text']
        tokens = tokenizer(text, max_length=256, padding='max_length', truncation=True) # return list, not tensor
        example['input_ids'] = tokens['input_ids']
        example['attention_mask'] = tokens['attention_mask']
        return example

def half_data(example):
    example['input_ids'] = example['input_ids'][:256]
    if 'attention_mask' in example:
        example['attention_mask'] = example['attention_mask'][:256]
    if 'labels' in example:
        example['labels'] = example['labels'][:256]

    return example

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

encoder_dim = 512
decoder_dim = 512
context_length = 1024
compression = 1
n_layers = 16
n_heads = 4

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
vocab_size = len(tokenizer)
llama_config_kwargs = {
    'hidden_size':encoder_dim,
    'intermediate_size': 4*encoder_dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': n_heads,
    'vocab_size': vocab_size
}
print (llama_config_kwargs)
# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
autoencoder = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=tokenized_length, compression=1, freeze_encoder=False)
#load_model(autoencoder, '/home/bbadger/Desktop/fineweb_autoencoding_transformer_512c1_d512_n8_c256_b32x4/checkpoint-200000/model.safetensors')
model = RetrievalAutoencoderTransformer(autoencoder, ngram_size=ngram, padding_side='right', pad_token_id=tokenizer.pad_token_id)

safetensors.torch.load_model(model, '/home/bbadger/Desktop/fineweb_pretrainedauto_7gram_infonce_512_n8_c256_b32x4/checkpoint-200000/model.safetensors')
decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, model.model, decoder_model, tokenized_length=context_length, compression=compression, freeze_encoder=True)


# unrolled embedding transformer autoencoder
#encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
#decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
#model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, compression=compression, freeze_encoder=False)
#load_model(model, f'{checkpoint_root}/fineweb_autoencoding_transformer_512c1_d512_n8_c512_b128x1/checkpoint-200000/model.safetensors')

#decoder_dim=512
#encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
#decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
#autoencoder = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=tokenized_length, compression=1, freeze_encoder=False)
#load_model(autoencoder, '/home/bbadger/Desktop/fineweb_autoencoding_transformer_512c1_d512_n8_c256_b32x4/checkpoint-200000/model.safetensors')
#model = RetrievalAutoencoderTransformer(autoencoder, ngram_size=ngram, padding_side='right', pad_token_id=tokenizer.pad_token_id)

#safetensors.torch.load_model(model, '/home/bbadger/Desktop/fineweb_pretrainedauto_7gram_infonce_512_n8_c256_b32x4/checkpoint-200000/model.safetensors')
#decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
#model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, model.model, decoder_model, tokenized_length=context_length, compression=compression, freeze_encoder=True)

#llama_config_kwargs = { 
#    'hidden_size':decoder_dim//2,
#    'intermediate_size': 4*decoder_dim//2,
#    'num_hidden_layers': n_layers,
#    'num_attention_heads': 4,
#    'vocab_size': vocab_size
#}

#decoder_configuration = LlamaConfig(**llama_config_kwargs)
#encoder_model = AbbreviatedModel(encoder_model.encoder, tokenized_length=context_length)
#encoder_model = AbbreviatedModel(model.encoder, tokenized_length=512)
#decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=512)
#model = UnrolledAutoencodingTransformer(vocab_size, encoder_dim, encoder_model, decoder_model, tokenized_length=512, compression=compression, freeze_encoder=True)




tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = f"{data_root}/gsm-tokenized-train-c1024-8k-debatched"
test_path = f"{data_root}/gsm-tokenized-test-c1024-8k-debatched"

train_dataset = load_from_disk(train_path).map(half_data, batched=False, num_proc=12)
test_dataset = load_from_disk(test_path).map(half_data, batched=False, num_proc=12)
print (len(train_dataset[0]['input_ids']))

batch_size = 32

# get number of devices (assumes that all visible devices are used for training)
n_devices = 1
if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/tinygsm_transformer\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	gradient_accumulation_steps=1,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=10000,
	learning_rate=2e-4, 
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=200000,
	torch_compile=True
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	compute_metrics=compute_hamming_metric,
	preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

print (f"training model, saving to {output_dir}")
# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f"training begun: saving results in {output_dir}")
print (trainer.evaluate())
model.train()
trainer.train()
#trainer.train(output_dir + '/checkpoint-180000')