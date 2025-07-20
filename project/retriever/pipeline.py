from model import SiglipStyleModel, ColSentenceModel
import os
from transformers import Trainer, TrainingArguments
from callbacks import NotebookProgressCallbackNoTable
from transformers.utils.notebook import NotebookProgressCallback
from evaluation import compute_metrics
import wandb
from utils import get_train_and_test_data, collate_fn

# loss = "siglip"
loss = "clip"
batch_size = 128
# epochs = 2 * batch_size
epochs = 10
lr = 1e-6
eval_batch = 250
architecture = "Siglip"

models = {
    "ColSent": ColSentenceModel,
    "Siglip": SiglipStyleModel,
}
# model = RetrieverModel(loss_type=loss)
model = models[architecture](loss_type=loss)

data_paths = [
    ("microsoft/ms_marco", "v1.1"),
    ("microsoft/ms_marco", "v2.1"),
]
train_data, test_data = get_train_and_test_data(data_paths)

lr_n = "" if lr == 1e-7 else f"lr{lr:.0E}_"
b_n = "" if batch_size == 2 else f"b{batch_size}_"

model_name = model.model_name.split("/")[-1]
model_path = f"{loss}/{architecture}/{model_name}/{b_n}{lr_n}{data_paths[0][0]}{data_paths[0][1]}"
print(model_path)
print(train_data)
print(test_data)

os.environ["WANDB_PROJECT"] = "MSE"
os.environ["WANDB_LOG_MODEL"] = "false"
#wandb.init(entity="mse-jan-simon", name=model_path)

training_args = TrainingArguments(
    output_dir="models/" + model_path,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    learning_rate=lr,
    save_total_limit=1,
    remove_unused_columns=False,
    bf16=True,
    optim="adamw_bnb_8bit",
    logging_steps=100,
    eval_steps=200,
    eval_strategy="steps",
    eval_on_start=True,
    per_device_eval_batch_size=eval_batch,
    run_name=model_path,
    report_to='none',
    save_strategy="steps",
    save_steps=400,
    load_best_model_at_end=True,
    metric_for_best_model=f"eval_{data_paths[-1][0]}{data_paths[-1][1]}_best_score",
    max_steps=4000,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

trainer.remove_callback(NotebookProgressCallback)
trainer.add_callback(NotebookProgressCallbackNoTable)

# trainer.evaluate()
trainer.train()
# try: trainer.train(resume_from_checkpoint=True)
# except: trainer.train(resume_from_checkpoint=False)
trainer.save_model(output_dir=model_path)