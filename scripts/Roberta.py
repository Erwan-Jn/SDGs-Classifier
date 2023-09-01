import numpy as np
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from scripts.utils import DataProcess
from datasets import DatasetDict, Dataset

import tensorflow
from tensorflow.keras.callbacks import EarlyStopping

dp = DataProcess()
df = dp.clean_data_short()

train = df.loc[:25000,["cleaned_text", "sdg"]]
val = df.loc[25000:35000,["cleaned_text", "sdg"]]
test = df.loc[35000:,["cleaned_text", "sdg"]]

data_dict_train = {'text': train["cleaned_text"], 'labels': train["sdg"].astype(int) -1}
data_dict_val = {'text': val["cleaned_text"], 'labels': val["sdg"].astype(int) -1}
data_dict_test = {'text': test["cleaned_text"], 'labels': test["sdg"].astype(int) -1}

ds_train = Dataset.from_dict(data_dict_train)
ds_val = Dataset.from_dict(data_dict_val)
ds_test = Dataset.from_dict(data_dict_test)

ds_final = DatasetDict({
    'train': ds_train,
    'test': ds_test,
    'val': ds_val})

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tokenized_ds_final = ds_final.map(preprocess_function, batched=True)

id2label = dict(zip([int(num) for num in np.arange(0, 16)],
            ["Poverty", "Hunger", "Health", "Education", "Gender",
             "Water", "Clean", "Work", "Inno", "Inequalities", "Cities",
             "Cons&Prod", "Climate", "OceanLife", "LandLife", "Peace"]
        ))
label2id = {v: k for k, v in id2label.items()}

model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=16, id2label=id2label, label2id=label2id)

# Set the classifier layer to be trainable
model.classifier.trainable = True
# Make the BERT layers non-trainable
for layer in model.layers:
    if "bert" in layer.name:
        layer.trainable = False

tf_train_set = model.prepare_tf_dataset(
    tokenized_ds_final["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    tokenized_ds_final["test"],
    shuffle=False,
    batch_size=32,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_ds_final["val"],
    shuffle=False,
    batch_size=32,
    collate_fn=data_collator,
)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

es = EarlyStopping(patience=10)

history = model.fit(x=tf_train_set,
                    validation_data=tf_validation_set,
                    epochs=20, callbacks=[es],
                    verbose=1)
