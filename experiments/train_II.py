import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from transformers import AdamW
import pandas as pd
import numpy as np

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Load the Excel file
file_path = r'/home2/hardik_2001cs27/utad/experiments/results.xlsx'  # Change this to your file's path
df = pd.read_excel(file_path)

# Extract a specific column (e.g., 'Name')
normal3 = 'Input Sentence'  # Change this to the column you want to extract
corrupted3 = 'Reconstructed Sentence'
maskin3 = 'Masking Pattern'
normal2 = df[normal3]
corrupted2 = df[corrupted3]
maskin2 = df[maskin3]

# Convert to a list or numpy array
normal = normal2.tolist()  # Convert to a list
corrupted = corrupted2.tolist()
maskin = maskin2.tolist()
masking_patterns = []
for m in maskin:
    mm=[]
    for ch in m:
        ch2 = int(ch)
        mm.append(ch2)
    masking_patterns.append(mm)

# Sample data
original_sentences = normal
corrupted_sentences = corrupted
labels = masking_patterns
# Create a custom dataset class
class TokenClassificationDataset(Dataset):
    def __init__(self, original_sentences, corrupted_sentences, labels, tokenizer):
        self.original_sentences = original_sentences
        self.corrupted_sentences = corrupted_sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.original_sentences)

    def __getitem__(self, idx):
        original_sentence = self.original_sentences[idx]
        label = self.labels[idx]
        
        # Tokenize original sentence and create input ids and attention masks
        encoding = self.tokenizer(
            original_sentence,
            max_length=100,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Use the label to align with the tokenized sequence
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create a tensor of the same length as the tokenized input, with -100 where tokens should be ignored (padding, special tokens)
        label_tensor = torch.full(input_ids.shape, -100, dtype=torch.long)
        
        # Match labels with the original tokens, not including special tokens
        label_pos = [i for i, word in enumerate(self.tokenizer.tokenize(original_sentence))]
        for i, label_value in zip(label_pos, label):
            label_tensor[i] = label_value
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

# Initialize tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = TokenClassificationDataset(original_sentences, corrupted_sentences, labels, tokenizer)

# Create a data loader for batching
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize BERT for token classification with 2 classes (normal/corrupted)
num_labels = 2  # 0 or 1
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels)
model = BertForTokenClassification.from_pretrained('bert-base-uncased', config=config)

# Example of training loop
optimizer = AdamW(model.parameters(), lr=5e-5)

# Simple training loop (just one step for demonstration)
model.train()
epochs = 3
loss_values=[]
for epoch in range(epochs):
    total_loss=0
    for batch in dataloader:
        input_ids2 = batch['input_ids']
        attention_mask2 = batch['attention_mask']
        labels2 = batch['labels']
        labels3 = torch.stack(labels2, dim = 1)

        # Forward pass
        outputs = model(input_ids=input_ids2, attention_mask=attention_mask2, labels=labels3)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # Loss and backward pass
        loss = outputs.loss
        total_loss += loss
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()

    # Calculate average loss for the epoch
    average_loss = total_loss/(len(dataloader)*epoch+1)
    loss_values.append(average_loss)

    print("Epoch: {} -> loss: {}".format(epoch+1, average_loss))

print("Training complete.")

# Assuming loss_values is a list or numpy array of loss values
loss_values_cpu = [loss.cpu().detach().numpy() for loss in loss_values]

# Plot loss vs epoch curve
plt.plot(np.array(range(1, epochs+1)), np.array(loss_values_cpu))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()

# Example of inference using the trained model
model.eval()
def split_lines(text, n):
    words = text.split()
    lines = [' '.join(words[i:i+n]) for i in range(0, len(words), n)]
    return lines
# Test with new data
# open file
with open(r"/home2/hardik_2001cs27/utad/datasets/ag_od/test/business-outliers.txt",'r') as f:
    file22 = f.read()

# Split the paragraph into lines after every 100 words
file2 = split_lines(file22, 100)

for line in file2:
    encoding = tokenizer(line, truncation=True, padding='max_length', return_tensors='pt')

    # Predict if tokens are normal or corrupted
    outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # Output the tokenization and the prediction
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze())
    print("Tokens:", tokens)
    print("Predictions:", predictions)