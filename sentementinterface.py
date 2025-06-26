import os
import sys
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from transformers import BertTokenizer
import gdown

#making model
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
    def train(model, data_loader, optimizer, scheduler, device):
        for batch in data_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
    def evaluate(model, data_loader, device):
        predictions = []
        actual_labels = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)
    def predict_sentiment(text, model, tokenizer, device, max_length=128):
        encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
        return "positive" if preds.item() == 1 else "negative"
    
if __name__ == '__main__':
    
    #downloads pretrained model if it is not allready downloaded
    if(not(os.path.isfile(os.getcwd() +"\\bert_classifier1.pth"))):
        url = 'https://drive.google.com/file/d/1bN4eu00gT4RHsdPX8QZu81AVNBnyRp8u/view?usp=drive_link'
        destination = os.getcwd() +"\\bert_classifier1.pth"
        gdown.download(url, destination, quiet=False,fuzzy=True)
        print(f"model dowloaded to {destination}")
    bert_model_name = 'bert-base-uncased'
    num_classes = 2
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(bert_model_name, num_classes).to(device)
    model.load_state_dict(torch.load('bert_classifier1.pth', map_location=torch.device(device)))
    file = open(sys.argv[1]) 
    text = file.read()
    print(BERTClassifier.predict_sentiment(text, model, tokenizer, device, max_length=128))
