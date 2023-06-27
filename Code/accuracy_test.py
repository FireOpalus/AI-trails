import torch
import pandas as pd
import numpy as np

from albert_few_shot import max_length, albertBertClassifier, tokenizer, count_classes, device

modelname = "/home/mihaii/Document/AI-trails/Code/huggingface-softmax.pt"
# modelname = "huggingface-softmax.pt"
albertBertClassifier.load_state_dict(torch.load(modelname))

## 模型accuracy测试
print("Test start")
albertBertClassifier.eval()

File_test = pd.read_csv('./data/test.csv', header = 0, encoding = 'utf-8')
test_raw = File_test.iloc[np.where(File_test['Title'].notnull())]
accuracy = 0.0
num_right = 0
print("Testing...")
for index, row in test_raw.iterrows():
    text =row['Abstract'].strip()
    label = row['Classification']
    with torch.no_grad():
        ids = tokenizer.encode(text, max_length = max_length, padding = 'max_length', truncation = True)
        ids = torch.from_numpy(np.array([ids])).to(device)
        pred = albertBertClassifier(ids)
        pred_cls = pred.max(1)[1].cpu().detach().numpy()[0]
        if count_classes[pred_cls] == label:
            num_right += 1
accuracy = num_right / 7600
print("Test accomplished")
print("The test accuracy is: " + str(accuracy))