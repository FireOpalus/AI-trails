import torch
from transformers import AlbertTokenizer, AlbertModel
from transformers import BertTokenizer, AlbertModel, AlbertConfig
from torch.utils import data
from sklearn.model_selection import train_test_split

#设置模型训练参数
shot = 80           #X-shot
max_length = 128    #文本最大长度
epoch = 500         #训练最大轮数
batch_size = 64     #每次抽取的测试集样本数
learn_rate = 1.5e-5 #模型学习率

#加载预训练模型
pretrained = './albert-base-v2/'
tokenizer = AlbertTokenizer.from_pretrained(pretrained)
model = AlbertModel.from_pretrained(pretrained)
config = AlbertConfig.from_pretrained(pretrained)

#加载训练数据
import pandas as pd

Users_vc = pd.read_csv('./data/train_1000.csv', header = 0, encoding = 'utf-8')
count_classes = pd.value_counts(Users_vc['Classification'], sort = True).index.tolist()
print("Classes: " + str(count_classes))

#构建模型网络结构
class AlbertClassifier(torch.nn.Module):
    def __init__(self, bert_model, bert_config, num_class):
        super().__init__()
        self.bert_model = bert_model
        self.bert_config = bert_config
        self.fc2 = torch.nn.Linear(bert_config.hidden_size, num_class)
        self.softmax = torch.nn.Softmax(dim = 1)
    
    def forward(self, token_ids):
        bert_out = self.bert_model(token_ids)[1]
        bert_out = self.fc2(bert_out)
        return self.softmax(bert_out)

albertBertClassifier = AlbertClassifier(model, config, len(count_classes))
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
albertBertClassifier = albertBertClassifier.to(device)

#准备训练数据和验证数据
import numpy as np
raw = Users_vc.iloc[np.where(Users_vc['Title'].notnull())]

mdata, label = [], []
for index, row in raw.iterrows():
    ids = tokenizer.encode(row['Abstract'].strip(), max_length = max_length, padding = 'max_length', truncation = True)
    mdata.append(ids)
    label.append(row['Classification'] - 1)

num_data = len(label)

X_train, X_test, y_train, y_test = train_test_split(mdata, label, test_size = num_data - 4 * shot, shuffle = True, stratify = label)

class DataGen(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return np.array(self.data[index]), np.array(self.label[index])
    
#制作相关数据
train_dataset = DataGen(X_train, y_train)
test_dataset = DataGen(X_test, y_test)
train_dataloader = data.DataLoader(train_dataset, batch_size = batch_size)
test_dataloader = data.DataLoader(test_dataset, batch_size = batch_size)

#定义优化器和损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params = albertBertClassifier.parameters(), lr = learn_rate)

print("preparation accomplished")

if __name__ == "__main__":
    #模型训练与测试
    from tqdm import tqdm

    test_accu_max = 0
    count_smaller = 0


    for epoch in tqdm(range(epoch)):
        loss_sum = 0.0
        accu = 0
        albertBertClassifier.train()
        for step, (token_ids, label) in enumerate(train_dataloader):
            token_ids = token_ids.to(device)
            label = label.to(device)
            out = albertBertClassifier.forward(token_ids)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().data.numpy()
            accu += (out.argmax(1) == label).sum().cpu().data.numpy()
            test_loss_sum=0.0
        test_accu=0
        albertBertClassifier.eval()
        for step,(token_ids,label) in enumerate(test_dataloader):
            token_ids = token_ids.to(device)
            label = label.to(device)
            with torch.no_grad():
                out = albertBertClassifier(token_ids)
                loss = criterion(out,label)
                test_loss_sum += loss.cpu().data.numpy()
                test_accu += (out.argmax(1) == label).sum().cpu().data.numpy()
        print("epoch % d,train loss:%f,train acc:%f,test loss:%f,test acc:%f, max acc:%f"%\
            (epoch,loss_sum/len(train_dataset), accu/len(train_dataset),
            test_loss_sum/len(test_dataset), test_accu/len(test_dataset), test_accu_max))
    
        if test_accu/len(test_dataset) > test_accu_max:
            count_smaller = 0
            test_accu_max = test_accu/len(test_dataset)
            torch.save(albertBertClassifier.state_dict(), 'huggingface-softmax.pt') #保存测试准确度最高的模型
        else:
            count_smaller += 1    #如果连续100次未对最大值进行刷新则判断过拟合
            if count_smaller >= 50:    #手动选择是否继续
                ans = input("Continue?(Y/n)")
                if ans == "n":
                    break
                else:
                    count_smaller = 0
                    print("Continue...")
        
## 载入最优模型，修改为训练得到的最好模型
    albertBertClassifier.load_state_dict(torch.load('huggingface-softmax.pt'))

## 模型accuracy测试
    print("Testing...")
    albertBertClassifier.eval()

    File_test = pd.read_csv('./data/test.csv', header = 0, encoding = 'utf-8')
    test_raw = File_test.iloc[np.where(File_test['Title'].notnull())]
    accuracy = 0.0
    num_right = 0
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