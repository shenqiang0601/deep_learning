import torch
from datasets import load_dataset
import torch.nn.functional as F
from transformers import BertTokenizer

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path='data', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label


dataset = Dataset('train')
print(len(dataset), dataset[0])


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=10,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break

print(len(loader))
print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

from transformers import BertModel

# 加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese')

# 不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

# 模型试算
out = pretrained(input_ids=input_ids,
                 attention_mask=attention_mask,
                 token_type_ids=token_type_ids)

print(out.last_hidden_state.shape)


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)
        # 可加入CNN卷积层，可以自行操作
        # self.conv1D = torch.nn.Conv1d(in_channels=500, out_channels=500, kernel_size=1)
        # self.MaxPool1D = torch.nn.MaxPool1d(4, stride=2)
        # self.Dropout = torch.nn.Dropout(p=0.5, inplace=False)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        print(out.shape)
        return out


model = Model()
print(model)
# model.summary()
model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape

from transformers import AdamW

# 训练
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
epochs = 30

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 1 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)

        print('epochs:', i, 'loss:', loss.item(), 'accuracy:', accuracy)

    if i == epochs:
        torch.save(model, 'text_classfiy.model')
        # model_load = torch.load('model/命名实体识别_中文.model')
        break


# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=10,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):

        if i == 5:
            break

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(correct / total)