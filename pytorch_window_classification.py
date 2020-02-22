# PyTorch example——Word Window Classification

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

# 简易的训练测试数据：
sents = [s.lower().split() for s in ["we 'll always have Paris",
                                           "I live in Germany",
                                           "He comes from Denmark",
                                           "The capital of Denmark is Copenhagen",
                                           "She comes from Paris"]]
labels = [[0, 0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1]]



"""
大致的步骤：

数据预处理：
- 字典：词与index对应的字典；
- 字典：index对词对应的字典；
- 函数：将词语句子转化为index向量；
- 函数：为了进行window classification，句子的两边需要padding
- 函数：为了处理不同的句子的长度，需要把句子统一长度，也是padding（PyTorch中提供了相应的函数）
- 函数：把标签转化成one-hot的形式
- 综合函数：利用上面的工具，把原始数据转化成模型需要的数据

模型搭建与训练：
- 定义模型的类：
    初始化函数；
    forward函数；

- 损失计算的函数；
- 训练函数（数据加载，反向传播，计算损失）
- 迭代调用训练函数
"""

# ============数据预处理部分：==================
# 形成词汇库：（注意加上两个特殊字符<pad>和<unk>）
corpus = []
for sent in sents:
    corpus.extend(sent)
corpus = list(set(corpus))
corpus = ['<pad>','<unk>']+corpus

# 词与index相互转换的字典：
word2id = {w:i for i,w in enumerate(corpus)}
id2word = {i:w for i,w in enumerate(corpus)}

# 函数：给定一个句子，转化成index向量：
# 注意，这里可能会遇到未知词，要替换成<unk>对应的index
def sent2ids(sent):
    return [word2id.get(w,word2id['<unk>']) for w in sent]

# 函数：为window classification进行的padding：
# example:
# ['we', "'ll", 'always', 'have', 'paris'] -->
# ['<pad>', '<pad>', 'we', "'ll", 'always', 'have', 'paris', '<pad>', '<pad>']
def window_pad_for_sent(sent,window_step,pad_token='<pad>'):
    return [pad_token]*window_step+sent+[pad_token]*window_step

# 把标签转化成one-hot的形式：
def label2onehot(label):
    """
    这里的label是诸如[0,0,1]这样的
    需要转化成[[0,1],
              [0,1],
              [1,0]]这样.
    """
    one_hots = torch.zeros((len(label),2))
    true = torch.tensor(label) #[0,0,1]
    false = ~true.byte() #[1,1,0]
    # 这里是torch中一个功能，即对tensor的byte直接按位取反，很方便
    one_hots[:,0] = true
    one_hots[:,1] = false
    return one_hots

def data_tranform(sents,labels,window_step):
    """
    给定句子和标签，
    把句子转化为index矩阵（要经过padding补齐定长）；
    把标签转化为one-hot矩阵（也要padding补齐）；
    注意，index的长度会比one-hot更长，因为多了window padding
    """
    window_padded_sents_ids = [torch.tensor(sent2ids(window_pad_for_sent(sent,window_step))) for sent in sents]
    padded_inputs = nn.utils.rnn.pad_sequence(window_padded_sents_ids,batch_first=True)

    labels_ohs = [torch.tensor(label2onehot(label)) for label in labels]
    padded_ys = nn.utils.rnn.pad_sequence(labels_ohs,batch_first=True)
    return TensorDataset(padded_inputs,padded_ys)

# padded_inputs,padded_ys = data_tranform(sents,labels,2)
# print(padded_inputs.size())
# print(padded_ys.size())


# ============模型搭建部分：==================

class SoftmaxWordWindowClassifier(nn.Module):
    def __init__(self,params,vocab_size,pad_idx=0):
        super(SoftmaxWordWindowClassifier,self).__init__()
        # 定义变量：
        self.window_size = 2*params['window_step']+1
        self.embed_dim = params["embed_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.num_classes = 2
        self.freeze_embeddings = params["freeze_embeddings"]

        # 定义网络层：
        # Embedding layer:
        self.embed_layer = nn.Embedding(vocab_size,self.embed_dim,padding_idx=0)
        # 是否冻结词向量
        if self.freeze_embeddings:
            self.embed_layer.weight.requires_grad = False

        # Hidden layer（就是一层神经网络，使用tanh激活函数）:
        self.hidden_layer = nn.Sequential(nn.Linear(self.window_size*self.embed_dim,
                                                    self.hidden_dim),
                                          nn.Tanh())

        # Output layer:
        self.output_layer = nn.Linear(self.hidden_dim,self.num_classes)
        # 最后输出softmax值的对数，使用这个函数可以避免一些潜在的数学问题
        # 使用LogSoftmax之后，后面计算loss就可以直接取其负数
        self.log_softmax = nn.LogSoftmax(dim=2)

    # 定义前向传播的方法：
    def forward(self,inputs):
        """
        batch_size : B
        window_num : N
        window_size : S
        embed_dim : D
        """
        batch_size,sent_len = inputs.size()
        # 获取所有的滑动窗口，所以每一句话会获得若干个窗口
        # 则inputs会变形为(batch_size,window_num,window_size)
        # 可以使用torch tensor的unfold函数轻松实现这个功能：
        windows = inputs.unfold(dimension=1, size=self.window_size, step=1)
        _,window_num,_ = windows.size()

        # shape sanity check:
        assert windows.size() == (batch_size,window_num,self.window_size)

        embedded_windows = self.embed_layer(windows)
        # print('embedded_windows:',embedded_windows.size())
        # (B,N,S)->(B,N,S,D)
        # 注意，本来embedding layer的输入应该是一串indices，然后通过这些indices可以直接look up词向量
        # 但是这里我们的输入多了一个维度，相当于多串indices。但是应该照样可以handle，就当做矩阵运算了，所以输出也是在对应位置多了一个维度

        # 接着把一个window中的各个词向量给拼起来，实际上就是reshape一下。即(B,N,S,D)->(B,N,S*D)
        embedded_windows_concat = embedded_windows.view(batch_size,window_num,-1)
        # print('embedded_windows_concat:',embedded_windows_concat.size())
        hidden = self.hidden_layer(embedded_windows_concat) # (B,N,S*D)->(B,N,H)
        # print('hidden:',hidden.size())
        output = self.output_layer(hidden)   # (B,N,H)->(B,N,2)
        # print('output:',output.size())
        output = self.log_softmax(output)    # (B,N,2)->(B,N,2)

        # 返回一个batch的输出
        return output

# 计算损失的函数：
def loss_function(outputs,labels):
    """
    这里的outputs是一个batch的输出；
    labels则是padding好的onehot标签。
    实际上，我们可以发现，outputs和labels的形状此时完全一样：
    outputs:(B,N,2) labels:(B,N,2)
    因此可以直接element-wise相乘，取负数，得到loss矩阵.
    """
    B, N, num_classes = outputs.size()
    loss_matrix = outputs*labels
    loss = -loss_matrix.sum().float()/(B*N) # ??
    return loss


# 训练函数（一个epoch，每个epoch会有多个batch）：
def train_epoch(train_data,model,loss_function,optimizer):
    epoch_loss = 0
    for inputs , ys in train_data: # padded_inputs,padded_ys
        optimizer.zero_grad() # 每个batch都要把梯度清零
        outputs = model.forward(inputs)
        loss = loss_function(outputs, ys) #计算一个batch的loss
        # pass gradients back, startiing on loss value
        loss.backward()
        # update parameters
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss
     


# ============训练测试部分：==================

params = {"batch_size": 4,
          "window_step": 2,
          "embed_dim": 25,
          "hidden_dim": 25,
          "num_classes": 2,
          "freeze_embeddings": False,
         }

learning_rate = .0002
num_epochs = 10000
model = SoftmaxWordWindowClassifier(params, len(corpus))
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

dataset = data_tranform(sents,labels,params['window_step'])
print(type(dataset))
print('------------')
train_loader = DataLoader(dataset,batch_size=params['batch_size'],shuffle=True)

losses = []
for epoch in range(num_epochs):
    epoch_loss = train_epoch(train_loader,model,loss_function,optimizer)
    if epoch % 100 == 0:
        losses.append(epoch_loss)
        print('epoch%d loss:'%epoch,epoch_loss)
