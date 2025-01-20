import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
from sklearn.metrics import roc_auc_score,f1_score,recall_score
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader,TensorDataset


#生成包含統計數據和影像資料的資料集
class CustomDataset(Dataset):
    def __init__(self, numeric_data, image_data, labels):
        self.numeric_data = numeric_data
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        numeric_input = self.numeric_data[index]
        image_input = self.image_data[index]
        label = self.labels[index]
        return numeric_input, image_input, label

#產生測試資料集的準確率和F1-Score的公式
def evaluate(model, loader):
  loss_function = nn.CrossEntropyLoss()
  predictions = []
  loss = []
  true_labels=[]
  for i,(numeric_input, image_input,label) in enumerate(loader):
    true_labels.extend(label.numpy())
    y_pred = model(image_input, numeric_input)
    _loss = loss_function(y_pred, label).detach().numpy().flatten()
    y_pred = np.argmax(y_pred.detach().numpy(),axis=1)
    predictions.extend(y_pred)
    loss.extend(_loss)
  predictions = np.array(predictions)
  f1=f1_score(true_labels,predictions)
  acc = np.mean(predictions==true_labels)
  avg_loss = np.mean(loss)
  return acc, avg_loss ,f1

#讀取影像
A158200_img = ImageFolder('/content/drive/MyDrive/sun_moon_crop_img/A158200',
        transform = transforms.Compose([transforms.Resize(512),
         transforms.ToTensor(),
        transforms.Grayscale()
        #transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ]))

#讀取統計數據並轉成Tensor
A158200_stat=pd.read_csv('/content/drive/MyDrive/ocr_stat/A158200_stat_path.csv',index_col=0)
A158200_num=A158200_stat.drop(['label','path'],axis=1)
A158200_num=torch.Tensor(A158200_num.values)

#生成影像和label的tensor
img_tensor=[]
label_tensor=[]
for a,b in enumerate(A158200_img):
  image, label=b
  img_tensor.append(image)
  label_tensor.append(label)

img_tensor=torch.stack(img_tensor)
label_tensor=torch.tensor(label_tensor)


from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

#生成image+numeric+label的資料集
custom_dataset = CustomDataset(A158200_num, img_tensor, label_tensor)


#將資料集分成測試和訓練資料集
batch_size=32
train_data,val_data = random_split(custom_dataset,[0.8,0.2],generator=torch.Generator().manual_seed(42))
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")

train_img = DataLoader(train_data, batch_size)
val_img = DataLoader(val_data, batch_size)


#CNN+MLP Model
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()

        self.image_features = nn.Sequential(
           nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=3, stride=2),
           nn.Dropout(),
           nn.Conv2d(32, 64, kernel_size=3),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=3, stride=2),
           nn.Dropout()
        )
        self.numeric_features = nn.Sequential(
           nn.Linear(38, 64),
           nn.ReLU(),
           nn.Dropout(),

        )
        self.combined_features = nn.Sequential(
          nn.Linear(3*512*512+64, 64*2),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(128, 2)
        )


    def forward(self,x,y):
        x= self.image_features(x)
        x=x.view(x.size(0),-1)
        # Output
        y=self.numeric_features(y)
        z=torch.cat((x,y),1)
        z=self.combined_features(z)
        return z


model_config = {'model': 'TwoLayerNN',
                'lr': 0.001     , ##      TODO: Tuning the hyperparameters      ##,
                'momentum': 0.9  , ##      TODO: Tuning the hyperparameters      ##,
                'weight_decay': 0.001
                }

model=MLPModel()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = model_config['lr'],
            momentum = model_config['momentum'],
            weight_decay = model_config['weight_decay'])

# start training
num_epochs = 10
for epoch in range(num_epochs):
  for i,(numeric_input, image_input,label) in enumerate(train_img):

    output = model(image_input,numeric_input)
    loss = loss_function(output,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 300 ==0:
      pred_y = torch.max(output,1)[1].data.squeeze()
      accuracy = sum(pred_y==label) / float(label.size(0))
      print("Train Loss: %.4f, Train Accuracy: %.4f" % (loss.item(), accuracy))
  test_acc, test_loss ,test_f1 = evaluate(model,val_img)
  print("Test Acc: %.4f Loss: %.4f Test f1_score: %.4f" %(test_acc, test_loss,test_f1))