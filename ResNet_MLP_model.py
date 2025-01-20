from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.cuda import amp
import time
import copy

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 512

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224


    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "efficient":
        """ efficientnet_b1
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.efficientnet_b1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def evaluate(model,model_1, loader):
  loss_function = nn.CrossEntropyLoss()


  predictions = []
  loss = []
  true_labels=[]
  for i,(numeric_input, image_input,label) in enumerate(loader):
    true_labels.extend(label.numpy())
    img_output=model(image_input)
    y_pred = model_1(image_output, numeric_input)
    _loss = loss_function(y_pred, label).detach().numpy().flatten()
    y_pred = np.argmax(y_pred.detach().numpy(),axis=1)
    predictions.extend(y_pred)
    loss.extend(_loss)
  predictions = np.array(predictions)
  f1=f1_score(true_labels,predictions)
  acc = np.mean(predictions==true_labels)
  avg_loss = np.mean(loss)
  return acc, avg_loss ,f1


# ResNet+MLP model
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
          nn.Linear(246080, 64*2),
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

#ResNet model settings
device = CONFIG['device']
# Send the model to GPU
model = model.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.

params_to_update = model.parameters()

if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

model_name = "resnet"

# Number of classes in the dataset
num_classes = 256

feature_extract = True

model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
scaler = amp.GradScaler()


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


# start training
model_1=MLPModel()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = model_config['lr'])

num_epochs = 10
for epoch in range(num_epochs):
  for i,(numeric_input, image_input,label) in enumerate(A287570_train_img):
    #Resnet on image data
    img_output=model(image_input)
    model_1.train()
    optimizer.zero_grad()

    #Image+Numeric
    output = model_1(image_output,numeric_input)
    loss = loss_function(output,label)
    loss.backward()
    optimizer.step()
    if i % 300 ==0:
      pred_y = torch.max(output,1)[1].data.squeeze()
      accuracy = sum(pred_y==label) / float(label.size(0))
      print("Train Loss: %.4f, Train Accuracy: %.4f" % (loss.item(), accuracy))
  test_acc, test_loss ,test_f1 = evaluate(model,model_1,A287570_val_img)
  print("Test Acc: %.4f Loss: %.4f Test f1_score: %.4f" %(test_acc, test_loss,test_f1))