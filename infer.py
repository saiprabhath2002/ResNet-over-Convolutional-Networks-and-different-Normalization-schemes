import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import precision_score, f1_score, accuracy_score
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from matplotlib.cm import inferno

accuracy_train=[]
microf1_train=[]
macrof1_train=[]
error_train=[]
loss_train=[]
precision_score_train=[]
accuracy_validation=[]
microf1_validation=[]
macrof1_validation=[]
error_validation=[]
loss_validation=[]
precision_score_validation=[]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])


def inverse_normalize(img_tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )

    return inv_normalize(img_tensor)




def gradcam2( sample_x, sample_y, target_layer,model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    global gradients, activations
    gradients, activations = None, None
    
    def backward_hook(module, grad_input, grad_output):
      global gradients 
      gradients = grad_output
    def forward_hook(module, args, output):
      global activations
      activations = output
        
    forward_handle = target_layer.register_full_backward_hook(backward_hook)  #prepend=False
    backward_handle = target_layer.register_forward_hook(forward_hook)
    
    
    output = model(sample_x.unsqueeze(0)) 
    sample_y = torch.tensor([sample_y]).to(device)
    loss = criterion(output,sample_y) 
    loss.backward() 
    
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    forward_handle.remove()
    backward_handle.remove()
    
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)

    heatmap /= torch.max(heatmap)
    heatmap=heatmap.to('cpu')
    heatmap=heatmap.detach().numpy()
    original_image_tensor = inverse_normalize(sample_x)
    original_image = transforms.ToPILImage()(original_image_tensor)
    original_image=np.array(original_image)
    heatmap_resized = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_LINEAR)
    heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))
    heatmap_colored = inferno(heatmap_normalized)[:, :, :3] 
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    combined_img = cv2.addWeighted(original_image, 0.5, heatmap_colored, 0.5, 0)

    return combined_img

def collect_req_images_from_validation_data(loader,model):
    mapping={2:0,7:1,13:2,18:3,19:4,22:5,23:6}
    correctly_classified=[[] for _ in range(7)]
    incorrectly_classified=[[] for _ in range(7)]
    for i,(x,y) in enumerate(loader) :
#         print(i)
        x=x.to(device)
        for j,img in enumerate(x) :
            y_orig=y[j].item()
            y_pred=torch.argmax(model(img.unsqueeze(0))).item()
#                 print(y_orig,y_pred)
            if y_orig in mapping:
                if (y_orig==y_pred) and len(correctly_classified[mapping[y_orig]])<5:
                    correctly_classified[mapping[y_orig]].append(gradcam2(img,y[j],model.layers[-1],model))
                elif (y_orig!=y_pred) and len(incorrectly_classified[mapping[y_orig]])<5:
                    incorrectly_classified[mapping[y_orig]].append(gradcam2(img,y[j],model.layers[-1],model))
    return correctly_classified,incorrectly_classified


if(False):
    correctly_classified,incorrectly_classified=collect_req_images_from_validation_data(test_loader,m)
    for cls,i in enumerate(correctly_classified):
        name=str(cls)+"_1_"
        for img_c,j in enumerate(i):
            p=name+str(img_c)+".png"
            cv2.imwrite(p,j)
    for cls,i in enumerate(incorrectly_classified):
        name=str(cls)+"_0_"
        for img_c,j in enumerate(i):
            p=name+str(img_c)+".png"
            cv2.imwrite(p,j)


class batch_nomalization(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(batch_nomalization, self).__init__()
        self.momentum = momentum
        self.norm_across = (0, 2, 3) 

        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=self.norm_across, keepdim=True)
            var = x.var(dim=self.norm_across, keepdim=True, unbiased=False)
            n = x.numel() / x.size(1)   # Update running estimates
            self.running_mean =  (1 - self.momentum) * self.running_mean +self.momentum * mean 
            self.running_var =   (1 - self.momentum) * self.running_var + self.momentum * (n / (n - 1)) * var
        else:
            mean = self.running_mean
            var = self.running_var

        return self.gamma * ((x - mean) / torch.sqrt(var + 1e-5)) + self.beta


class Instance_Normalization(nn.Module):
    def __init__(self,num_features):
        super(Instance_Normalization, self).__init__()
        self.shape=(1,num_features,1,1)
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self,x):
        var = x.var(dim=(2,3), keepdim=True)
        mean = x.mean(dim=(2,3), keepdim=True)
        return (self.gamma * ((x-mean)/torch.sqrt(var + 1e-5)) + self.beta)


class layer_Normalization(nn.Module):
    def __init__(self,num_features):
        super(layer_Normalization, self).__init__()
        self.shape=(1,num_features,1,1)
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    
    def forward(self,x):
        var = x.var(dim=(1,2,3), keepdim=True,unbiased=False)
        mean = x.mean(dim=(1,2,3), keepdim=True)
        return (self.gamma * ((x-mean)/torch.sqrt(var + 1e-5)) + self.beta)
    
    
class BatchInstanceNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(BatchInstanceNorm, self).__init__()
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        if self.training:
            var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
            self.running_mean = (1 - self.momentum) * self.running_mean +  self.momentum * mean_bn 
            self.running_var =   (1 - self.momentum) * self.running_var + self.momentum * var_bn
        else:
            mean_bn = self.running_mean
            var_bn = self.running_var

        x_bn = (x - mean_bn) / torch.sqrt(var_bn + 1e-5)
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_in = (x - mean_in) / torch.sqrt(var_in + 1e-5)
        x = 0.5 * x_bn + (1 - 0.5) * x_in
        return ( x * self.gamma + self.beta)

    
    
class GroupNorm(nn.Module):
    def __init__(self, num_features, group=4):
        super(GroupNorm, self).__init__()
        self.group = group
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.view(N, self.group, C // self.group, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + 1e-5)
        x = x_normalized.view(N, C, H, W)
        return (x * self.gamma + self.beta)
    
    
class no_norm(nn.Module):
    def __init__(self,num_features):
        super(no_norm,self).__init__()
    def forward(self,x):
        return x

def layer_normalization(dim, norm_type):
    if norm_type == "inbuilt":
        return nn.BatchNorm2d(dim)
    elif norm_type=="bn":
        return batch_nomalization(dim)
    elif norm_type=="in":
        return Instance_Normalization(dim)
    elif norm_type=="ln":
        return layer_Normalization(dim)
    elif norm_type=="nn":
        return no_norm(dim)
    elif norm_type=="bin":
        return BatchInstanceNorm(dim)
    elif norm_type=="gn":
        return GroupNorm(dim)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, norm_type="torch_bn"):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = layer_normalization(out_channels, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2, bias=False)
        self.bn2 = layer_normalization(out_channels, norm_type)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, n_channels=[16, 32, 64], n_layers=[2,2,2], n_classes=25, norm_type="torch_bn", input_size=256):
        super(ResNet, self).__init__()
        self.in_channels = n_channels[0]
        self.conv = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = layer_normalization(self.in_channels, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList()
        for index, num_blocks in enumerate(n_layers):
            out_channels = n_channels[index]
            strides = [2] + [1] * (num_blocks - 1)
            for stride in strides:
                downsample = None
                if stride != 1 or self.in_channels != out_channels:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        layer_normalization(out_channels, norm_type),
                    )
                self.layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride, downsample=downsample, norm_type=norm_type))
                self.in_channels = out_channels
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, n_classes)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def store_results_train(actual,predicted,epoch,l):
    accuracy = accuracy_score(actual, predicted)
    micro_f1 = f1_score(actual, predicted, average='micro')
    macro_f1 = f1_score(actual, predicted, average='macro')
    precision = precision_score(actual, predicted, average='macro')
    error = 1 - accuracy
    accuracy_train.append(accuracy)
    microf1_train.append(micro_f1)
    macrof1_train.append(macro_f1)
    error_train.append(error)
    precision_score_train.append(precision)
    loss_train.append(l)
    print("====================train_data======================")
    print(f"for epoch {epoch} accuracy:{accuracy},loss:{l}")
    print("====================================================")
    
def store_results_validation(actual,predicted,epoch,l):
    accuracy = accuracy_score(actual, predicted)
    micro_f1 = f1_score(actual, predicted, average='micro')
    macro_f1 = f1_score(actual, predicted, average='macro')
    precision = precision_score(actual, predicted, average='macro')
    error = 1 - accuracy
    accuracy_validation.append(accuracy)
    microf1_validation.append(micro_f1)
    macrof1_validation.append(macro_f1)
    error_validation.append(error)
    precision_score_validation.append(precision)
    loss_validation.append(l)
    print("====================validation_data======================")
    print(f"for epoch {epoch} accuracy:{accuracy},loss:{l}")
    print("====================================================")
    

def store_train_results_to_csv(accuracy_train, microf1_train, macrof1_train, precision_score_train, error_train, loss_train, filename="training_results.csv"):
    data = {
        "Accuracy": accuracy_train,
        "Micro F1 Score": microf1_train,
        "Macro F1 Score": macrof1_train,
        "Precision": precision_score_train,
        "Error": error_train,
        "Loss": loss_train
    }
    
    df = pd.DataFrame(data)
    
    df.to_csv(filename, index=False)


def store_validation_results_to_csv(accuracy_validation, microf1_validation, macrof1_validation, precision_score_validation, error_validation,loss_validation,  filename="validation_results.csv"):
    data = {
        "Accuracy": accuracy_validation,
        "Micro F1 Score": microf1_validation,
        "Macro F1 Score": macrof1_validation,
        "Precision": precision_score_validation,
        "Error": error_validation,
        "Loss":loss_validation
    }
    
    df = pd.DataFrame(data)
    
    df.to_csv(filename, index=False)

def store(name):
    store_train_results_to_csv(accuracy_train, microf1_train, macrof1_train, precision_score_train, error_train, loss_train, name+"_training_results.csv")
    store_validation_results_to_csv(accuracy_validation, microf1_validation, macrof1_validation, precision_score_validation, error_validation,loss_validation, name+"_validation_results.csv")
    
def plot_lists(*lists, labels=None, title='Plot of Lists',name=None, xlabel='Epoch', ylabel='Value', legend=True):
    plt.figure(figsize=(10, 6)) 
    for i, lst in enumerate(lists):
        plt.plot(lst, label=labels[i] if labels else f'List {i + 1}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if legend:
        plt.legend()
    
    plt.grid(True)
    plt.savefig(name)
    plt.show()
    
def saveall(name):
#     torch.save(m.state_dict(), name+'.pth')
    store(name)
    plot_lists(accuracy_train, microf1_train, macrof1_train, precision_score_train, error_train, loss_train,
           labels=['Accuracy', 'Micro F1', 'Macro F1', 'Precision', 'Error', 'Loss'],name=name+"_train.png")
    plot_lists(accuracy_validation, microf1_validation, macrof1_validation, precision_score_validation, error_validation,loss_validation,
           labels=['Accuracy', 'Micro F1', 'Macro F1', 'Precision', 'Error', 'Loss'],name=name+"_validation.png")
    
def clear_data():
    accuracy_train.clear()
    microf1_train.clear()
    macrof1_train.clear()
    error_train.clear()
    loss_train.clear()
    precision_score_train.clear()
    accuracy_validation.clear()
    microf1_validation.clear()
    macrof1_validation.clear()
    error_validation.clear()
    loss_validation.clear()
    precision_score_validation.clear()


def write_list_to_txt(lst, file_path):
    with open(file_path, 'w') as f:
        for item in lst:
            f.write("%s\n" % item)

            
class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = os.listdir(directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
            
def train(norm,pth,num_skip,data_pth,op_file_path):

    test_dataset = CustomImageDataset(directory=data_pth , transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    loaded_model = torch.load(pth)
    m=ResNet(norm_type=norm,n_layers=[num_skip]*3)
    m.load_state_dict(loaded_model)
    m=m.to(device)
    m.eval()

    predicted=[]
    for _,x in enumerate(tqdm(test_loader)):
        x=x.to(device)
        y_pred=m(x)
        predicted.extend(torch.argmax(y_pred,dim=1).squeeze().tolist())
    write_list_to_txt(predicted,op_file_path)




def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_file', type=str, help='Path to the model file')
    parser.add_argument('--normalization', type=str, help='Normalization method')
    parser.add_argument('--n', type=int, help='Number parameter')
    parser.add_argument('--test_data_file', type=str, help='Path to the test data file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')

    args = parser.parse_args()

    model_file = args.model_file
    normalization = args.normalization
    n = args.n
    test_data_file = args.test_data_file
    output_file = args.output_file

    train(norm=normalization,pth=model_file,num_skip=n,data_pth=test_data_file,op_file_path=output_file)

    

if __name__ == "__main__":
    main()
