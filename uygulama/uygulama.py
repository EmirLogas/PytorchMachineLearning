import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader, Dataset
import glob
import cv2
import random

class MyDataset(Dataset):
    def __init__(self, train):
        # Veri kümesinin yolu
        data_path = "/dataset"

        # Veri kümesine ait klasördeki her bir JPG uzantılı resmin dosya yolu bir listeye kaydedildi.
        self.data_list = glob.glob(data_path + "/*/*.JPG")

        # Liste karıştırıldı.
        random.shuffle(self.data_list)

        # train True ise eğitim
        # train False ise doğrulama verim kümesi olacak şekilde ayarlandı.
        if train == True:
            self.data_list = self.data_list[:220]
        else:
            self.data_list = self.data_list[220:]

    def __getitem__(self, index):
        # görüntü dosyadan okunur.
        image = cv2.imread(self.data_list[index])

        # görüntü pytorch tensor veri tipine dönüştürülür ve 64x64 boyutuna indirgenir
        image = ToTensor()(image)
        image = Resize((64,64))(image)

        # dosya yolu bilgisinden etiket değeri elde edilir.
        label = self.data_list[index].split('/')[4]
        return image, int(label)
  
    def __len__(self):
        return len(self.data_list)

ds_train = MyDataset(train = True)
ds_val = MyDataset(train = False)

dl_train = DataLoader(ds_train,batch_size=32,shuffle=True)
dl_val = DataLoader(ds_val,batch_size=32,shuffle=True)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(16384, 1000)
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = F.relu(out)
        out = self.conv_layer2(out)
        out = F.relu(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = F.relu(out)
        out = self.conv_layer4(out)
        out = F.relu(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

model = MyModel()

# kayıp fonksiyonu
loss_fn = nn.CrossEntropyLoss()

# Optimize edici olarak stochastic gradient descent seçtik, parametre olarak 
# model parametrelerini ve öğrenme oranını alır
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def dogrulama():
    correct = 0
    with torch.no_grad():  # İleri yayılımda gradyanların hesaplanmaması için kullanılır. Gradyan hesabı eğitim için gerekli. gradyan hesabının yapılmaması için test aşamasında bu yapıyı kullanıyoruz.
        for data, label in dl_val:
            tahmin = model(data())   # görüntüler modelden geçirilir ve tahmin değerleri elde edilir.
            correct += (tahmin.argmax(1) == label()).type(torch.float).sum().item()  # kaç tahmin doğru olarak bulundu

    sonuc = correct / len(ds_val)  # doğru olarak bulunan tahminlerin toplam veri sayısına oranı
    return sonuc

epoch = 50

for idx in range(epoch):
    for data, label in dl_train:
        # her iterasyonda batch_size miktarınca görüntü modelden geçirilerek her 
        # görüntü için tahmin değerleri hesaplanıyor.
        tahmin = model(data)

        # elde edilen tahmin değerlerinin olması gereken değerlerden ne kadar uzak 
        # olduğuna dair kayıp hesaplanıyor.
        loss = loss_fn(tahmin, label)

        # Backpropagation (geri yayılım ve ağırlıkların güncellenmesi )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Her epoch'ta kullanıcı bilgilendiriliyor.
    print("Epoch :",idx, "Loss :", loss.item(), "val acc", dogrulama())