import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import ToTensor, Resize

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = F.relu(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool4(out)

        out = out.view(out.size(0), -1)

        out = self.dropout(out)
        out = self.fc1(out)

        out = F.relu(out)

        out = self.dropout(out)
        out = self.fc2(out)

        return out

model = MyModel()

model.load_state_dict(torch.load('model_weights3.pth'))
model.eval()

# Resimleri okumak için gerekli olan transform
image = cv2.imread('asd.jpeg')
image = cv2.resize(image, (64, 64))
image = ToTensor()(image)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)

print(output.argmax().item())


# WEBCAM
"""vid = cv2.VideoCapture(0)

while(True):
    # her itarasyonda webcam'dan bir görüntü okutulur.
    ret, frame = vid.read()

    # görüntü modelin kabul edeceği şekle dönüştürülür
    im = ToTensor()(frame)
    im = Resize((64, 64))(im)
    im = im.unsqueeze(0)

    # görüntü model tarafından test edilir.
    with torch.no_grad():
        model_out = model(im)

    # Elde edilen sonuç terminale yazdırılır.
    print(model_out.argmax().item())

    # Sonuç değeri görüntü üzerine yazdırılır.
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    frame = cv2.putText(frame, str(model_out.argmax().item()),
                        org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
"""