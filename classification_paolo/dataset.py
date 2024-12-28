import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, type="gender", test=False):
        self.image_dir = image_dir
        self.label_file = label_file
        self.image_labels = []
        self.type=type

        if transform is None:
            if test:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                filename = parts[0]

                if type == "gender":
                    attribute = int(parts[3])
                elif type == "bag":
                    attribute = int(parts[4])
                elif type == "hat":
                    attribute = int(parts[5])

                if attribute in [0, 1]:
                    image_path = os.path.join(self.image_dir, filename)
                    if os.path.exists(image_path):
                        self.image_labels.append((filename, attribute))
                    else:
                        print(f"File non trovato, ignorato: {image_path}")


    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        filename, label = self.image_labels[idx]
        image_path = os.path.join(self.image_dir, filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"File non trovato durante il caricamento: {image_path}")
            return None
        if self.transform:
            image = self.transform(image)

        return image, label
