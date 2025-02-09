from torch.utils.data import Dataset
from PIL import Image
import os,pdb
import pandas as pd

class ButterflyDataset(Dataset):
    def __init__(self, data, root_dir, phase='train', transforms=None,class_nums=2):
        self.data = data
        self.root_dir = root_dir
        self.transforms = transforms

        # Validate the 'hybrid_stat' column to ensure it contains only expected values
        valid_classes = {"hybrid", "non-hybrid"}
        self.data["hybrid_stat"] = self.data["hybrid_stat"].str.strip().str.lower()  # Normalize the values
        if not set(self.data["hybrid_stat"].unique()).issubset(valid_classes):
            raise ValueError("Unexpected values found in 'hybrid_stat' column.")

        # Define classes explicitly to avoid relying on sorted order
        self.phase = phase
        if self.phase=='train' and class_nums==196:
            self.classes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13']
            for ii in range(14):
                for jj in range(14):
                    if ii==jj:
                        continue
                    self.classes.append(str(ii)+'_'+str(jj))
        else:
            self.classes = ["non-hybrid", "hybrid"]  

        self.cls_lbl_map = {cls: i for i, cls in enumerate(self.classes)}

        # Generate labels using a vectorized approach for efficiency
        
        if self.phase=='train' and class_nums==196:
            self.labels = self.data["subspecies"].map(self.cls_lbl_map).tolist()
        else:
            self.labels = self.data["hybrid_stat"].map(self.cls_lbl_map).tolist()
        # pdb.set_trace()
        print("Created base dataset with {} samples".format(len(self.data)))
        
    def get_file_path(self, x):
        filepath = os.path.join(self.root_dir, x['hybrid_stat'], x['filename'])            
        return filepath
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data.iloc[index]
        img_path = self.get_file_path(x)
        try:
            img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
        except Exception as e:
            raise FileNotFoundError(f"Error loading image at {img_path}: {e}")
        
        lbl = self.labels[index]
       
        # print(self.labels)
        # if not pd.isna(lbl):
        #     print(img_path,' label is nan, idx=', index)

        if self.transforms:
            img = self.transforms(img)
            
        return img, lbl, img_path

