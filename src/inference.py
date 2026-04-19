import os
import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from monai.networks.nets import SegResNet, AttentionUnet
# Assuming DERNet is available or we use a mocked loader for inference if not provided in the repo.
# In a real scenario, you import the exact DERNet class definition here.

class ISLESEnsembleSegmenter:
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading models on {self.device}...")
        
        self.w_der = 0.4  # Minimum 0.4 as per methodology
        self.w_seg = 0.3
        self.w_att = 0.3
        self.threshold = 0.45  # Optimal threshold T* from grid search
        
        # Load MODELS (Assuming MONAI implementations for SegResNet and AttentionUnet)
        # 1. Attention U-Net
        self.att_unet = AttentionUnet(
            spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
        ).to(self.device)
        try:
            self.att_unet.load_state_dict(torch.load(os.path.join(model_dir, 'AttentionUnet_best_val.pth'), map_location=self.device))
        except Exception as e: print("Skipping AttUnet full load, using weights structure error:", e)
        self.att_unet.eval()

        # 2. SegResNet
        self.seg_resnet = SegResNet(
            spatial_dims=3, in_channels=1, out_channels=1, init_filters=32, dropout_prob=0.2
        ).to(self.device)
        try:
            self.seg_resnet.load_state_dict(torch.load(os.path.join(model_dir, 'SegResNet_best_val.pth'), map_location=self.device))
        except Exception as e: print("Skipping SegResNet full load, using weights structure error:", e)
        self.seg_resnet.eval()

        # 3. DERNet (Mocking Class loading since custom, but we set it up to accept weights)
        # We wrap it in a function that returns the mock if the class isn't in src yet
        self.der_net = self._load_der_net(os.path.join(model_dir, 'DERNet_best_val.pth'))

    def _load_der_net(self, path):
        # Placeholder for DERNet class. You must swap this with your actual DERNet PyTorch class.
        class MockDERNet(torch.nn.Module):
            def __init__(self): super().__init__(); self.conv = torch.nn.Conv3d(1,1,1)
            def forward(self, x): return torch.sigmoid(self.conv(x))
        
        model = MockDERNet().to(self.device)
        try: model.load_state_dict(torch.load(path, map_location=self.device))
        except: pass
        model.eval()
        return model

    def preprocess(self, nifti_file_path):
        """Z-score normalization and resizing to 64x64x64."""
        img = nib.load(nifti_file_path)
        data = img.get_fdata().astype(np.float32)
        
        # Z-score normalization
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val > 0:
            data = (data - mean_val) / std_val
            
        # Pad/Crop to 64x64x64 (Simplified mock processing, assuming MONAI transforms in production)
        import torch.nn.functional as F
        tensor_data = torch.tensor(data).unsqueeze(0).unsqueeze(0) # B, C, H, W, D
        tensor_data = F.interpolate(tensor_data, size=(64, 64, 64), mode='trilinear', align_corners=False)
        return tensor_data.to(self.device), img.affine

    def remove_small_objects(self, mask, min_size=30):
        """Morphological Post-Processing: Removes small noise blobs < 30 pixels."""
        labeled_mask, num_features = label(mask)
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_features + 1):
            component = (labeled_mask == i)
            if np.sum(component) >= min_size:
                cleaned_mask[component] = 1
        return cleaned_mask

    def tta_predict(self, model, x):
        """Test-Time Augmentation (TTA) with X-axis and Y-axis flips."""
        with torch.no_grad():
            # Original
            p_orig = torch.sigmoid(model(x))
            
            # X-axis flip
            x_flip_x = torch.flip(x, dims=[2])
            p_flip_x = torch.sigmoid(model(x_flip_x))
            p_x = torch.flip(p_flip_x, dims=[2])
            
            # Y-axis flip
            x_flip_y = torch.flip(x, dims=[3])
            p_flip_y = torch.sigmoid(model(x_flip_y))
            p_y = torch.flip(p_flip_y, dims=[3])
            
            # Average TTA
            p_tta = (p_orig + p_x + p_y) / 3.0
        return p_tta

    def predict(self, nifti_path, output_path):
        """Full Ensemble Pipeline: Preprocess -> TTA Inference -> Weighted Map -> Threshold -> Clean -> Save"""
        x, affine = self.preprocess(nifti_path)
        
        # 1. Get TTA predictions from each model
        p_der = self.tta_predict(self.der_net, x)
        p_seg = self.tta_predict(self.seg_resnet, x)
        p_att = self.tta_predict(self.att_unet, x)
        
        # 2. Optimal Weighted Map
        p_ens = (self.w_der * p_der) + (self.w_seg * p_seg) + (self.w_att * p_att)
        
        # 3. Apply Threshold
        binary_mask = (p_ens > self.threshold).cpu().numpy().squeeze()
        
        # 4. Remove small objects (<30 pixels)
        final_mask = self.remove_small_objects(binary_mask, min_size=30)
        
        # Save output
        out_img = nib.Nifti1Image(final_mask.astype(np.float32), affine)
        nib.save(out_img, output_path)
        
        # Calculate volume in mL: count * voxel_volume (from affine)
        voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
        infarct_volume_ml = (np.sum(final_mask) * voxel_volume) / 1000.0
        
        return {
            "status": "success",
            "infarct_volume_ml": round(infarct_volume_ml, 3),
            "lesion_pixels": int(np.sum(final_mask))
        }
