import torch
import torch.nn.functional as F
import numpy as np
import cv2
import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from Models.DeepLabV3Plus.modeling import deeplabv3plus_resnet101

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
NUM_CLASSES = 3
OUTPUT_STRIDE = 8
IMG_SIZE = (512, 512)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
print(f"Loading model on {DEVICE}...")
model = deeplabv3plus_resnet101(
    num_classes=NUM_CLASSES, 
    output_stride=OUTPUT_STRIDE, 
    pretrained_backbone=False
)

checkpoint = torch.load("best.pth", map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

class SegmentationGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, target_class=1):
        with torch.set_grad_enabled(True):
            output = self.model(x)
            target = output[:, target_class, :, :].sum()
            self.model.zero_grad()
            target.backward()
            
            gradients = self.gradients.cpu().data.numpy()[0]
            activations = self.activations.cpu().data.numpy()[0]
            
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            
            for i, w in enumerate(weights):
                cam += w * activations[i]
                
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
            cam = cam - np.min(cam)
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            return cam

try:
    grad_cam = SegmentationGradCAM(model, model.backbone.layer4)
except AttributeError:
    print("Error hooking GradCAM. Check model structure.")

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_ndarray = np.array(image)
    img_ndarray = img_ndarray.transpose((2, 0, 1)) / 255.0
    return torch.from_numpy(img_ndarray).float().unsqueeze(0)

def clean_mask(pred_mask_numpy):
    """
    Post-processing to remove noise (small specks) and smooth boundaries.
    """
    # Kernel size determines how much 'smoothing' happens. 
    # (5,5) is standard for removing small dots.
    kernel = np.ones((5, 5), np.uint8)
    
    # Morphological Open: Removes small noise (salt noise)
    cleaned = cv2.morphologyEx(pred_mask_numpy, cv2.MORPH_OPEN, kernel)
    
    # Morphological Close: Fills small holes inside the object
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def colorize_mask(pred_mask_numpy):
    """
    Converts the numpy mask index map (0,1,2) to an RGB image.
    """
    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    h, w = pred_mask_numpy.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cid in range(NUM_CLASSES):
        color_image[pred_mask_numpy == cid] = colors[cid]
        
    return color_image

def apply_heatmap(original_image, cam_map):
    img = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    
    heatmap = (cam_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return result

def apply_overlay(original_image, mask_rgb):
    """
    Overlays the colored mask onto the original image.
    """
    # 1. Convert Tensor to BGR Image
    img = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 2. Convert Mask RGB to BGR
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    
    # 3. Blend
    # Using 0.6 for image and 0.6 for mask makes it brighter and clearer
    result = cv2.addWeighted(img, 0.7, mask_bgr, 0.6, 0)
    return result

def image_to_base64(image_array):
    """Converts an OpenCV image (numpy) to Base64 string"""
    _, buffer = cv2.imencode('.png', image_array)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_tensor = transform_image(contents).to(DEVICE)
        
        # 1. Heatmap
        cam_map = grad_cam(input_tensor, target_class=1)
        heatmap_img_bgr = apply_heatmap(input_tensor, cam_map)
        
        # 2. Segmentation
        with torch.no_grad():
            output = model(input_tensor)
        
        # Get the raw prediction mask (indices 0, 1, 2) as Numpy array
        pred_mask_numpy = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # --- CLEAN UP STEP ---
        pred_mask_numpy = clean_mask(pred_mask_numpy)
        
        # Colorize and Overlay
        segmentation_img_rgb = colorize_mask(pred_mask_numpy)
        segmentation_overlay_bgr = apply_overlay(input_tensor, segmentation_img_rgb)

        heatmap_b64 = image_to_base64(heatmap_img_bgr)
        segmentation_b64 = image_to_base64(segmentation_overlay_bgr)

        response_data = {
            "segmentation": segmentation_b64,
            "heatmap": heatmap_b64
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)