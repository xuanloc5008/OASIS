import requests
import base64

# Config
url = "http://127.0.0.1:8000/predict"
image_path = "/Users/xuanloc/Desktop/Screenshot 2025-10-26 at 10.32.33.png" # REPLACE with a real image path

# Send Request
with open(image_path, "rb") as f:
    files = {"file": f}
    print("Sending request...")
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    
    # Helper to save base64 string
    def save_b64(b64_str, filename):
        # Remove the header "data:image/png;base64,"
        clean_str = b64_str.split(",")[1]
        img_data = base64.b64decode(clean_str)
        with open(filename, "wb") as f:
            f.write(img_data)
        print(f"Saved: {filename}")

    # Save both images
    save_b64(data["segmentation"], "result_segmentation.png")
    save_b64(data["heatmap"], "result_heatmap.png")
    
else:
    print("Error:", response.text)