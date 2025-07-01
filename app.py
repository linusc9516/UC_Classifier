from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
from torchvision import transforms
import base64
import io

app = Flask(__name__)

# Load the model
model = torch.jit.load('./2classifier_rn18_layer3_4_unfreeze.pt')
# model = torch.jit.load('4classifier.pt')
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define image transform pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names_2c = ['Normal', 'Ulcerative Colitis']
# class_names_4c = ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']

def image_to_base64(image):
    """Convert PIL image to base64 string for HTML display"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def analyze_image(image_file):
    """Analyze a single image and return prediction"""
    image = Image.open(image_file.stream).convert('RGB')
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    
    prediction = class_names_2c[predicted_class.item()]
    
    # Reset stream position for base64 conversion
    image_file.stream.seek(0)
    image_for_display = Image.open(image_file.stream).convert('RGB')
    base64_image = image_to_base64(image_for_display)
    
    return {
        'filename': image_file.filename,
        'prediction': prediction,
        'base64': base64_image
    }

def generate_conclusion(results):
    """Generate medical conclusion based on multiple image analyses"""
    total_images = len(results)
    uc_count = sum(1 for r in results if r['prediction'] == 'Ulcerative Colitis')
    normal_count = total_images - uc_count
    uc_percentage = (uc_count / total_images) * 100 if total_images > 0 else 0
    
    # Determine risk level and conclusion
    if uc_percentage >= 70:
        risk_level = 'high'
        conclusion = 'HIGH RISK: Strong evidence of Ulcerative Colitis detected'
        summary = f'Analysis of {total_images} colonoscopy images shows {uc_count} images with UC indicators ({uc_percentage:.1f}%). This suggests a high probability of active ulcerative colitis.'
        recommendation = 'Immediate consultation with a gastroenterologist is strongly recommended. Further diagnostic tests including complete colonoscopy, biopsy, and laboratory tests should be considered.'
    elif uc_percentage >= 40:
        risk_level = 'medium'
        conclusion = 'MODERATE RISK: Possible signs of Ulcerative Colitis detected'
        summary = f'Analysis of {total_images} images shows {uc_count} images with potential UC indicators ({uc_percentage:.1f}%). Results are inconclusive but warrant attention.'
        recommendation = 'Consultation with a gastroenterologist is recommended. Additional imaging and clinical correlation may be needed for definitive diagnosis.'
    elif uc_percentage > 0:
        risk_level = 'low'
        conclusion = 'LOW RISK: Minimal signs detected, mostly normal findings'
        summary = f'Analysis of {total_images} images shows {uc_count} images with mild UC indicators ({uc_percentage:.1f}%). Most images appear normal.'
        recommendation = 'Continue regular monitoring. Discuss findings with your healthcare provider during routine follow-up visits.'
    else:
        risk_level = 'normal'
        conclusion = 'NORMAL: No signs of Ulcerative Colitis detected'
        summary = f'Analysis of all {total_images} colonoscopy images shows normal findings with no indicators of ulcerative colitis.'
        recommendation = 'Continue with regular preventive care and screening as recommended by your healthcare provider.'
    
    return {
        'total_images': total_images,
        'uc_count': uc_count,
        'normal_count': normal_count,
        'uc_percentage': uc_percentage,
        'risk_level': risk_level,
        'conclusion': conclusion,
        'summary': summary,
        'recommendation': recommendation
    }

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return render_template('index.html', error="No files uploaded")
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return render_template('index.html', error="No files selected")
    
    # Process each uploaded image
    image_results = []
    for file in files:
        if file and file.filename != '':
            try:
                result = analyze_image(file)
                image_results.append(result)
            except Exception as e:
                print(f"Error processing {file.filename}: {str(e)}")
                continue
    
    if not image_results:
        return render_template('index.html', error="No valid images could be processed")
    
    # Generate overall conclusion
    conclusion_data = generate_conclusion(image_results)
    
    results = {
        'images': image_results,
        'conclusion': conclusion_data
    }
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
