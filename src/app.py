import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# 1. Configuración de la página
st.set_page_config(page_title="AI Skin Lesion Analyzer", page_icon="🔬", layout="centered")

# Diccionario de clases para que el usuario lo entienda
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
full_names = {
    'akiec': 'Queratosis Actínica (Pre-cáncer)',
    'bcc': 'Carcinoma Basocelular (Cáncer de piel)',
    'bkl': 'Lesión Benigna (Queratosis)',
    'df': 'Dermatofibroma (Benigno)',
    'mel': 'Melanoma (Cáncer de piel agresivo)',
    'nv': 'Nevus Melanocítico (Lunar común)',
    'vasc': 'Lesión Vascular (Benigna)'
}

# 2. Función para cargar el modelo (Caché para que no se recargue cada vez)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 7)
    
    # Cargamos los pesos de nuestro mejor modelo
    model.load_state_dict(torch.load('data/best_efficientnet_model.pth', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# 3. Función para preparar la imagen
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Añadimos la dimensión del batch

# 4. Interfaz de Usuario
st.title("AI Skin Lesion Analyzer")
st.write("Sube una imagen dermatoscópica para analizar el riesgo de la lesión.")
st.warning("**Aviso Médico:** Esta aplicación es un proyecto de demostración de IA y no sustituye el diagnóstico de un dermatólogo profesional.")

uploaded_file = st.file_uploader("Selecciona una imagen de un lunar o lesión...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen subida', use_container_width=True)
    
    # Botón de predicción
    if st.button('Analizar Lesión'):
        with st.spinner('La IA está analizando las texturas...'):
            # Preprocesar y predecir
            img_tensor = preprocess_image(image).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                # Aplicamos Softmax para convertir la salida en porcentajes de probabilidad
                probabilities = F.softmax(outputs, dim=1) 
                
            # Obtener el resultado con mayor probabilidad
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = class_names[predicted_idx.item()]
            confidence_percentage = confidence.item() * 100
            
            # Mostrar resultados
            st.markdown("###Resultado del Análisis")
            st.success(f"**Diagnóstico de la IA:** {full_names[predicted_class]}")
            st.info(f"**Nivel de confianza:** {confidence_percentage:.2f}%")
            
            # Mostrar barra de progreso para la confianza
            st.progress(int(confidence_percentage))