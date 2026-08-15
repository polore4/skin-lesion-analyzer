import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import base64
import os

# 1. Configuración de la página (Cambiado a layout="wide")
st.set_page_config(page_title="AI Skin Lesion Analyzer | Pol Orellana", page_icon="🔬", layout="wide")

# --- FUNCIÓN AUXILIAR PARA IMÁGENES ---
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

img_perfil = get_base64_of_bin_file("data/perfil.png")
img_linkedin = get_base64_of_bin_file("data/linkedin.png")
img_github = get_base64_of_bin_file("data/github.png")

# --- MARCA PERSONAL: Menú Lateral con HTML/CSS ---
with st.sidebar:
    st.markdown("## Sobre Mí")
    
    if img_perfil:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{img_perfil}" style="border-radius: 50%; width: 150px; height: 150px; object-fit: cover; border: 3px solid #56a996; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    st.markdown("""
    **Pol Orellana Méndez**  
    *Estudiante de Inteligencia Artificial en la Universidad Autónoma de Barcelona*
    
    Apasionado por el Machine Learning, la Visión Artificial y el desarrollo de soluciones tecnológicas con impacto real.
    """)
    st.markdown("---")
    
    # Botones unificados con el color primario de tu marca (#56a996)
    st.markdown(
        f"""
        <style>
        .custom-btn {{
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #56a996; 
            color: white !important;
            padding: 10px 15px;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            margin-bottom: 10px;
            transition: background-color 0.3s;
        }}
        .custom-btn:hover {{ background-color: #458778; }}
        </style>
        
        <a href="https://www.linkedin.com/in/pol-orellana/" target="_blank" class="custom-btn">
            <img src="data:image/png;base64,{img_linkedin}" width="120" style="margin-right: 10px;">
        </a>
        <a href="https://github.com/polore4" target="_blank" class="custom-btn">
            <img src="data:image/png;base64,{img_github}" width="120" style="margin-right: 10px;">
        </a>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.caption("© 2026 Pol Orellana. Todos los derechos reservados.")
# ------------------------------------

# Diccionario de clases
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

# 2. Función para cargar el modelo
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 7)
    
    try:
        model.load_state_dict(torch.load('data/best_efficientnet_model.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        st.error("No se ha encontrado el archivo del modelo. Asegúrate de haberlo entrenado.")
        
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 4. Interfaz Principal
st.title("🔬 AI Skin Lesion Analyzer")
st.write("Sube una imagen dermatoscópica para analizar el riesgo de la lesión.")

# --- INYECCIÓN DE CSS PARA EL PANEL DE FONDO ---
st.markdown("""
<style>
/* 1. Creamos la clase oculta que nos sirve de gancho */
.mi-tarjeta {
    display: none;
}

/* 2. Apuntamos al contenedor exterior (stLayoutWrapper) para el fondo y la sombra */
div[data-testid="stLayoutWrapper"]:has(.mi-tarjeta) {
    background-color: #394346 !important;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

/* 3. Apuntamos al div stVerticalBlock para quitar el borde y ajustar el padding a 10px */
div[data-testid="stVerticalBlock"]:has(.mi-tarjeta) {
    border: none !important;
    padding: 20px !important;
}

/* 4. Zona de subir archivo con el color de fondo de la app (#222e31) */
[data-testid="stFileUploaderDropzone"] {
    background-color: #222e31 !important;
    border: 1px dashed #56a996 !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --- EL PANEL DE FONDO ---
with st.container(border=True):
    # Inyectamos nuestra clase "gancho" para que el CSS superior sepa qué fondo pintar
    st.markdown('<div class="mi-tarjeta"></div>', unsafe_allow_html=True)
    
    st.warning("⚠️ **Aviso Médico:** Esta aplicación es un proyecto de demostración de IA y no sustituye el diagnóstico de un dermatólogo profesional.")
    
    uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Imagen a analizar', use_container_width=True)
            analyze_btn = st.button('Analizar Lesión', use_container_width=True, type="primary")
        
        with col2:
            if analyze_btn:
                with st.spinner('La IA está analizando las texturas...'):
                    img_tensor = preprocess_image(image).to(device)
                    
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    predicted_class = class_names[predicted_idx.item()]
                    confidence_percentage = confidence.item() * 100
                    
                    st.markdown("### Resultado")
                    if predicted_class in ['mel', 'bcc']:
                        st.error(f"**Diagnóstico:** {full_names[predicted_class]}")
                    elif predicted_class == 'akiec':
                        st.warning(f"**Diagnóstico:** {full_names[predicted_class]}")
                    else:
                        st.success(f"**Diagnóstico:** {full_names[predicted_class]}")
                        
                    st.info(f"**Confianza de la IA:** {confidence_percentage:.2f}%")
                    st.progress(int(confidence_percentage))