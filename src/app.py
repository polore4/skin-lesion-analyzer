import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import base64
import os
import io

# 1. Configuración de la página
st.set_page_config(
    page_title="AI Skin Lesion Analyzer | Pol Orellana", 
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="expanded" # Aseguramos que siempre inicie abierto
)

# --- DICCIONARIO DE TRADUCCIONES ---
TRANSLATIONS = {
    "🇪🇸 Español": {
        "about_me": "## Sobre Mí",
        "desc": "**Pol Orellana Méndez**\n\n*Estudiante de Inteligencia Artificial en la Universidad Autónoma de Barcelona*\n\nApasionado por el Machine Learning, la Visión Artificial y el desarrollo de soluciones tecnológicas con impacto real.",
        "rights": "© 2026 Pol Orellana. Todos los derechos reservados.",
        "title": "Analizador de lesiones cutáneas con IA",
        "subtitle": "Sube una imagen dermatoscópica para analizar el riesgo de la lesión.",
        "warning": "⚠️ **Aviso Médico:** Esta aplicación es un proyecto de demostración de IA y no sustituye el diagnóstico de un dermatólogo profesional.",
        "privacy": "🔒 **Privacidad:** Las imágenes se procesan temporalmente en la memoria del servidor y **no se guardan** ni se utilizan para reentrenar el modelo. Se eliminan automáticamente al cerrar la página.",
        "upload": "Selecciona una imagen...",
        "caption": "Imagen a analizar",
        "btn_analyze": "Analizar Lesión",
        "btn_change": "Cambiar imagen",
        "spinner": "La IA está analizando las texturas...",
        "result": "### Resultado",
        "diag": "**Diagnóstico:**",
        "conf": "**Confianza de la IA:**",
        "error_model": "No se ha encontrado el archivo del modelo. Asegúrate de haberlo entrenado.",
        "explanation_title": "¿Cómo funciona?",
        "explanation_text": """
        **1. Sube tu imagen**  
        Utiliza el recuadro de la izquierda para arrastrar y soltar una imagen dermatoscópica (un lunar o mancha).
        
        **2. Procesamiento de Inteligencia Artificial**  
        Nuestro modelo basado en la arquitectura *EfficientNet-B0* extraerá y procesará las texturas y patrones microscópicos de la lesión utilizando *Computer Vision*.
        
        **3. Resultados en tiempo real**  
        La red neuronal clasificará la imagen entre 7 categorías médicas distintas y te proporcionará un porcentaje de confianza instantáneo.
        """,
        "classes": {
            'akiec': 'Queratosis Actínica (Pre-cáncer)',
            'bcc': 'Carcinoma Basocelular (Cáncer de piel)',
            'bkl': 'Lesión Benigna (Queratosis)',
            'df': 'Dermatofibroma (Benigno)',
            'mel': 'Melanoma (Cáncer de piel agresivo)',
            'nv': 'Nevus Melanocítico (Lunar común)',
            'vasc': 'Lesión Vascular (Benigna)'
        }
    },
    "🇦🇩 Català": {
        "about_me": "## Sobre Mi",
        "desc": "**Pol Orellana Méndez**\n\n*Estudiant d'Intel·ligència Artificial a la Universitat Autònoma de Barcelona*\n\nApassionat pel Machine Learning, la Visió Artificial i el desenvolupament de solucions tecnològiques amb impacte real.",
        "rights": "© 2026 Pol Orellana. Tots els drets reservats.",
        "title": "Analitzador de lesions cutànies per IA",
        "subtitle": "Puja una imatge dermatoscòpica per analitzar el risc de la lesió.",
        "warning": "⚠️ **Avís Mèdic:** Aquesta aplicació és un projecte de demostració d'IA i no substitueix el diagnòstic d'un dermatòleg professional.",
        "privacy": "🔒 **Privacitat:** Les imatges es processen temporalment a la memòria del servidor i **no es guarden** ni s'utilitzen per reentrenar el model. S'eliminen automàticament en tancar la pàgina.",
        "upload": "Selecciona una imatge...",
        "caption": "Imatge a analitzar",
        "btn_analyze": "Analitzar Lesió",
        "btn_change": "Canviar imatge",
        "spinner": "La IA està analitzant les textures...",
        "result": "### Resultat",
        "diag": "**Diagnòstic:**",
        "conf": "**Confiança de la IA:**",
        "error_model": "No s'ha trobat l'arxiu del model. Assegura't d'haver-lo entrenat.",
        "explanation_title": "Com funciona?",
        "explanation_text": """
        **1. Puja la teva imatge**  
        Utilitza el requadre de l'esquerra per arrossegar i deixar anar una imatge dermatoscòpica (una piga o taca).
        
        **2. Processament d'Intel·ligència Artificial**  
        El nostre model basat en l'arquitectura *EfficientNet-B0* extraurà i processarà les textures i patrons microscòpics de la lesió utilitzant *Computer Vision*.
        
        **3. Resultats en temps real**  
        La xarxa neuronal classificarà la imatge entre 7 categories mèdiques diferents i et proporcionarà un percentatge de confiança instantani.
        """,
        "classes": {
            'akiec': 'Queratosi Actínica (Pre-càncer)',
            'bcc': 'Carcinoma Basocel·lular (Càncer de pell)',
            'bkl': 'Lesió Benigna (Queratosi)',
            'df': 'Dermatofibroma (Benigne)',
            'mel': 'Melanoma (Càncer de pell agressiu)',
            'nv': 'Nevus Melanocític (Piga comuna)',
            'vasc': 'Lesió Vascular (Benigna)'
        }
    },
    "🇬🇧 English": {
        "about_me": "## About Me",
        "desc": "**Pol Orellana Méndez**\n\n*Artificial Intelligence Student at Autonomous University of Barcelona*\n\nPassionate about Machine Learning, Computer Vision, and developing technological solutions with real impact.",
        "rights": "© 2026 Pol Orellana. All rights reserved.",
        "title": "AI Skin Lesion Analyzer",
        "subtitle": "Upload a dermatoscopic image to analyze the lesion's risk.",
        "warning": "⚠️ **Medical Disclaimer:** This application is an AI demonstration project and does not replace a professional dermatologist's diagnosis.",
        "privacy": "🔒 **Privacy:** Images are processed temporarily in the server's memory and are **not saved** or used to retrain the model. They are automatically deleted when you close the page.",
        "upload": "Select an image...",
        "caption": "Image to analyze",
        "btn_analyze": "Analyze Lesion",
        "btn_change": "Change image",
        "spinner": "The AI is analyzing textures...",
        "result": "### Result",
        "diag": "**Diagnosis:**",
        "conf": "**AI Confidence:**",
        "error_model": "Model file not found. Make sure you have trained it.",
        "explanation_title": "How it works?",
        "explanation_text": """
        **1. Upload your image**  
        Use the box on the left to drag and drop a dermatoscopic image (a mole or spot).
        
        **2. Artificial Intelligence Processing**  
        Our model, based on the *EfficientNet-B0* architecture, will extract and process the microscopic textures and patterns of the lesion using *Computer Vision*.
        
        **3. Real-time Results**  
        The neural network will classify the image into 7 different medical categories and provide an instant confidence percentage.
        """,
        "classes": {
            'akiec': 'Actinic Keratoses (Pre-cancer)',
            'bcc': 'Basal Cell Carcinoma (Skin cancer)',
            'bkl': 'Benign Lesion (Keratosis)',
            'df': 'Dermatofibroma (Benign)',
            'mel': 'Melanoma (Aggressive skin cancer)',
            'nv': 'Melanocytic Nevus (Common mole)',
            'vasc': 'Vascular Lesion (Benign)'
        }
    }
}

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

# --- INYECCIÓN DE CSS ---
st.markdown("""
<style>
/* 1. COMPACTAR SIDEBAR PARA ELIMINAR EL SCROLL */
[data-testid="stSidebar"] hr { margin: 0.7rem 0 !important; }
[data-testid="stSidebar"] h2 { padding-top: 0 !important; padding-bottom: 0 !important; margin-bottom: 0.5rem !important; }
[data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.4rem !important; }
[data-testid="stSidebarUserContent"] { padding-top: 2rem !important; }

/* 2. ESTILOS DE LA APP PRINCIPAL PARA EVITAR SCROLL VERTICAL */
header { visibility: hidden !important; } /* Ocultar el header vacío de arriba */

div[data-testid="stAppViewBlockContainer"] { 
    max-width: 95% !important; 
    padding-top: 1rem !important;  /* Quitar espacio por arriba */
    padding-bottom: 0rem !important; /* Quitar espacio por abajo */
    padding-left: 2rem !important; 
    padding-right: 2rem !important; 
}

/* Controlar la imagen para que JAMÁS haga scroll hacia abajo */
[data-testid="stImage"] img {
    max-height: 42vh !important; /* Límite de alto basado en la pantalla */
    object-fit: contain !important; /* Mantiene la proporción sin recortar el lunar */
    border-radius: 8px !important;
}

/* 3. ESTILOS DEL PANEL Y EL UPLOADER */
.mi-tarjeta { display: none; }
div[data-testid="stLayoutWrapper"]:has(.mi-tarjeta) { background-color: #394346 !important; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
div[data-testid="stVerticalBlock"]:has(.mi-tarjeta) { border: none !important; padding: 15px 25px !important; }

[data-testid="stFileUploaderDropzone"] {
    background-color: #222e31 !important;
    border: 2px dashed #56a996 !important;
    border-radius: 15px !important;
    padding: 3rem 1rem !important;
    display: flex; justify-content: center; align-items: center; flex-direction: column;
}
[data-testid="stFileUploaderDropzone"] svg { color: #56a996 !important; width: 50px !important; height: 50px !important; margin-bottom: 10px; }
[data-testid="stFileUploaderDropzone"] button { background-color: #56a996 !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 0.5rem 1.5rem !important; font-weight: bold !important; }
[data-testid="stFileUploaderDropzone"] button:hover { background-color: #458778 !important; }

/* 4. BLOQUEAR EL SIDEBAR (No se puede ocultar) */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# --- Menú Lateral con HTML/CSS ---
with st.sidebar:
    lang_choice = st.selectbox("🌐 Idioma / Language", ["🇪🇸 Español", "🇦🇩 Català", "🇬🇧 English"])
    t = TRANSLATIONS[lang_choice]
    
    st.markdown("---")
    st.markdown(t["about_me"])
    
    if img_perfil:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin-bottom: 10px;">
                <img src="data:image/png;base64,{img_perfil}" style="border-radius: 50%; width: 130px; height: 130px; object-fit: cover; border: 3px solid #56a996; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    st.markdown(t["desc"])
    st.markdown("---")
    
    st.markdown(
        f"""
        <style>
        .custom-btn {{
            display: flex; align-items: center; justify-content: center;
            background-color: #56a996; color: white !important; padding: 10px 15px;
            text-decoration: none; border-radius: 8px; font-weight: bold; margin-bottom: 10px;
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
    st.caption(t["rights"])

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 7)
    
    try:
        model.load_state_dict(torch.load('data/best_efficientnet_model.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        st.error(t["error_model"])
        
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
st.title(t["title"])
st.write(t["subtitle"])

# --- EL PANEL DE FONDO ---
with st.container(border=True):
    st.markdown('<div class="mi-tarjeta"></div>', unsafe_allow_html=True)
    st.warning(t["warning"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Inicializamos la memoria de la imagen si no existe
        if 'image_buffer' not in st.session_state:
            st.session_state.image_buffer = None
            
        # Si NO hay imagen en memoria, mostramos la caja gigante de subir archivo
        if st.session_state.image_buffer is None:
            uploaded_file = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Si se sube una foto, la guardamos en memoria y recargamos la app instantáneamente
                st.session_state.image_buffer = uploaded_file.getvalue()
                st.rerun()
            analyze_btn = False # El botón analizar no existe aún
            
        # Si YA HAY imagen en memoria, el uploader DESAPARECE y ponemos la foto
        else:
            image = Image.open(io.BytesIO(st.session_state.image_buffer)).convert('RGB')
            st.image(image, caption=t["caption"], use_container_width=True)
            
            # Colocamos dos botones pequeños debajo de la foto
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                analyze_btn = st.button(t["btn_analyze"], use_container_width=True, type="primary")
            with col_btn2:
                # Si le da a cambiar imagen, borramos la memoria y recargamos
                if st.button(t["btn_change"], use_container_width=True):
                    st.session_state.image_buffer = None
                    st.rerun()
            
    with col2:
        # Si NO se ha pulsado el botón de analizar, mostramos la explicación
        if not analyze_btn:
            st.markdown(f"### {t['explanation_title']}")
            st.markdown(t['explanation_text'])
            
        # Si se HA PULSADO el botón, la explicación desaparece y corremos la IA
        else:
            with st.spinner(t["spinner"]):
                img_tensor = preprocess_image(image).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_class = class_names[predicted_idx.item()]
                confidence_percentage = confidence.item() * 100
                
                st.markdown(t["result"])
                diag_name = t["classes"][predicted_class]
                
                if predicted_class in ['mel', 'bcc']:
                    st.error(f"{t['diag']} {diag_name}")
                elif predicted_class == 'akiec':
                    st.warning(f"{t['diag']} {diag_name}")
                else:
                    st.success(f"{t['diag']} {diag_name}")
                    
                st.info(f"{t['conf']} {confidence_percentage:.2f}%")
                st.progress(int(confidence_percentage))
                
st.info(t["privacy"])