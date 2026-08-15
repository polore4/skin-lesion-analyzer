<div align="center">
  <a href="#-español">🇪🇸 Español</a> •
  <a href="#-català">🇦🇩 Català</a> •
  <a href="#-english">🇬🇧 English</a>
</div>

---

# 🇪🇸 Español

# Analizador de Lesiones Cutáneas con IA
**Clasificador de Visión Artificial Médica y Deep Learning**

## Autor
* **Pol Orellana Méndez** - *Estudiante de Inteligencia Artificial en la Universidad Autónoma de Barcelona (UAB)*

## Visión General del Proyecto
Este proyecto aborda un problema crítico en la atención médica: los largos tiempos de espera para las consultas dermatológicas, que a menudo retrasan el diagnóstico de cánceres de piel agresivos como el melanoma. 

Nuestra solución es un **Analizador de Lesiones Cutáneas impulsado por IA**, un proyecto de Visión Artificial que clasifica imágenes dermatoscópicas en 7 categorías diagnósticas diferentes. El sistema utiliza un conjunto de datos médicos del mundo real fuertemente desbalanceado (HAM10000). Para abordar este desbalance y evitar que la IA pase por alto casos críticos de cáncer, el sistema implementa **Transfer Learning, Pesos de Clase (penalizando los falsos negativos en melanomas) y Data Augmentation (Aumento de Datos)**. 

El proyecto evolucionó a través de un proceso científico iterativo:
* **Modelo Base:** Una arquitectura `ResNet18` que logró una precisión de ~52%, pero sufrió de sobreajuste (*overfitting*) y altas tasas de falsos positivos debido a la agresiva ponderación de clases.
* **Modelo Avanzado:** Una arquitectura `EfficientNet-B0` que implementa un programador de tasa de aprendizaje (*Learning Rate Scheduler*) y *Early Stopping*, logrando extraer texturas microscópicas mucho más finas y alcanzando una **precisión global de ~83%**.
* **Interfaz Interactiva (Dashboard):** Una aplicación web personalizada construida con Streamlit para que los usuarios finales suban imágenes y reciban predicciones de la IA en tiempo real con sus respectivos porcentajes de confianza.

---

## Obtención de los Datasets (Requisito)

Debido a que este proyecto depende de imágenes médicas de alta resolución (3GB+), **este repositorio es estrictamente para el código**. El conjunto de imágenes y los pesos del modelo entrenado (`.pth`) deben alojarse localmente para eludir los límites de almacenamiento de GitHub.

Para ejecutar este proyecto, primero debes descargar el dataset base:

1. **Descargar los Datos:** Ve a [Kaggle: HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) y descarga el archivo.
2. **Extraer y Estructurar:** 
   * Crea una carpeta `data/` en la raíz de este proyecto.
   * Dentro de `data/`, crea una carpeta `images/`.
   * Mueve **todas** las imágenes `.jpg` de las dos partes de Kaggle a la única carpeta `data/images/`.
   * Coloca el archivo `HAM10000_metadata.csv` directamente dentro de la carpeta `data/`.

*Nota: El archivo `.gitignore` ya está configurado para evitar subir accidentalmente estos archivos masivos a GitHub.*

---

## Configuración e Instalación

**Paso 1: Crear un Entorno Virtual**
Se recomienda encarecidamente utilizar Python 3.11 o 3.12 para garantizar la compatibilidad con PyTorch.

```bash
python -m venv venv
# Activarlo en Windows
.\venv\Scripts\activate
# Activarlo en Mac/Linux
source venv/bin/activate
```

**Paso 2: Instalar Dependencias**
Ejecuta el siguiente comando en tu terminal para instalar Pandas, Scikit-Learn, Streamlit y las herramientas básicas:

```bash
pip install -r requirements.txt
```

**Paso 3: Instalar PyTorch con CUDA (Para Aceleración GPU)**
Para entrenar las redes neuronales de manera eficiente usando una GPU de NVIDIA, instala la versión específica de PyTorch:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

---

## Cómo Ejecutar el Proyecto

Asegúrate de que tu terminal esté abierta en la raíz del repositorio del proyecto y que tu entorno virtual esté activo. Ejecuta los Jupyter Notebooks en orden estrictamente secuencial para procesar los datos crudos y entrenar la IA, y luego lanza la aplicación web.

### 1. Exploración y Preparación de Datos
Ejecuta estos notebooks para analizar el desbalance de los datos y mapear las rutas absolutas de las imágenes.
* `notebooks/01_eda_ham10000.ipynb`
* `notebooks/02_data_preparation.ipynb`

### 2. DataLoaders de PyTorch
Configura la clase personalizada `Dataset`, divide los datos (80/20) y aplica el *Data Augmentation* geométrico.
* `notebooks/03_pytorch_dataset.ipynb`

### 3. Entrenamiento y Evaluación del Modelo (Iteración Científica)
* **Modelo Base:** Ejecuta `04_model_training.ipynb` y `05_model_evaluation.ipynb` para ver el rendimiento de ResNet18.
* **Modelo Avanzado (Recomendado):** Ejecuta `06_advanced_training.ipynb` para entrenar el modelo EfficientNet-B0, seguido de `07_advanced_evaluation.ipynb` para generar la Matriz de Confusión final y el Reporte de Clasificación.

### 4. Lanzar la Interfaz Interactiva
Despliega la interfaz de usuario de Streamlit con el estilo personalizado para el usuario final.

```bash
streamlit run src/app.py
```

---

## Análisis Profundo: Estructura del Repositorio y Lógica del Código

Nuestro código base ha sido estructurado en un pipeline profesional separando el análisis exploratorio de datos, la ingeniería de deep learning y el despliegue frontend.

### 1. La Aplicación Principal (Frontend)
**Carpeta `src/`**
* **`app.py`:** El dashboard principal de Streamlit. Carga dinámicamente los pesos del modelo `EfficientNet-B0` entrenado (`.pth`), procesa las imágenes subidas por el usuario en tiempo real aplicando las mismas normalizaciones exactas usadas durante el entrenamiento, y devuelve una distribución de confianza Softmax. Cuenta con una inyección CSS personalizada para reflejar la interfaz visual (UI/UX) de la marca personal del creador.
* **`.streamlit/config.toml`:** Configuración global del tema que establece el modo oscuro y las paletas de colores de la marca.

### 2. Lógica de Machine Learning (Backend)
**Carpeta `notebooks/`** (El Laboratorio de Investigación y Entrenamiento)
* **`01_eda_ham10000.ipynb`:** Análisis exploratorio de datos inicial graficando la distribución de clases utilizando Seaborn.
* **`02_data_preparation.ipynb`:** Mapeos y limpieza del dataset. Genera el archivo `HAM10000_metadata_prepared.csv`.
* **`03_pytorch_dataset.ipynb`:** Contiene la lógica para el `DataLoader` de PyTorch y sus `transforms`. Es crucial para evitar cuellos de botella en la memoria RAM durante el entrenamiento con GPU.
* **`04_model_training.ipynb` & `06_advanced_training.ipynb`:** Los bucles de entrenamiento principales. Implementan *Backpropagation*, *CrossEntropyLoss* con pesos de clase, optimizadores Adam, programadores de tasa de aprendizaje (`ReduceLROnPlateau`) y mecanismos de *Early Stopping*.
* **`05_model_evaluation.ipynb` & `07_advanced_evaluation.ipynb`:** Scripts de validación que desactivan los gradientes (`torch.no_grad()`) para ejecutar los modelos contra el 20% del set de prueba reservado, generando métricas de precisión/exhaustividad y mapas de calor visuales (Matrices de Confusión).

### 3. Datos y Recursos de Marca
**Carpeta `data/`** *(Ignorada por Git)*
* Aloja los metadatos crudos en `.csv`, las 10,015 imágenes médicas en `.jpg` y los pesos de los modelos guardados en `.pth`.
* Incluye los recursos de la marca (`perfil.png`, `linkedin.png`, `github.png`) codificados dinámicamente en base64 por el frontend para renderizar la barra lateral personal del creador.

<br><br>

---

# 🇦🇩 Català

# Analitzador de Lesions Cutànies amb IA
**Classificador de Visió Artificial Mèdica i Deep Learning**

## Autor
* **Pol Orellana Méndez** - *Estudiant d'Intel·ligència Artificial a la Universitat Autònoma de Barcelona (UAB)*

## Visió General del Projecte
Aquest projecte aborda un problema crític en l'atenció mèdica: els llargs temps d'espera per a les consultes dermatològiques, que sovint endarrereixen el diagnòstic de càncers de pell agressius com el melanoma. 

La nostra solució és un **Analitzador de Lesions Cutànies impulsat per IA**, un projecte de Visió Artificial que classifica imatges dermatoscòpiques en 7 categories diagnòstiques diferents. El sistema utilitza un conjunt de dades mèdiques del món real fortament desbalancejat (HAM10000). Per abordar aquest desequilibri i evitar que la IA passi per alt casos crítics de càncer, el sistema implementa **Transfer Learning, Pesos de Classe (penalitzant els falsos negatius en melanomes) i Data Augmentation (Augment de Dades)**. 

El projecte va evolucionar a través d'un procés científic iteratiu:
* **Model Base:** Una arquitectura `ResNet18` que va assolir una precisió de ~52%, però va patir de sobreajust (*overfitting*) i altes taxes de falsos positius a causa de l'agressiva ponderació de classes.
* **Model Avançat:** Una arquitectura `EfficientNet-B0` que implementa un programador de taxa d'aprenentatge (*Learning Rate Scheduler*) i *Early Stopping*, aconseguint extreure textures microscòpiques molt més fines i assolint una **precisió global de ~83%**.
* **Interfície Interactiva (Dashboard):** Una aplicació web personalitzada construïda amb Streamlit perquè els usuaris finals pugin imatges i rebin prediccions de la IA en temps real amb els seus respectius percentatges de confiança.

---

## Obtenció dels Datasets (Requisit)

Com que aquest projecte depèn d'imatges mèdiques d'alta resolució (3GB+), **aquest repositori és estrictament per al codi**. El conjunt d'imatges i els pesos del model entrenat (`.pth`) s'han d'allotjar localment per eludir els límits d'emmagatzematge de GitHub.

Per executar aquest projecte, primer has de descarregar el dataset base:

1. **Descarregar les Dades:** Ves a [Kaggle: HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) i descarrega l'arxiu.
2. **Extreure i Estructurar:** 
   * Crea una carpeta `data/` a l'arrel d'aquest projecte.
   * Dins de `data/`, crea una carpeta `images/`.
   * Mou **totes** les imatges `.jpg` de les dues parts de Kaggle a l'única carpeta `data/images/`.
   * Col·loca l'arxiu `HAM10000_metadata.csv` directament dins de la carpeta `data/`.

*Nota: L'arxiu `.gitignore` ja està configurat per evitar pujar accidentalment aquests arxius massius a GitHub.*

---

## Configuració i Instal·lació

**Pas 1: Crear un Entorn Virtual**
Es recomana encaridament utilitzar Python 3.11 o 3.12 per garantir la compatibilitat amb PyTorch.

```bash
python -m venv venv
# Activar-lo a Windows
.\venv\Scripts\activate
# Activar-lo a Mac/Linux
source venv/bin/activate
```

**Pas 2: Instal·lar Dependències**
Executa el següent comandament a la teva terminal per instal·lar Pandas, Scikit-Learn, Streamlit i les eines bàsiques:

```bash
pip install -r requirements.txt
```

**Pas 3: Instal·lar PyTorch amb CUDA (Per a Acceleració GPU)**
Per entrenar les xarxes neuronals de manera eficient usant una GPU de NVIDIA, instal·la la versió específica de PyTorch:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

---

## Com Executar el Projecte

Assegura't que la teva terminal estigui oberta a l'arrel del repositori del projecte i que el teu entorn virtual estigui actiu. Executa els Jupyter Notebooks en ordre estrictament seqüencial per processar les dades crues i entrenar la IA, i després llança l'aplicació web.

### 1. Exploració i Preparació de Dades
Executa aquests notebooks per analitzar el desequilibri de les dades i mapejar les rutes absolutes de les imatges.
* `notebooks/01_eda_ham10000.ipynb`
* `notebooks/02_data_preparation.ipynb`

### 2. DataLoaders de PyTorch
Configura la classe personalitzada `Dataset`, divideix les dades (80/20) i aplica el *Data Augmentation* geomètric.
* `notebooks/03_pytorch_dataset.ipynb`

### 3. Entrenament i Avaluació del Model (Iteració Científica)
* **Model Base:** Executa `04_model_training.ipynb` i `05_model_evaluation.ipynb` per veure el rendiment de ResNet18.
* **Model Avançat (Recomanat):** Executa `06_advanced_training.ipynb` per entrenar el model EfficientNet-B0, seguit de `07_advanced_evaluation.ipynb` per generar la Matriu de Confusió final i l'Informe de Classificació.

### 4. Llançar la Interfície Interactiva
Desplega la interfície d'usuari de Streamlit amb l'estil personalitzat per a l'usuari final.

```bash
streamlit run src/app.py
```

---

## Anàlisi Profunda: Estructura del Repositori i Lògica del Codi

El nostre codi base ha estat estructurat en un pipeline professional separant l'anàlisi exploratòria de dades, l'enginyeria de deep learning i el desplegament frontend.

### 1. L'Aplicació Principal (Frontend)
**Carpeta `src/`**
* **`app.py`:** El dashboard principal de Streamlit. Carrega dinàmicament els pesos del model `EfficientNet-B0` entrenat (`.pth`), processa les imatges pujades per l'usuari en temps real aplicant les mateixes normalitzacions exactes usades durant l'entrenament, i retorna una distribució de confiança Softmax. Compta amb una injecció CSS personalitzada per reflectir la interfície visual (UI/UX) de la marca personal del creador.
* **`.streamlit/config.toml`:** Configuració global del tema que estableix el mode fosc i les paletes de colors de la marca.

### 2. Lògica de Machine Learning (Backend)
**Carpeta `notebooks/`** (El Laboratori d'Investigació i Entrenament)
* **`01_eda_ham10000.ipynb`:** Anàlisi exploratòria de dades inicial graficant la distribució de classes utilitzant Seaborn.
* **`02_data_preparation.ipynb`:** Mapejos i neteja del dataset. Genera l'arxiu `HAM10000_metadata_prepared.csv`.
* **`03_pytorch_dataset.ipynb`:** Conté la lògica per al `DataLoader` de PyTorch i els seus `transforms`. És crucial per evitar colls d'ampolla en la memòria RAM durant l'entrenament amb GPU.
* **`04_model_training.ipynb` & `06_advanced_training.ipynb`:** Els bucles d'entrenament principals. Implementen *Backpropagation*, *CrossEntropyLoss* amb pesos de classe, optimitzadors Adam, programadors de taxa d'aprenentatge (`ReduceLROnPlateau`) i mecanismes d'*Early Stopping*.
* **`05_model_evaluation.ipynb` & `07_advanced_evaluation.ipynb`:** Scripts de validació que desactiven els gradients (`torch.no_grad()`) per executar els models contra el 20% del set de prova reservat, generant mètriques de precisió/exhaustivitat i mapes de calor visuals (Matrius de Confusió).

### 3. Dades i Recursos de Marca
**Carpeta `data/`** *(Ignorada per Git)*
* Allotja les metadades crues en `.csv`, les 10.015 imatges mèdiques en `.jpg` i els pesos dels models guardats en `.pth`.
* Inclou els recursos de la marca (`perfil.png`, `linkedin.png`, `github.png`) codificats dinàmicament en base64 pel frontend per renderitzar la barra lateral personal del creador.

<br><br>

---

# 🇬🇧 English

# AI Skin Lesion Analyzer
**Medical Computer Vision & Deep Learning Classifier**

## Author
* **Pol Orellana Méndez** - *Artificial Intelligence Student at Universidad Autónoma de Barcelona (UAB)*

## Project Overview
This project addresses a critical healthcare issue: the long waiting times for dermatological consultations, which often delay the diagnosis of aggressive skin cancers like melanoma. 

Our solution is an **AI-driven Skin Lesion Analyzer**, a Computer Vision project that classifies dermatoscopic images into 7 different diagnostic categories. The system utilizes a heavily imbalanced real-world medical dataset (HAM10000). To tackle this imbalance and prevent the AI from missing critical cancer cases, the system implements **Transfer Learning, Class Weights (penalizing false negatives for melanoma), and Data Augmentation**. 

The project evolved through a scientific iterative process:
* **Baseline Model:** A `ResNet18` architecture that achieved ~52% accuracy but suffered from overfitting and high false-positive rates due to the aggressive class weighting.
* **Advanced Model:** An `EfficientNet-B0` architecture implementing a Learning Rate Scheduler and Early Stopping, which successfully extracted finer microscopic textures and achieved a **~83% global accuracy**.
* **Interactive Dashboard:** A custom Streamlit web interface for end-users to upload images and receive real-time AI predictions with confidence percentages.

---

## Obtaining the Datasets (Required)

Because this project relies on high-resolution medical images (3GB+), **this repository is strictly for code**. The image dataset and the trained model weights (`.pth`) must be hosted locally to bypass GitHub's storage limits.

To run this project, you must first download the foundational dataset:

1. **Download the Data:** Go to [Kaggle: HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and download the archive.
2. **Extract and Structure:** 
   * Create a `data/` folder in the root of this project.
   * Inside `data/`, create an `images/` folder.
   * Move **all** the `.jpg` images from both Kaggle parts into the single `data/images/` folder.
   * Place the `HAM10000_metadata.csv` directly inside the `data/` folder.

*Note: The `.gitignore` is already configured to prevent accidentally pushing these massive files to GitHub.*

---

## Setup & Configuration

**Step 1: Create a Virtual Environment**
It is highly recommended to use Python 3.11 or 3.12 for PyTorch compatibility.

```bash
python -m venv venv
# Activate it (Windows)
.\venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate
```

**Step 2: Install Dependencies**
Run the following command in your terminal to install Pandas, Scikit-Learn, Streamlit, and basic tools:

```bash
pip install -r requirements.txt
```

**Step 3: Install PyTorch with CUDA (For GPU Acceleration)**
To train the neural networks efficiently using an NVIDIA GPU, install the specific PyTorch build:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

---

## How to Run the Project

Ensure your terminal is open at the root of the project repository and your virtual environment is active. Execute the Jupyter Notebooks in strict sequential order to process the raw data and train the AI, then launch the dashboard.

### 1. Data Exploration and Preparation
Run these notebooks to analyze the dataset imbalance and map the absolute paths of the images.
* `notebooks/01_eda_ham10000.ipynb`
* `notebooks/02_data_preparation.ipynb`

### 2. PyTorch DataLoaders
Sets up the custom `Dataset` class, splits the data (80/20), and applies geometric Data Augmentation.
* `notebooks/03_pytorch_dataset.ipynb`

### 3. Model Training & Evaluation (Scientific Iteration)
* **Baseline:** Run `04_model_training.ipynb` and `05_model_evaluation.ipynb` to see the ResNet18 performance.
* **Advanced (Recommended):** Run `06_advanced_training.ipynb` to train the EfficientNet-B0 model, followed by `07_advanced_evaluation.ipynb` to generate the final Confusion Matrix and Classification Report.

### 4. Launch the Interactive Dashboard
Spins up the custom-styled Streamlit UI for the final user.

```bash
streamlit run src/app.py
```

---

## Deep Dive: Repository Structure & Code Logic

Our codebase has been structured into a professional pipeline separating exploratory data analysis, deep learning engineering, and frontend deployment.

### 1. The Core Application (Frontend)
**`src/` Folder**
* **`app.py`:** The main Streamlit dashboard. It dynamically loads the trained `EfficientNet-B0` model weights (`.pth`), processes user-uploaded images in real-time applying the exact same normalizations used during training, and outputs a Softmax confidence distribution. It features a custom CSS injection to reflect the creator's personal brand UI/UX.
* **`.streamlit/config.toml`:** Global theme configuration setting up the dark mode and brand color palettes.

### 2. Machine Learning Logic (Backend)
**`notebooks/` Folder** (The Research & Training Lab)
* **`01_eda_ham10000.ipynb`:** Initial exploratory data analysis plotting class distribution using Seaborn.
* **`02_data_preparation.ipynb`:** Mappings and dataset cleaning. Generates the `HAM10000_metadata_prepared.csv`.
* **`03_pytorch_dataset.ipynb`:** Contains the logic for the PyTorch `DataLoader` and `transforms`. Crucial for preventing RAM bottlenecks during GPU training.
* **`04_model_training.ipynb` & `06_advanced_training.ipynb`:** The core training loops. They implement Backpropagation, CrossEntropyLoss with class weights, Adam optimizers, Learning Rate Schedulers (`ReduceLROnPlateau`), and Early Stopping mechanisms.
* **`05_model_evaluation.ipynb` & `07_advanced_evaluation.ipynb`:** Validation scripts that disable gradients (`torch.no_grad()`) to run the models against the 20% holdout test set, generating precision/recall metrics and visual Heatmaps (Confusion Matrices).

### 3. Data & Brand Assets
**`data/` Folder** *(Ignored by Git)*
* Houses the raw `.csv` metadata, the 10,015 `.jpg` medical images, and the saved `.pth` model weights.
* Includes brand assets (`perfil.png`, `linkedin.png`, `github.png`) encoded dynamically into base64 by the frontend to render the creator's personal sidebar.