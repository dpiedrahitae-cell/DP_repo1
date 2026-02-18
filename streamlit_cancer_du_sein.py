import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path 


BASE_DIR = Path(__file__).resolve().parent


df2 = pd.read_csv(BASE_DIR / "breakhist_metadatavf.csv")
# Affiche les premières lignes du DataFrame pour avoir un aperçu des données

st.title("Suite IA pour le cancer du sein")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration - DataViz", "Modélisation","Resultats", "Introduction II", "Exploration - DataViz II", "Modélisation II", "Resultats II", "Interprétabilité","Conclusions","Conclusions II", "Limitations"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.write("Ichraf CHERIF, Diana PIEDRAHITA, Deborah RUBSTEIN")
    
    st.header("Partie 1 -  Prédiction de survie et du risque de récidive (METABRIC)")
    st.write("Le jeu de données METABRIC: contiennent 2509 patientes et 34 variables cliniques et moléculaires")
    st.subheader("Objectifs du projet")
    st.markdown("""
    <div style="font-size:18px;">
    <ul>
    <li>Prédire la survie des patientes atteintes de cancer du sein</li>
    <li>Identifier les variables pronostiques majeures</li>
    <li>Construire des groupes de risque cliniquement interprétables</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

if page == pages [1] :
    st.header("Data Visualization")
    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    La corrélation est centrale pour décider quelles variables garder (ou non) avant une modélisation de survie. 
    </p>
    """,
    unsafe_allow_html=True
)

    st.image(BASE_DIR /"correlation.png")
    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    Elle met en évidence de fortes associations entre plusieurs variables cliniques.
    """,
    unsafe_allow_html=True
)
    st.markdown("""
    <div style="font-size:18px;">
    <ul>Les variables 
    <li> « Tumor Stage »</li>
    <li>« Lymph nodes examined positive »</li>
    <li>« «Neoplasm Histologic Grade»  »</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    sont très fortement corrélés avec  «Nottingham prognostic index », leur inclusion simultanée dans un modèle peut entraîner des problèmes de multicolinéarité et une instabilité des estimations.
             </p>
    """,
    unsafe_allow_html=True
)

    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    Les données de survie (Patients Vital Status/Overall Vital Status) manquantes ont été supprimées. 
             </p>
    """,
    unsafe_allow_html=True
)
    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    Les valeurs manquantes des variables numériques ont été remplacées par la médiane et les valeurs manquantes des variables catégorielles ont été remplacées par la mode.
             </p>
    """,
    unsafe_allow_html=True
)

    st.markdown("""
    <div style="font-size:18px;">
    <ul>
    <li>Features_num = ['Age at Diagnosis','Tumor Size','Tumor Stage','Lymph nodes examined positive','Neoplasm Histologic Grade','Nottingham prognostic index','Mutation Count’] </li>
    <li>features_cat = ['ER Status','PR Status','HER2 Status','Chemotherapy', 'Hormone Therapy','Radio Therapy','Type of Breast Surgery','Pam50 + Claudin-low subtype','Inferred Menopausal State’]</li>

    </ul>
    </div>
    """, unsafe_allow_html=True)

if page == pages [2] :
    st.header("Modelisation")
    st.markdown("""
    <div style="font-size:18px;">
    <ul>Deux approches de modélisation de survie ont été mises en œuvre : 
    <li>Le modèle de Cox à risques proportionnels.</li>
    <li>Le Random Survival Forest (RSF).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: justify; font-size:18px;">
    Une analyse de survie spécifique globale et au cancer a été réalisée. L'estimateur de Kaplan-Meier est couramment utilisé pour estimer cette fonction en présence de censure. Le modèle de Cox à risques proportionnels est utilisé pour estimer l'impact de plusieurs variables sur la survie. Ce modèle suppose que les risques sont proportionnels dans le temps. 
        </p>
    """, unsafe_allow_html=True)
    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    L'indice de concordance de Harrell (c-index) a été utilisé pour évaluer la performance des modèles de survie. 
             </p>
    """, unsafe_allow_html=True)

    st.header("Modelisation COX")

    st.image(BASE_DIR /"CoxB1.png")   
    
    st.markdown("""
    <div style="font-size:18px;">
    <ul>Interprétation générale
    <li>Si l’intervalle ne croise pas 0, l’effet est statistiquement significatif.</li>
    <li>À droite de 0 (log(HR) > 0) → augmente le risque (pronostic plus mauvais).</li>
    <li>À gauche de 0 (log(HR) < 0) → diminue le risque (effet protecteur).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:18px;">
    <ul>Les résultats montrent: 
    <li>Âge devient extrêmement prédictif : HR ≈ 1,94 (car mortalité non-cancer + fragilité)</li>
    <li>Taille tumorale et ganglions sont importants</li>
    <li>HER2+ reste associé au risque</li>
    <li>chimiothérapie : HR > 1 (encore reflet de sévérité)</li>
    <li>radiothérapie : HR < 1 (effet “protecteur” possible, ou marqueur d’accès aux soins).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
Des courbes de survie de Kaplan–Meier ont été estimées pour chaque groupe de risque afin de faire une validation du modèle de Cox. 

    """,
    unsafe_allow_html=True
)


    st.image(BASE_DIR /"Groupe_risqueB1.png")   

    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    La figure montre une séparation nette des courbes de survie sur l’ensemble du suivi. Les patients classés à haut risque présentent une survie significativement plus faible, tandis que les groupes à risque intermédiaire et faible montrent des trajectoires de survie cohérentes.

    """,
    unsafe_allow_html=True
)
    st.header("Modelisation RSF")
    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    Nous avons exploré un Random Survival Forest (RSF) afin de lever deux limitations majeures du Cox : L’hypothèse des risques proportionnels, partiellement violée pour certaines covariables, et la contrainte d’effets log-linéaires. 

    """,
    unsafe_allow_html=True
)
    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    Le RSF, modèle non paramétrique, permet de capturer automatiquement des non-linéarités et des interactions potentielles entre variables cliniques. 

    """,
    unsafe_allow_html=True
)

    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    La stratification des patientes - risque produit des courbes de Kaplan–Meier nettement séparées, confirmée par un test de log-rank très significatif (χ² = 87,07 ; p ≈ 1,0×10⁻²⁰). 

    """,
    unsafe_allow_html=True
)
    st.image(BASE_DIR /"Group_risqueRSF.png")   
    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
Dans RSF, interprétabilité c’est plus difficil que le Cox, car il n’y a pas de coefficients (paramétrique et non linéaire). Pour répondre à la question ‘quelles sont les variables qui contribuent aux prédictions?’ on a besoin d' utiliser permutation importance pour évaluer l’importance.

    """,
    unsafe_allow_html=True
)

    st.image(BASE_DIR /"permutanceRSF.png")   
    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    L’analyse d’importance par permutation du modèle RSF pour la survie globale met en évidence une domination marquée de l’âge au diagnostic, reflétant le fait que l’endpoint OS inclut des décès non liés au cancer. Les variables oncologiques classiques, telles que l’atteinte ganglionnaire et le Nottingham Prognostic Index, restent importantes mais jouent un rôle secondaire par rapport à l’âge. 

    """,
    unsafe_allow_html=True
)
if page == pages [3] :
    st.header("Resultats")


    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    Un GridSearch a été utilisé pour tester plusieurs pénalisations dans le modele de COX, qui rend les coefficients plus stables, améliorent souvent la performance et réduit la sensibilité aux petites variations.
    """,
    unsafe_allow_html=True
)



    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
    Pour le modele RSF, une optimisation a été egalement fait en utilisant une grille de hyperparamètres. Puis, une validation croisée à 5 fols est faite.
Ce gain est presque pareil que le résultat sans optimisation, indiquant que le modèle a atteint un plateau raisonnable.    """,
    unsafe_allow_html=True
)
    st.image(BASE_DIR /"Resultats.png")   


if page == pages[4] : 
    st.header("Partie 2 - Classification d'images histopathologiques du sein (BreakHist)")
    st.header("Introduction II")
    st.write("Le jeu de données BreaKHis:")
    st.markdown("""
    <div style="font-size:18px;">
    <ul>
    <li>7 909 images microscopiques réparties entre deux classes principales</li>
    <li>Grossissement 40x, 100x, 200x et 400x (lentille objective 4x, 10x, 20x et 40x avec lentille oculaire 10x)</li>
    <li>Format PNG
    <li>Tumeurs Bénignes et Maligne </li>
    <li>Huit sous-types histologiques </li>
    <li> Provenant respectivement de 24 et 58, soit un total de 82 patients </li>   
    </ul>
    </div>
    """, unsafe_allow_html=True)



if page == pages [5] :
    st.header("DataVisualization II")
    st.write("### Distribution des classes Béninges et Maligne")

    fig = plt.figure()
    sns.countplot(x= 'label', data= df2)
    plt.title("Distribution des classes")
    plt.xlabel("0=Bénignes, 1= Maligne")
    st.pyplot(fig)
    
    st.write("### Distribution du niveau de magnification par classes")
    fig = plt.figure()
    sns.countplot( x='magnification', hue='label', data=df2)
    st.pyplot(fig)

    st.write("### Images histopathologiques de tumeur bénigne et maligne sous différents facteurs de grossissement 40x, 100x, 200x et 400x.")
    fig, ax = plt.subplots(2, 4, figsize=(12,12))
    img1 = cv2.imread(BASE_DIR /"SOB_B_A-14-22549AB-40-029.png")
    img1 = cv2.resize(img1, (75,75))
    ax[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("Bénigne 40x")
    ax[0, 0].axis("off")

    img2 = cv2.imread(BASE_DIR /"SOB_B_F-14-29960AB-100-001.png")
    img2 = cv2.resize(img2, (75, 75))
    ax[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[0, 1].set_title("Bénigne 100x")
    ax[0, 1].axis("off")

    img3 = cv2.imread(BASE_DIR /"SOB_B_F-14-23222AB-200-009.png")
    img3 = cv2.resize(img3, (75, 75))
    ax[0, 2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    ax[0, 2].set_title("Bénigne 200x")
    ax[0, 2].axis("off")

    img4 = cv2.imread(BASE_DIR /"SOB_B_A-14-29960CD-400-015.png")
    img4 = cv2.resize(img4, (75, 75))
    ax[0, 3].imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    ax[0, 3].set_title("Bénigne 400x")
    ax[0, 3].axis("off")

    img5 = cv2.imread(BASE_DIR /"SOB_M_DC-14-10926-40-001.png")
    img5 = cv2.resize(img5, (75, 75))
    ax[1, 0].imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    ax[1, 0].set_title("Maligne 40x")
    ax[1, 0].axis("off")

    img6 = cv2.imread(BASE_DIR /"SOB_M_DC-14-14926-100-016.png")
    img6 = cv2.resize(img6, (75, 75))
    ax[1, 1].imshow(cv2.cvtColor(img6, cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title("Maligne 100x")
    ax[1, 1].axis("off")

    img7 = cv2.imread(BASE_DIR /"SOB_M_MC-14-18842-200-004.png")
    img7 = cv2.resize(img7, (75, 75))
    ax[1, 2].imshow(cv2.cvtColor(img7, cv2.COLOR_BGR2RGB))
    ax[1, 2].set_title("Maligne 200x")
    ax[1, 2].axis("off")

    img8 = cv2.imread(BASE_DIR /"SOB_M_LC-14-15570C-400-015.png")
    img8 = cv2.resize(img8, (75, 75))
    ax[1, 3].imshow(cv2.cvtColor(img8, cv2.COLOR_BGR2RGB))
    ax[1, 3].set_title("Maligne 400x")
    ax[1, 3].axis("off")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    
if page == pages[6]:
    st.header("Modélisation II")

    st.markdown("<div class='font18'>Type de probléme:</div>", unsafe_allow_html=True)
    st.badge(":orange-badge ⚠️ Classification d'images a partir de Jeux de données Désequilibrés.")
    st.badge("Modélisation avec Résaux de Neurones Convolutifs (CNN).")

    st.markdown("<div class='font30'> Modèle ResNet18</div>", unsafe_allow_html=True)


    st.markdown("<div class='font18'>Prétraitement des donnes :</div>", unsafe_allow_html=True)
    st.badge("Redimension", color="gray")
    st.badge("Normalisation", color="gray")

    st.markdown("<div class='font18'>Optimisation :</div>", unsafe_allow_html=True)
    st.badge("Analyse par grossisement d'images", icon=":material/check:", color="green")
    st.badge("fine-tunning", icon=":material/check:", color="green")
    st.badge("Selection d'images par patient", icon=":material/check:", color="gray")

if page == pages [7] :
    st.header("Resultats II")
    st.subheader("Modèle ResNET18")
    st.write("Le grossissement 200X combiné au dégel de la Layer 4 permet d'obtenir une exactitude globale de 98 %")
    st.image(BASE_DIR /"Rapport de classification_ResNet.png")   
    
    st.markdown("""
    <div style="font-size:18px;">
    <ul>Interprétation générale
    <li>Classe Bénin</li>
    <li>Rappel (0,95) → Le modèle reconnaît maintenant 114 cas bénins sur 120</li>
    <li>Précision (0,97) → Les faux positifs sont rares 6 cas sur 120 (0,05%).</li>
    <li>Classe Malin</li>          
    <li>Rappel (0,99)  → Le modèle reconnaît maintenant 280 cas bénins sur 283.</li>
    <li>F1-Score (0,98) → La capacité du modèle à classer les images est très satisfaisante. 
     <li>La matrice de confusion montre que le modèle a un taux de Faux Négatifs très faible (0,01%).
    </ul>
    </div>
   
    """, unsafe_allow_html=True)
    
    st.image(BASE_DIR /"MC_ResNet.png") 

    st.markdown(
    """
    <p style="text-align: justify; font-size:18px;">
     Grace à L’AUC de 0.99 : le modèle démontre une performance remarquable. Plus qu'une simple corrélation, il assure une distinction nette et précise entre tissus sains et malins, minimisant ainsi drastiquement les risques d'erreur de diagnostic. 
    """,
    unsafe_allow_html=True
)
    st.image(BASE_DIR /"ROC_ResNet18.png")


if page == pages [8] :
    st.header("Interpretabilité")
    st.write("### Grad-CAM image originale Maligne vs Grad-CAM ResNet18")
    st.image(BASE_DIR /"GRAD-CAM_ResNet18.png")
    st.markdown("""
    <div style="font-size:18px;">
    <ul> • Le Grad-CAM utilise les gradients de la dernière couche convolutive pour produire une carte thermique.<ul>
    <ul> •  Pour la classe Maligne, on remarque que le modèle s'active sur les zones de prolifération anarchique. Cela valide que notre modele n'apprend pas par hasard, mais se base sur des caractéristiques biologiques réelles.<ul>
    <ul> •  À l'inverse, pour la classe Bénigne, il identifie des structures organisées.<ul> 
    <ul> •  Le ResNet18 a réussi à extraire des biomarqueurs visuels cohérents avec la littérature médicale.<ul>
         </div> 
     
    """, unsafe_allow_html=True)

if page == pages [9] :
    st.header("Conclusions")
    st.markdown("""
        <div class='font18'>
        Ce projet avait pour objectif général d'explorer l'intelligence artificielle afin d'améliorer à la fois le diagnostic histopathologique et le pronostic du cancer du sein , dans le but de mieux aider les professionnels de santé. </div>
    """, unsafe_allow_html=True)

    
    st.subheader("Conclusions – Analyse METABRIC")

    st.markdown("""
        <div class='font18'>
        L’analyse de la base <b>METABRIC</b> a permis de comparer les performances du 
        <b>modèle de Cox</b> et du <b>Random Survival Forest (RSF)</b> pour prédire la survie 
        des patientes atteintes de cancer du sein à partir de données cliniques et moléculaires.
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Principaux résultats")

    st.markdown("""
        <div class='font20'>
        • Les modèles de <b>Cox</b> montrent une bonne discrimination (C-index ≈ 0.677), 
        améliorée par l’ajout de variables cliniques et thérapeutiques.<br><br>
        
        • La variable cible: la survie globale (Overall Survival, OS) est un indicateur clair,<br>
           <b> robuste, cliniquement pertinent et cohérent avec une prédiction centrée patient.<br>

        • Les facteurs pronostiques majeurs sont : <b>âge</b>, <b>taille tumorale</b>, 
        <b>atteinte ganglionnaire</b>, <b>grade</b>, <b>HER2+</b> et <b>chimiothérapie</b>.<br>
                
        •Les facteurs protecteurs incluent : <b>récepteurs hormonaux positifs</b>, 
        <b>hormonothérapie</b> et <b>radiothérapie</b>.<br>

        • Les <b>Random Survival Forest</b> capturent des non‑linéarités et améliorent légèrement 
        certaines performances. 
    

        • Globalement, le <b>modèle de Cox</b> reste un excellent compromis entre 
        <b>performance</b>, <b>stabilité</b> et <b>applicabilité clinique</b> pour la stratification du risque.
        </div>
    """, unsafe_allow_html=True)

if page == pages[10]:   

    st.subheader("Conclusions – Analyse BreakHist ")
    
    st.markdown("""
        <div class='font18'>
        Cette étude visait à évaluer la capacité d'un modèle ResNet18 à classifier les images histopathologiques 
        des tumeurs mammaires Bénigne vs Maligne. 
        </div>
    """, unsafe_allow_html=True)
    st.subheader("Principaux résultats")
    st.markdown("""
        <div class='font18'>
        •   <b>L’analyse par chaque groupe de grossissement (200x) et Combinaison avec le fine-tuning (dégel de couche 4) 
            constitue le niveau le plus performant pour cette architecture.
                  
        •   Le modèle atteint <b>≈ 98 % d’exactitude</b>, différenciant de manière fiable les tissus 
            bénins et malins. 
                
        •   Cette résolution offre un équilibre idéal entre richesse des détails 
            cellulaires et stabilité de l’apprentissage.
        </div>
    """, unsafe_allow_html=True)

if page == pages [11] :
    st.header("Limitations")   
    st.markdown("""
        <div class='font18'>
        • Malgré les bons résultats obtenus avec les deux ensembles de données, les résultats
            METABRIC et Breakhist sont très complexes et nécessitent une expertise médicale pour
            être correctement interprétés et validés  
                
        •   En ce qui concerne le jeu des données de METABRIC : les données manquantes, Multicolinéarité / redondance entre variables tumorales.
                
        •   Pour le dataset BreakHis : les différences entre les résultats selon le type de grossissement , la persistance de faux négatifs. 
                 
        </div>
    """, unsafe_allow_html=True)

    st.write("### Les modèles proposés doivent impérativement être utilisés par du personnel médical qualifié et constituent un outil pour améliorer à la fois le Diagnostic et le Pronostic du cancer du sein.")

    st.write("### La validation humaine est la garantie finale  contre les erreurs résiduelles de chaque modèle.")























