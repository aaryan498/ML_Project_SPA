<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=shark&color=gradient&height=100&section=header&fontSize=40&animation=twinkling&fontColor=000000" alt="banner" />
</p>

<h1 align="center">🎓 Student Performance Predictor 🎓</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-ff69b4.svg" />
  <img src="https://img.shields.io/badge/ML-LogisticRegression-green.svg" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen.svg" />
  <img src="https://img.shields.io/github/license/aaryan498/Student-Performance-Predictor?color=orange" />
  <img src="https://img.shields.io/github/stars/aaryan498/Student-Performance-Predictor?style=social" />
  <img src="https://img.shields.io/github/forks/aaryan498/Student-Performance-Predictor?style=social" />
</p>

---

## 📑 Table of Contents  

<p align="center">
  <a href="#-profile-stats">
    <img src="https://img.shields.io/badge/Profile%20Stats-📊-blue?style=for-the-badge" />
  </a>  
  <br/>
  <a href="#-project-overview">
    <img src="https://img.shields.io/badge/Project%20Overview-📖-green?style=for-the-badge" />
  </a>  
  <br/>
  <a href="#-file-structure">
    <img src="https://img.shields.io/badge/File%20Structure-📂-orange?style=for-the-badge" />
  </a>  
  <br/>
  <a href="#-how-to-run-the-project">
    <img src="https://img.shields.io/badge/How%20to%20Run-🚀-red?style=for-the-badge" />
  </a>  
  <br/>
  <a href="#-features">
    <img src="https://img.shields.io/badge/Features-🎯-purple?style=for-the-badge" />
  </a>  
  <br/>
  <a href="#-project-showcase">
    <img src="https://img.shields.io/badge/Project%20Showcase-📸-brightgreen?style=for-the-badge" />
  </a>  
  <br/>
  <a href="#-tech-stack">
    <img src="https://img.shields.io/badge/Tech%20Stack-📊-teal?style=for-the-badge" />
  </a>  
  <br/>
  <a href="#-internship-details">
    <img src="https://img.shields.io/badge/Internship%20Details-🏅-yellow?style=for-the-badge" />
  </a>  
  <br/>
  <a href="#-author">
    <img src="https://img.shields.io/badge/Author-👨‍💻-black?style=for-the-badge" />
  </a>  
</p>

---

## 📊 Profile Stats

<p align="center">
  <a href="https://github.com/aaryan498">
    <img src="https://komarev.com/ghpvc/?username=aaryan498&style=flat-square&color=blue" alt="Profile Views"/>
  </a>
  <a href="https://github.com/aaryan498?tab=followers">
    <img src="https://img.shields.io/github/followers/aaryan498?label=Followers&style=social" alt="GitHub Followers"/>
  </a>
  <a href="https://github.com/aaryan498">
    <img src="https://img.shields.io/github/stars/aaryan498?label=Stars&style=social" alt="GitHub Stars"/>
  </a>
</p>

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=aaryan498&show_icons=true&theme=radical" alt="GitHub Stats" height="150"/>
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=aaryan498&layout=compact&theme=radical" alt="Top Languages" height="150"/>
</p>

---

✨ *An ML-powered app to predict student performance based on input data, deployed with Streamlit for interactive use.*


---

# 📖 Project Overview  

This project is a **Student Performance Analysis System** designed to evaluate and predict student outcomes.  

✨ **What it does:**  
- 📊 Predicts a student’s **academic performance**.  
- 📝 Takes into account **test scores, parental background, and study habits**.  
- 🎯 Provides **grades and performance insights** in a user-friendly way.  
- 💡 Helps identify whether a student is likely to be a **top performer or needs improvement**.  

🔑 **Key Highlights:**  
- Built using **Python, Scikit-learn, Pandas, Matplotlib, Seaborn, and Streamlit**.  
- Implements **data preprocessing, exploratory data analysis (EDA), and machine learning modeling**.  
- Provides a **Streamlit web interface** where users can enter student details and get predictions instantly.  
- Displays **summary tables, grade classification, and insights** in a clear and visually appealing way.  
- Stores multiple student records during runtime for tracking and comparison.  

---

## 📂 File Structure
```
ML_Project_SPA/
├── predict.py                        # Data cleaning, EDA, model training & CLI
├── app.py                            # Streamlit web app (interactive UI)
├── README.md                         # Project documentation (this file)
├── requirements.txt                  # Required dependencies
├── student_performance_model.pkl     # Saved model + feature_columns (joblib artifact)
└── StudentsPerformance.csv           # Original dataset (Kaggle)
```

---

## 🚀 How to Run the Project

Follow these steps to set up and run the **Student Performance Analysis System** on your local machine.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/aaryan498/ML_Project_SPA.git
```
### 2️⃣ Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```
### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
### 4️⃣ Run Data Preprocessing & Model Training  

This step will clean the dataset, perform EDA, and train the model.  
It will also generate the **student_performance_model.pkl** file (if not already present).  

```bash
python predict.py
```
### 5️⃣ Launch the Streamlit Web App  

Run the app to get an interactive interface for student performance prediction.  

```bash
streamlit run app.py
```
### 6️⃣ Access the App  

Once Streamlit starts, it will show a local URL such as:  

```bash
# This is just an example not the actual link.
Local URL: http://localhost:8501
```
👉 Open this link in your browser to use the app.

### 7️⃣ Exit  

When you’re done, press `CTRL + C` in the terminal to stop the server.  
Deactivate the virtual environment:  

```bash
deactivate
```
⚡ That’s it! Your Student Performance Predictor is now up and running 🎉

---

## 🎯 Features  

✨ **Core Functionality**  
- ✅ Predicts student performance as **Pass / Fail** with probability scores.  
- ✅ Assigns **letter grades (A–F)** based on average scores.  
- ✅ Saves and displays a **summary of all students entered** in the session.  

📊 **Insights & Visualization**  
- 📈 Displays **performance probabilities** with progress bars.  
- 🃏 Attractive **grade cards** with dynamic color coding (A=Green, B=Blue, …, F=Red).  
- 📊 Interactive **summary tables** with conditional styling (green for Pass, red for Fail).  
- 🍰 Built-in **visual insights**:    

---

## 📸 Project Showcase  

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=28&pause=1000&color=36BCF7&center=true&vCenter=true&width=650&lines=✨+Have+a+look+at+my+App+Screenshots+✨" alt="Typing SVG" />
</p>

---

<p align="center">
  <!-- Fancy GIF Carousel effect -->
  <img src="screenshot-carousel.gif" alt="Project Carousel Demo" width="80%" style="border-radius:15px; box-shadow: 0 0 15px rgba(0,0,0,0.3);" />
</p>

---

### 🔗 Live Demo  
[![Live Demo](https://img.shields.io/badge/Click%20Here-Live%20Demo-brightgreen?style=for-the-badge&logo=google-chrome)](https://your-demo-link.com)

---

## 📊 Tech Stack  

### 🖥️ Programming Language  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  

### 📚 Libraries & Frameworks  
**Data Handling & Analysis**  
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)  

**Machine Learning**  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![Joblib](https://img.shields.io/badge/Joblib-2E7D32?style=for-the-badge&logo=python&logoColor=white)  

**Exploratory Data Analysis & Visualization**  
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-009688?style=for-the-badge&logo=python&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)  

**Web App Deployment**  
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)  
  

---

## 🏅 Internship Details  

This project was developed as part of my **College Internship Project** at:  

🎓 **Ajay Kumar Garg Engineering College, Ghaziabad, Uttar Pradesh**  

### 📌 Internship Highlights  
- 📅 **Duration:** 10 Days  
- 💻 **Domain:** Python Programming & Machine Learning  
- 📝 **Research Work:** Writing a research paper on this project is also an integral part of the internship  

---

## 👨‍💻 Author  

<p align="center">
  <img src="https://avatars.githubusercontent.com/u/106829221?v=4" width="120" style="border-radius:50%" alt="Author Profile Picture"/>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=shark&color=gradient&height=100&section=header&text=Aaryan%20Kumar&fontSize=40&animation=twinkling&fontColor=000000" alt="banner" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&pause=1000&color=00FFEE&center=true&vCenter=true&width=600&lines=🚀+Aspiring+SDE;🤖+AI%2FML+Enthusiast;🌐+Tech+Explorer" alt="Typing SVG" />
</p>

<p align="center">
  <a href="https://github.com/aaryan498">
    <img src="https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github"/>
  </a>
  <a href="https://www.linkedin.com/in/aaryan-kumar-ai-498-coder/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin"/>
  </a>
  <a href="mailto:aaryankumarofficial498@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact-red?style=for-the-badge&logo=gmail"/>
  </a>
</p>

<p align="center">
  ⭐ If you like this project, don't forget to star the repo! ⭐
</p>





