import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    artifacts = joblib.load("student_performance_model.pkl")
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]
    return model, feature_columns

model, feature_columns = load_model()

# ---------------------------
# Feature Engineering
# ---------------------------
def add_engineered_features(df):
    score_cols = ['math_score', 'reading_score', 'writing_score']
    df = df.copy()

    # Average and total
    df['average'] = df[score_cols].mean(axis=1)
    df['total'] = df[score_cols].sum(axis=1)

    # Letter grade
    def to_grade(avg):
        if avg >= 90: return 'A'
        if avg >= 80: return 'B'
        if avg >= 60: return 'C'
        if avg >= 45: return 'D'
        if avg >= 33: return 'E'
        return 'F'

    df['grade'] = df['average'].apply(to_grade)
    return df

# ---------------------------
# App Layout
# ---------------------------

st.markdown("<h1 style='text-align: center; color: darkblue;'>🎓 Student Performance Prediction App</h1>", unsafe_allow_html=True)

# Small profile picture below the heading
st.image("profile.jpeg", width=150, caption="Author: Aaryan Kumar")

st.markdown("""
<div style="text-align: left; padding: 15px; font-size:16px; color: #ffffff; line-height: 2; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    <h2 style="background: linear-gradient(90deg, #36BCF7, #34eb7f); -webkit-background-clip: text; color: transparent; font-size:32px; margin-bottom: 15px;">
        👋 Welcome to the Student Performance Predictor!
    </h2>
    <ul style="list-style-type: none; padding-left: 0;">
        <li style="margin-bottom: 10px;">✨ <b>Enter the student details</b> in the sidebar and get instant insights into academic performance.</li>  
        <li style="margin-bottom: 10px;">🎯 Predict whether a student will <b style='color:#ff5733'>Pass or Fail</b> with probability scores.</li>  
        <li style="margin-bottom: 10px;">📊 Discover which <b style='color:#1f4e79'>Grade</b> you currently stand at and visualize your progress.</li>  
        <li style="margin-bottom: 10px;">🚀 All results are displayed with <b>interactive charts, progress bars, and dynamic grade cards</b>.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("📝 Enter Student Details")

gender = st.sidebar.selectbox("Gender", ["male", "female"])
race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_education = st.sidebar.selectbox("Parental Education", [
    "some high school", "high school", "some college", 
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])

math_score = st.sidebar.slider("Math Score", 0, 100, 50)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 50)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 50)

# ---------------------------
# Create DataFrame from Inputs
# ---------------------------
user_data = pd.DataFrame({
    'gender': [gender],
    'race_ethnicity': [race_ethnicity],
    'parental_education': [parental_education],
    'lunch': [lunch],
    'test_prep': [test_prep],
    'math_score': [math_score],
    'reading_score': [reading_score],
    'writing_score': [writing_score]
})

user_data = add_engineered_features(user_data)

# Encode categorical variables
user_data_encoded = pd.get_dummies(user_data, drop_first=True)
user_data_encoded = user_data_encoded.reindex(columns=feature_columns, fill_value=0)

# ---------------------------
# Session State for all entries
# ---------------------------
if "all_students" not in st.session_state:
    st.session_state.all_students = pd.DataFrame()

# ---------------------------
# Prediction Button
# ---------------------------
if st.sidebar.button("🚀 Predict Result"):
    with st.spinner("Predicting..."):
        prediction = model.predict(user_data_encoded)[0]
        prediction_proba = model.predict_proba(user_data_encoded)[0]

    # ---------------------------
    # Display Results
    # ---------------------------
    st.subheader("🔮 Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("Prediction", prediction)
    col2.metric("Pass Probability", f"{prediction_proba[1]*100:.2f}%")

    st.markdown("**📊 Probabilities**")
    colA, colB = st.columns(2)
    colA.progress(float(prediction_proba[1]))
    colA.write(f"Pass: {prediction_proba[1]*100:.2f}%")

    colB.progress(float(prediction_proba[0]))
    colB.write(f"Fail: {prediction_proba[0]*100:.2f}%")

    if prediction == "Pass":
        st.success("✅ The student is likely to Pass")
    else:
        st.error("❌ The student is likely to Fail")

    # ---------------------------
    # 🎯 Attractive Grade Card (Option 1)
    # ---------------------------
    student_grade = user_data['grade'].iloc[0]
    grade_colors = {"A": "green", "B": "blue", "C": "orange", "D": "brown", "E": "purple", "F": "red"}
    grade_color = grade_colors.get(student_grade, "black")

    st.markdown(
        f"""
        <div style="
            background-color:#f8f9fa;
            padding:20px;
            border-radius:15px;
            text-align:center;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
            font-size:28px;
            font-weight:bold;
            color:{grade_color};
        ">
            🎓 Predicted Grade: {student_grade}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------
    # Append result to session data
    # ---------------------------
    user_data_with_result = user_data.copy()
    user_data_with_result['Prediction'] = prediction
    user_data_with_result['Fail %'] = prediction_proba[0]*100
    user_data_with_result['Pass %'] = prediction_proba[1]*100

    st.session_state.all_students = pd.concat(
        [st.session_state.all_students, user_data_with_result],
        ignore_index=True
    )

# ---------------------------
# Summary Section
# ---------------------------
if not st.session_state.all_students.empty:
    with st.expander("📊 Summary of All Students Entered", expanded=True):

        # Coloring Prediction & Grade (Option 2)
        def prediction_color(val):
            return f'background-color: {"lightgreen" if val=="Pass" else "salmon"}'

        def grade_color(val):
            colors = {"A": "green", "B": "blue", "C": "orange", "D": "brown", "E": "purple", "F": "red"}
            return f'color: {colors.get(val, "black")}; font-weight: bold; text-align:center;'

        styled_df = st.session_state.all_students.style.map(prediction_color, subset=['Prediction'])
        styled_df = styled_df.map(grade_color, subset=['grade'])

        st.dataframe(styled_df, use_container_width=True)

    st.subheader("📈 Visual Insights")
    st.bar_chart(st.session_state.all_students[['math_score', 'reading_score', 'writing_score']])
    
    prediction_counts = st.session_state.all_students['Prediction'].value_counts().reset_index()
    prediction_counts.columns = ['Prediction', 'Count']

    fig = px.pie(prediction_counts, names='Prediction', values='Count', title="Prediction Distribution")
    st.plotly_chart(fig)
