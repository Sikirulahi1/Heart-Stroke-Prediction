import streamlit as st
import pickle
import numpy as np


feature_name = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']

# Load the files

with open('Json_files/min_max_dict.json', 'rb') as fp:
    min_max_dict = pickle.load(fp)

with open('Json_files/cat_dict.json', 'rb') as fp:
    cat_dict = pickle.load(fp)


path = "Analysis_and_model\model.pkl"
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_value(feature_name, val, my_dict = cat_dict):
    feature_dict = my_dict[feature_name]
    for key, value in feature_dict.items():
        if val == key:
            return value

def scale_value(feature_name, value, dic = min_max_dict):
    maximum = dic[feature_name]['max']
    minimum = dic[feature_name]['min']
    scaled_value = (value - minimum)/ (maximum - minimum)
    return scaled_value

def get_feature_dic(feature_name,dic = cat_dict):
    return dic[feature_name]


def main():
    # Heart Stroke Prediction Application
    st.title("Heart Stroke Prediction")
    activities = ["Home", "Predictions", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    st.sidebar.markdown(
            """ Developed by Sikirulahi  
                Email me @ : kareemsikiru@gmail.com
                """)

    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#4073FF;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Definition : According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths..</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
    
        st.write("""The main functionality of this application is to predict whether a patient is likely to get stroke.
             """)
        
        
        
    elif choice == "Predictions":
        st.subheader("Heart Stroke Prediction")
        
        age_class = st.number_input("Enter your age")
        age_scaled = scale_value("age", age_class)
        
        gender_class = st.selectbox("Select your gender", tuple(get_feature_dic("gender").keys()))
        gender_code = get_value("gender", gender_class)
        gender_scaled = scale_value("gender", gender_code)
        
        ever_married_class = st.radio("Have you ever married ?", tuple(get_feature_dic("ever_married").keys()))
        ever_married_code = get_value("ever_married", ever_married_class)
        ever_married_scaled = scale_value("ever_married", ever_married_code)
        
        
        work_type_class = st.selectbox("Whats your work class ?", tuple(get_feature_dic("work_type").keys()))
        work_type_code = get_value("work_type", work_type_class)
        work_type_scaled = scale_value("work_type", work_type_code)
        
        
        Residence_type_class = st.selectbox("What's your residence type ?", tuple(get_feature_dic("Residence_type").keys()))
        Residence_type_code = get_value("Residence_type", work_type_class)
        Residence_type_scaled = scale_value("Residence_type", work_type_code)
        
        avg_glucose_level_class = st.number_input("Enter your average glucose level:")
        avg_glucose_level_scaled = scale_value("avg_glucose_level", avg_glucose_level_class)
        
        bmi_class = st.number_input("Enter your body max index:")
        bmi_scaled = scale_value("bmi", bmi_class)
        
        smoking_status_class = st.selectbox("Whats your smoking status ?", tuple(get_feature_dic("smoking_status").keys()))
        smoking_status_code = get_value("smoking_status", smoking_status_class)
        smoking_status_scaled = scale_value("smoking_status", smoking_status_code)
        
        
        hypertension_class = st.radio("Do you have any hypertension ?", tuple(get_feature_dic("hypertension").keys()))
        hypertension_code = get_value("hypertension", ever_married_class)
        hypertension_scaled = scale_value("hypertension", ever_married_code)
        
        heart_disease_class = st.radio("Do you have any heart diseases ?", tuple(get_feature_dic("heart_disease").keys()))
        heart_disease_code = get_value("heart_disease", ever_married_class)
        heart_disease_scaled = scale_value("heart_disease", ever_married_code)
        
        
        
        feature_values = [gender_scaled, age_scaled, ever_married_scaled,
        work_type_scaled, Residence_type_scaled, avg_glucose_level_scaled, bmi_scaled, smoking_status_scaled, hypertension_scaled, heart_disease_scaled]
        
        
        Result = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']
        single_sample = np.array(feature_values).reshape(1,-1)
        
        
        
        if st.button("Predict"):
            preds = load_model(path).predict(single_sample)
              
            if preds == 0:
                st.write("No, Not likely")
                # st.image("correct.jpg")
                      
            elif preds == 1:
                st.write("Yes, likely")
                # st.image("medical.jpg")
                    
            else:
                st.write("Invalid")
        
        
        
        
        
        
        
        
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#4073FF;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Definition : </h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                                     <div style="background-color:#4073FF;padding:10px">
                                     <h4 style="color:white;text-align:center;">This application is developed by Sikirulahi for the purpose of predicting whether a patient is likely to get stroke or not</h4>
                                     <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                     </div>
                                     <br></br>
                                     <br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)
    



if __name__ == "__main__":
    main()