import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#python -m pip install -r requirements.txt
#run this command in terminal to install lib we need to this code


df = None
file_name = ""
model_choise_AI = ""
issupervise = ""
type_of_task = ""
type_Oftask_ai = ""
suggested_model = ""
potential_targets = []
list_of_unique_ratio = []
num_classes = 0
num_numeric = 0
num_categorical = 0
non_linear = False
target = None
outlier_report = {}





with st.sidebar:
    st.title("Auto AI system")
    st.image("loggo.png")
    Data = st.file_uploader("Upload your data")
    if Data:
        st.success("The Data Uploaded")
    choise=st.selectbox("service",["  ","Data Analysis","Auto AI system","the evaluation visualization"])




    



if Data:
    file_name = Data.name
    if file_name.endswith(".csv"):
        df = pd.read_csv(Data)
        
    elif file_name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(Data)
       
    elif file_name.endswith(".json"):
        df = pd.read_json(Data)
       
    elif file_name.endswith(".txt"):
        content = Data.read().decode("utf-8")
       

    if df is not None:
        num_numeric = df.select_dtypes(include='number').shape[1]
        num_categorical = df.select_dtypes(include='object').shape[1]


if choise == "Data Analysis":
    st.title("The Head Of Data")
    st.dataframe(df.head(5))

    st.title("Data Overview Table (Full Summary)")

    summary_data = []

    
    num_cols = df.select_dtypes(include="number").columns
    outlier_threshold = 0.05  
    missing_threshold = 0.05  

   
    outlier_report = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        outlier_ratio = len(outliers) / len(df[col].dropna())
        outlier_report[col] = outlier_ratio > outlier_threshold 

    for col in df.columns:
        col_type = df[col].dtype
        missing_count = df[col].isna().sum()
        missing_ratio = df[col].isna().mean()
        unique_count = df[col].nunique()
        top_value = df[col].mode()[0] if unique_count > 0 else None
        top_freq = df[col].value_counts().iloc[0] if unique_count > 0 else None
        mean_val = df[col].mean() if col_type in ["int64", "float64"] else None
        std_val = df[col].std() if col_type in ["int64", "float64"] else None
        min_val = df[col].min() if col_type in ["int64", "float64"] else None
        max_val = df[col].max() if col_type in ["int64", "float64"] else None

     
        has_outliers = outlier_report.get(col, False)
        has_missing = missing_ratio > missing_threshold

        summary_data.append({
            "Column": col,
            "Type": col_type,
            "Missing Count": missing_count,
            "Missing %": round(missing_ratio*100, 2),
            "Unique Values": unique_count,
            "Top Value": top_value,
            "Top Freq": top_freq,
            "Mean": mean_val,
            "Std": std_val,
            "Min": min_val,
            "Max": max_val,
            "Has Outliers": has_outliers,
            "Has Significant Missing": has_missing
        })

    summary_df = pd.DataFrame(summary_data)

  
    def highlight_issues(val, col_name):
        if col_name in ["Has Significant Missing", "Missing Count", "Missing %"] and val:
            return "background-color: red; color: white"
        if col_name == "Has Outliers" and val == True:
            return "background-color: red; color: white"
        return ""

 
    styled_df = summary_df.style.applymap(lambda val: highlight_issues(val, "Has Outliers"), subset=["Has Outliers"]) \
                                .applymap(lambda val: highlight_issues(val, "Has Significant Missing"), subset=["Has Significant Missing"])

    st.dataframe(styled_df)


    
   
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        outliers_sum = outliers[col].sum()
        if not outliers.empty:
            outlier_report[col] = len(outliers)

    st.title("Outlier")
    if not outlier_report:
        st.success("No significant outliers detected ")
    else:
        st.dataframe(pd.DataFrame.from_dict(outlier_report, orient="index", columns=["Outlier Count"]))

  
    
 
    st.title("Categorical Imbalance")
    cat_cols = df.select_dtypes(include="object").columns

    if len(cat_cols) == 0:
        st.info("No categorical columns detected")
    else:
        for col in cat_cols:
            counts = df[col].value_counts(normalize=True)
            max_ratio = counts.max()

            st.write(f" {col}")
            st.dataframe(counts)

           
            st.bar_chart(df[col].value_counts())

            if max_ratio > 0.75:
                st.error("Imbalanced Column")
                st.warning(" Recommendation: Use SMOTE, class weights, or undersampling")
            else:
                st.success("Balanced ")


    st.title("Numeric Columns Distribution (Boxplot & Histogram)")

    for col in num_cols:
      with st.expander(f"{col}"):
       
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].boxplot(df[col].dropna())
        axes[0].set_title(f"Boxplot of {col}")

        axes[1].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        axes[1].set_title(f"Histogram of {col}")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()



 
 

    if len(num_cols) > 1:
        st.title("Correlation Heatmap")
        corr = df[num_cols].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(plt)
        plt.close()



    st.title("Suggested Target Column")

    potential_targets = []
    list_of_unique_ratio = []


    for col in df.columns:
      nunique_ratio = df[col].nunique() / df.shape[0]
      if nunique_ratio < 0.1 and df[col].dtype != "object":
          potential_targets.append(col)
          list_of_unique_ratio.append(nunique_ratio)

    if potential_targets:
     target_index = list_of_unique_ratio.index(min(list_of_unique_ratio))
     target = potential_targets[target_index]
     st.write(f"Suggested target column: {target}")

    
     missing_ratio = df[target].isna().mean()
     issupervise = "Supervised Learning" if missing_ratio < 0.25 else "UnSupervised Learning"
     st.write(f"Learning type: {issupervise}")

    
     st.subheader(f"Distribution of Target: {target}")
     if df[target].dtype == "object":
        
         st.bar_chart(df[target].value_counts())
     else:
        
        plt.figure(figsize=(6,3))
        plt.hist(df[target].dropna(), bins=30, color='lightgreen', edgecolor='black')
        plt.title(f"Histogram of {target}")
        st.pyplot(plt)
        plt.close()
    else:
      st.info("No suitable target column found automatically")


if choise == "Auto AI system":
    model_choise = st.selectbox(
        "Choose your AI model", [" ", "Machine", "Deep learning", "NLP", "let the AI choise"]
    )

    if model_choise == "let the AI choise" and df is not None:
        if file_name.endswith(".csv") or df.shape[0] < 100000:
            model_choise_AI = "ML"
            st.write("The AI recommends Machine Learning (ML)")

    if (model_choise == "Machine" or model_choise_AI == "ML") and df is not None:

        superviseML = st.selectbox(
            "Is the data supervised?", [" ", "supervise", "unsupervise", "let the AI choise"]
        )

        if superviseML in ["supervise", "unsupervise"]:
            issupervise = superviseML
            if issupervise == "supervise":
                target = st.selectbox("Choose your label column", df.columns)
                num_classes = df[target].nunique()

        elif superviseML == "let the AI choise":
            potential_targets = []
            list_of_unique_ratio = []

            for col in df.columns:
                nunique_ratio = df[col].nunique() / df.shape[0]
                if nunique_ratio < 0.1 and df[col].dtype != "object":
                    potential_targets.append(col)
                    list_of_unique_ratio.append(nunique_ratio)

            if potential_targets:
                target_index = list_of_unique_ratio.index(min(list_of_unique_ratio))
                target = potential_targets[target_index]
                st.write(f"Suggested label column: {target}")
                num_classes = df[target].nunique()
                missing_ratio = df[target].isna().mean()
                if missing_ratio < 0.25:
                    issupervise = "Supervised Learning"
                else:
                    issupervise = "UnSupervised Learning"
            else:
                issupervise = "UnSupervised Learning"

        st.write(f"Learning type: {issupervise}")

        if issupervise:
            type_of_task = st.selectbox(
                "Choose your type of task", ["", "let the AI choise", "Classification", "Regression"]
            )

            if type_of_task == "let the AI choise" and target is not None:
                if num_classes <= 15 and df[target].dtype == "object":
                    type_Oftask_ai = "Classification"
                elif num_classes <= 10 and df[target].dtype in ["int64", "float64"]:
                    type_Oftask_ai = "Classification"
                else:
                    type_Oftask_ai = "Regression"

            if type_of_task in ["Classification"] or type_Oftask_ai == "Classification":

                classification_model_name = st.selectbox(
                    "Choose your classification model",
                    [
                        "", "Logistic Regression", "SVM", "Decision Tree",
                        "Random Forest", "Gradient Boosting", "LightGBM",
                        "all them", "let AI choise"
                    ]
                )

                if classification_model_name == "let AI choise":

                    non_linear = True if num_numeric > 1 else False

                    if df.shape[0] < 10000:
                        if num_classes <= 2:
                            if num_categorical == 0:
                                suggested_model = "Logistic Regression"
                            elif non_linear:
                                suggested_model = "SVM"
                            else:
                                suggested_model = "Decision Tree"
                        elif num_classes <= 5:
                            if num_categorical > 0 and num_numeric > 0:
                                suggested_model = "Decision Tree"
                            else:
                                suggested_model = "SVM"
                        else:
                            suggested_model = "Decision Tree"

                    elif df.shape[0] < 500000:
                        suggested_model = "Random Forest" if non_linear else "Gradient Boosting"
                    else:
                        suggested_model = "LightGBM"

                    st.write(f"Suggested model: {suggested_model}")

  









