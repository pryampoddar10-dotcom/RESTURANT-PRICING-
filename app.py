
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

st.set_page_config(page_title="AI Restaurant Decision System", layout="wide")

st.title("🍽️ AI Restaurant Decision Intelligence System")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully")

    # Encoding
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview","Customer Insights","Predictive Models","Menu Intelligence","Strategy Engine","Upload New Data"
    ])

    with tab1:
        st.subheader("Demand Overview")
        fig = px.histogram(df, x="time_slot")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Customer Segmentation")
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_encoded['cluster'] = kmeans.fit_predict(df_encoded)
        fig2 = px.scatter(df_encoded, x="avg_spend_customer", y="visit_frequency", color="cluster")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Classification Model")

        X = df_encoded.drop(columns=["discount_purchase_intent"])
        y = df_encoded["discount_purchase_intent"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:,1] if len(np.unique(y))==2 else None

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
        st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
        st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = px.line(x=fpr, y=tpr, title="ROC Curve")
            st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": clf.coef_[0]
        }).sort_values(by="importance", ascending=False)

        fig_imp = px.bar(importance, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("Regression Model")
        X_reg = df_encoded.drop(columns=["total_orders_hour"])
        y_reg = df_encoded["total_orders_hour"]

        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        reg = LinearRegression()
        reg.fit(X_train, y_train)

        st.write("Regression model trained")

    with tab4:
        st.subheader("Association Rules")
        basket = df['items_ordered'].str.get_dummies(sep=',')
        frequent = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent, metric="confidence", min_threshold=0.5)

        st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

    with tab5:
        st.subheader("Prescriptive Strategy Engine")

        st.write("Recommended Actions:")

        high_demand = df[df['demand_level']=="High"]
        st.write("👉 Increase prices during peak hours")

        discount_users = df[df['price_sensitivity_score']>=4]
        st.write("👉 Offer discounts to high price-sensitive users")

        st.write("👉 Promote combo meals for higher revenue")

    with tab6:
        st.subheader("Upload New Customers")

        new_file = st.file_uploader("Upload new data", type=["csv"], key="new")

        if new_file:
            new_df = pd.read_csv(new_file)
            new_df_enc = new_df.copy()

            for col in new_df_enc.select_dtypes(include=['object']).columns:
                new_df_enc[col] = le.fit_transform(new_df_enc[col])

            preds = clf.predict(new_df_enc)
            new_df["Predicted Purchase Intent"] = preds

            st.dataframe(new_df)
