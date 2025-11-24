import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Air Quality Assessment",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: inherit;
        text-align: center;
        margin-bottom: 0;
        background: -webkit-linear-gradient(120deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 500;
        color: inherit;
        opacity: 0.8;
        margin-top: -5px;
        margin-bottom: 0;
    }
    .author-text {
        text-align: center;
        font-size: 0.9rem;
        font-weight: 600;
        color: inherit;
        opacity: 0.6;
        margin-top: 5px;
        margin-bottom: 40px;
        font-style: italic;
        letter-spacing: 1px;
    }
    .workflow-step {
        background: linear-gradient(135deg, #11998e 0%, #0f2027 100%);
        border-left: 5px solid #38ef7d;
        padding: 25px;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        height: 100%;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .workflow-step:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(56, 239, 125, 0.2);
        border-left: 5px solid #ffffff;
    }
    .workflow-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .workflow-title {
        font-weight: 800;
        font-size: 1.2rem;
        margin-bottom: 5px;
        color: #ffffff;
        letter-spacing: 0.5px;
    }
    .workflow-desc {
        font-size: 0.9rem;
        color: #e0e0e0;
        font-weight: 400;
    }
    .metric-container {
        text-align: center;
        padding: 10px;
    }
    .metric-value {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1;
        filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.1));
    }
    .metric-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 10px;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .cta-box {
        background: linear-gradient(90deg, rgba(17, 153, 142, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%);
        border: 1px solid rgba(17, 153, 142, 0.2);
        color: inherit;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 25px 0;
        font-weight: 500;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return scaler, model, True
    except Exception:
        return None, None, False

@st.cache_data
def load_dataset():
    try:
        if not os.path.exists("airquality"):
            if os.path.exists("airquality.zip"):
                with zipfile.ZipFile("airquality.zip", 'r') as zip_ref:
                    zip_ref.extractall("airquality")
        
        df = pd.read_csv("airquality/AirQualityUCI.csv", sep=";", decimal=",")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna()
        
        target_cols = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
        
        def remove_outliers_iqr(d, columns):
            df_c = d.copy()
            for col in columns:
                Q1 = df_c[col].quantile(0.25)
                Q3 = df_c[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_c = df_c[(df_c[col] >= lower) & (df_c[col] <= upper)]
            return df_c

        df_final = remove_outliers_iqr(df, target_cols)
        df_final = remove_outliers_iqr(df_final, target_cols)
        
        return df_final, True
    except:
        return None, False

def main():
    st.markdown('<div class="main-title">Benzene Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-Sensor Air Quality Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="author-text">Inferential Statistics Project - INT-24</div>', unsafe_allow_html=True)
    
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Select Module:", ["🏠 Home", "🧪 Concentration Predictor", "📊 Model Performance", "📈 Data Insights", "ℹ️ About System"])
    
    scaler, model, artifacts_ok = load_artifacts()
    df, data_ok = load_dataset()
    
    if page == "🏠 Home":
        st.markdown("### Welcome!!")
        st.markdown("This dashboard leverages advanced regression analysis to estimate Benzene (C6H6) concentrations using a multi-sensor array. It provides real-time estimation and deep insights into air quality parameters.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">4</div>
                <div class="metric-label">Sensor Inputs</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">95%</div>
                <div class="metric-label">Model R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            count = len(df) if data_ok else "N/A"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{count}</div>
                <div class="metric-label">Data Points</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="cta-box">
            👉 READY to estimate? Navigate to the <b>🧪 Concentration Predictor</b> menu to input sensor data!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### System Workflow")
        
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        
        with col_w1:
            st.markdown("""
            <div class="workflow-step">
                <div class="workflow-icon">📡</div>
                <div class="workflow-title">Acquisition</div>
                <div class="workflow-desc">Multi-Sensor Data</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_w2:
            st.markdown("""
            <div class="workflow-step">
                <div class="workflow-icon">🧹</div>
                <div class="workflow-title">Cleaning</div>
                <div class="workflow-desc">IQR Outlier Removal</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_w3:
            st.markdown("""
            <div class="workflow-step">
                <div class="workflow-icon">📈</div>
                <div class="workflow-title">Modeling</div>
                <div class="workflow-desc">OLS Regression</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_w4:
            st.markdown("""
            <div class="workflow-step">
                <div class="workflow-icon">🎯</div>
                <div class="workflow-title">Prediction</div>
                <div class="workflow-desc">Benzene Estimation</div>
            </div>
            """, unsafe_allow_html=True)

    elif page == "🧪 Concentration Predictor":
        if not artifacts_ok:
            st.error("⚠️ Model artifacts not found. Please run the training script first.")
            st.stop()
            
        st.markdown("### Real-time Estimator")
        st.info("Adjust the sensor values below to estimate the Benzene (C6H6) concentration in the air.")
        
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            s1 = st.slider("PT08.S1 (CO Sensor)", min_value=600.0, max_value=2000.0, value=1000.0, help="Tin oxide sensor response (nominally CO)")
            s2 = st.slider("PT08.S2 (NMHC Sensor)", min_value=300.0, max_value=2000.0, value=900.0, help="Titania sensor response (nominally NMHC)")
            
        with col_in2:
            s4 = st.slider("PT08.S4 (NO2 Sensor)", min_value=300.0, max_value=2500.0, value=1500.0, help="Tungsten oxide sensor response (nominally NO2)")
            s5 = st.slider("PT08.S5 (O3 Sensor)", min_value=200.0, max_value=2500.0, value=1000.0, help="Indium oxide sensor response (nominally O3)")
            
        if st.button("Calculate Concentration", type="primary"):
            features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
            input_df = pd.DataFrame([[s1, s2, s4, s5]], columns=features)
            
            try:
                scaled_input = scaler.transform(input_df)
                exog = sm.add_constant(scaled_input, has_constant='add')
                prediction = model.predict(exog)[0]
                
                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.success("Estimation Complete")
                    st.metric(label="Benzene (C6H6)", value=f"{prediction:.4f} µg/m³")
                
                with col_res2:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Concentration Level"},
                        gauge = {
                            'axis': {'range': [None, 50]},
                            'bar': {'color': "#11998e"},
                            'steps': [
                                {'range': [0, 10], 'color': "#e8f5e9"},
                                {'range': [10, 25], 'color': "#c8e6c9"},
                                {'range': [25, 50], 'color': "#a5d6a7"}],
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    elif page == "📊 Model Performance":
        if not data_ok or not artifacts_ok:
            st.warning("Data or Model artifacts not found. Cannot evaluate performance.")
            st.stop()
            
        st.markdown("### Model Evaluation Results")
        
        features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
        X = df[features]
        y = df['C6H6(GT)']
        
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
        
        X_test_sm = sm.add_constant(X_test)
        y_pred = model.predict(X_test_sm)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        col3.metric("RMSE", f"{rmse:.4f}")
        
        st.markdown("### Actual vs Predicted Analysis")
        
        plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
        fig_scatter = px.scatter(
            plot_df, x='Actual', y='Predicted', 
            labels={'Actual': 'Actual Benzene (µg/m³)', 'Predicted': 'Predicted Benzene (µg/m³)'},
            title='Actual vs Predicted Concentration',
            opacity=0.6,
            trendline='ols',
            trendline_color_override='red'
        )
        
        fig_scatter.add_shape(
            type="line", line=dict(dash='dash', color='grey'),
            x0=plot_df['Actual'].min(), y0=plot_df['Actual'].min(),
            x1=plot_df['Actual'].max(), y1=plot_df['Actual'].max()
        )
        
        fig_scatter.update_traces(marker=dict(color='#11998e', size=8))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("**Residual Distribution**")
        residuals = y_test - y_pred
        fig_res = px.histogram(residuals, nbins=30, title="Residuals Histogram", 
                               color_discrete_sequence=['#ff7f0e'], marginal="box")
        fig_res.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_res, use_container_width=True)

    elif page == "📈 Data Insights":
        if not data_ok:
            st.warning("Dataset not available for visualization.")
            st.stop()
            
        st.markdown("### Dataset Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Correlations", "Distributions", "3D Analysis"])
        
        with tab1:
            st.markdown("**Feature Correlation Heatmap**")
            features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'C6H6(GT)']
            corr = df[features].corr()
            fig_corr = px.imshow(
                corr, 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix (Coolwarm Style)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with tab2:
            st.markdown("**Target Variable Distribution**")
            fig_dist = px.histogram(df, x="C6H6(GT)", nbins=50, title="Benzene Concentration Distribution", 
                                    color_discrete_sequence=['#11998e'], marginal="box")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with tab3:
            st.markdown("**Multivariate Analysis**")
            x_ax = st.selectbox("X Axis", options=df.columns, index=0)
            y_ax = st.selectbox("Y Axis", options=df.columns, index=2)
            z_ax = st.selectbox("Z Axis (Color)", options=df.columns, index=1)
            
            fig_3d = px.scatter(df, x=x_ax, y=y_ax, color=z_ax, 
                                color_continuous_scale='Viridis', title=f"{y_ax} vs {x_ax}")
            st.plotly_chart(fig_3d, use_container_width=True)

    elif page == "ℹ️ About System":
        st.markdown("### Technical Details")
        st.markdown("This project utilizes Inferential Statistics and Machine Learning to model air quality.")
        
        st.markdown("#### The Dataset")
        st.markdown("Data sourced from the UCI Machine Learning Repository (Air Quality Data Set). It contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device.")
        
        st.markdown("#### The Model")
        st.markdown("**Multiple Linear Regression (OLS)**")
        st.latex(r'''
        Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon
        ''')
        
        st.markdown("#### Features")
        st.markdown("""
        * **PT08.S1(CO):** Tin oxide sensor response (Targeting Carbon Monoxide)
        * **PT08.S2(NMHC):** Titania sensor response (Targeting Non-Methane Hydrocarbons)
        * **PT08.S4(NO2):** Tungsten oxide sensor response (Targeting Nitrogen Dioxide)
        * **PT08.S5(O3):** Indium oxide sensor response (Targeting Ozone)
        """)
        
        st.markdown("---")
        st.caption("© INT-24 Inferential Statistics Project.")

if __name__ == "__main__":
    main()
