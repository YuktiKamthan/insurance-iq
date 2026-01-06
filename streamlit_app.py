"""
Insurance IQ - Streamlit Dashboard
Interactive UI for fraud detection with SHAP explanations
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="Insurance IQ - Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# API base URL
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #c62828;
    }
    .safe-alert {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛡️ Insurance IQ</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">AI-Powered Fraud Detection Platform</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Analytics", "🧪 MLflow Experiments", "🔬 Drift Monitoring"])

# ================== TAB 1: SINGLE PREDICTION ==================
with tab1:
    st.header("Fraud Detection with Explanations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Enter Claim Details")
        
        claim_amount = st.number_input(
            "Claim Amount ($)",
            min_value=0,
            max_value=500000,
            value=15000,
            step=1000,
            help="Total amount claimed"
        )
        
        days_to_report = st.number_input(
            "Days to Report",
            min_value=0,
            max_value=365,
            value=30,
            help="Days between incident and claim filing"
        )
        
        claimant_age = st.number_input(
            "Claimant Age",
            min_value=16,
            max_value=100,
            value=35,
            help="Age of the claimant"
        )
        
        prior_claims = st.number_input(
            "Prior Claims",
            min_value=0,
            max_value=50,
            value=1,
            help="Number of previous claims by this claimant"
        )
        
        predict_button = st.button("🔍 Analyze Claim", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        if predict_button:
            with st.spinner("Analyzing claim..."):
                try:
                    # Make prediction request
                    response = requests.post(
                        f"{API_URL}/predict/fraud",
                        json={
                            "claim_amount": claim_amount,
                            "days_to_report": days_to_report,
                            "claimant_age": claimant_age,
                            "prior_claims": prior_claims
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display result with styling
                        if result['prediction'] == 'fraud':
                            st.markdown(f"""
                            <div class="fraud-alert">
                                <h3>⚠️ FRAUD DETECTED</h3>
                                <p><strong>Fraud Probability:</strong> {result['fraud_probability']:.1%}</p>
                                <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                                <p><strong>Recommended Action:</strong> {result['recommended_action']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="safe-alert">
                                <h3>✅ LEGITIMATE CLAIM</h3>
                                <p><strong>Fraud Probability:</strong> {result['fraud_probability']:.1%}</p>
                                <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                                <p><strong>Recommended Action:</strong> {result['recommended_action']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Probability gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = result['fraud_probability'] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Probability"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred" if result['fraud_probability'] > 0.5 else "darkgreen"},
                                'steps': [
                                    {'range': [0, 33], 'color': "lightgreen"},
                                    {'range': [33, 66], 'color': "yellow"},
                                    {'range': [66, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error(f"Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Failed to connect to API: {str(e)}")
                    st.info("Make sure your API is running: kubectl port-forward service/insurance-iq-service 8000:8000")
        
        else:
            st.info("👆 Enter claim details and click 'Analyze Claim'")
    
    # Get SHAP explanation
    if predict_button:
        st.subheader("🔬 Model Explanation (SHAP)")
        
        with st.spinner("Generating explanation..."):
            try:
                explain_response = requests.post(
                    f"{API_URL}/predict/fraud/explain",
                    json={
                        "claim_amount": claim_amount,
                        "days_to_report": days_to_report,
                        "claimant_age": claimant_age,
                        "prior_claims": prior_claims
                    }
                )
                
                if explain_response.status_code == 200:
                    shap_result = explain_response.json()
                    
                    st.markdown("**Feature Impact on Prediction:**")
                    
                    # Show the raw SHAP data for now
                    st.json(shap_result)
                    
            except Exception as e:
                st.warning(f"Could not load SHAP explanation: {str(e)}")

# ================== TAB 2: BATCH ANALYSIS ==================
with tab2:
    st.header("Batch Claim Analysis")
    
    st.markdown("""
    Upload a CSV file with multiple claims for batch processing.
    
    **Required columns:** `claim_amount`, `days_to_report`, `claimant_age`, `prior_claims`
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sample data generator
        if st.button("📋 Generate Sample Data"):
            try:
                response = requests.post(
                    f"{API_URL}/demo/generate-claims",
                    json={"num_claims": 10}
                )
                if response.status_code == 200:
                    sample_data = response.json()['claims']
                    df = pd.DataFrame(sample_data)
                    st.session_state['sample_df'] = df
                    st.success("✅ Generated 10 sample claims!")
            except:
                st.error("Failed to generate sample data")
        
        if 'sample_df' in st.session_state:
            st.dataframe(st.session_state['sample_df'], use_container_width=True)
            
            if st.button("🔍 Analyze Sample Claims", type="primary"):
                with st.spinner("Processing batch..."):
                    try:
                        claims_list = st.session_state['sample_df'].to_dict('records')
                        response = requests.post(
                            f"{API_URL}/predict/fraud/batch",
                            json={"claims": claims_list}
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            st.session_state['batch_results'] = results
                            st.success(f"✅ Processed {len(results['predictions'])} claims!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        if 'batch_results' in st.session_state:
            results = st.session_state['batch_results']
            summary = results.get('summary', {})
            
            # Summary metrics
            st.subheader("📊 Batch Summary")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Total Claims", summary.get('total_claims', 0))
            
            with metrics_col2:
                st.metric("Fraud Detected", summary.get('fraud_detected', 0))
            
            with metrics_col3:
                st.metric("Fraud Rate", f"{summary.get('fraud_rate_percent', 0):.1f}%")
            
            # Processing time
            st.info(f"⚡ Processed in {summary.get('batch_processing_time_ms', 0):.0f}ms ({summary.get('avg_time_per_claim_ms', 0):.0f}ms per claim)")
            
            # Results table
            st.subheader("Detailed Results")
            results_df = pd.DataFrame(results['predictions'])
            st.dataframe(results_df, use_container_width=True)

# ================== TAB 3: ANALYTICS ==================
with tab3:
    st.header("Real-Time Analytics Dashboard")
    
    if st.button("🔄 Refresh Metrics"):
        try:
            response = requests.get(f"{API_URL}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                st.session_state['metrics'] = metrics
        except:
            st.error("Failed to fetch metrics")
    
    if 'metrics' in st.session_state:
        metrics = st.session_state['metrics']
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔢 Total Predictions", metrics.get('total_predictions', 0))
        
        with col2:
            uptime_hours = metrics.get('uptime_seconds', 0) / 3600
            st.metric("⏱️ Uptime", f"{uptime_hours:.1f}h")
        
        with col3:
            avg_time = metrics.get('average_prediction_time_ms', 0)
            st.metric("⚡ Avg Response Time", f"{avg_time:.0f}ms")
        
        with col4:
            fraud_preds = metrics.get('fraud_predictions', {})
            fraud_count = fraud_preds.get('fraud', 0)
            st.metric("⚠️ Fraud Detected", fraud_count)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud distribution pie chart
            fraud_data = metrics.get('fraud_predictions', {})
            if fraud_data:
                fig = px.pie(
                    values=list(fraud_data.values()),
                    names=list(fraud_data.keys()),
                    title="Fraud Distribution",
                    color_discrete_sequence=['#2e7d32', '#c62828']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk level distribution
            risk_data = metrics.get('risk_level_distribution', {})
            if risk_data:
                fig = px.bar(
                    x=list(risk_data.keys()),
                    y=list(risk_data.values()),
                    title="Risk Level Distribution",
                    labels={'x': 'Risk Level', 'y': 'Count'},
                    color=list(risk_data.values()),
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Refresh Metrics' to load analytics")

# ================== TAB 4: MLFLOW ==================
with tab4:
    st.header("🧪 MLflow Experiment Tracking")
    
    st.markdown("""
    View your ML experiments and tracked predictions.
    
    **MLflow UI:** [http://localhost:5000](http://localhost:5000)
    """)
    
    st.info("💡 To track a prediction, use the 'Track with MLflow' option in the Single Prediction tab.")
    
    # Add tracking option to single prediction
    st.subheader("Quick Test with MLflow Tracking")
    
    if st.button("🧪 Run Test Prediction with Tracking"):
        with st.spinner("Making tracked prediction..."):
            try:
                test_claim = {
                    "claim_amount": 75000,
                    "days_to_report": 95,
                    "claimant_age": 23,
                    "prior_claims": 6
                }
                
                response = requests.post(
                    f"{API_URL}/predict/fraud/tracked",
                    json=test_claim
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Prediction logged to MLflow!")
                    st.json(result)
                    st.markdown(f"**MLflow Run ID:** `{result.get('mlflow_run_id', 'N/A')}`")
                    st.markdown("[Open MLflow UI →](http://localhost:5000)")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ================== TAB 5: DRIFT MONITORING ==================
with tab5:
    st.header("Data Drift Detection")
    st.markdown("Monitor distribution shifts that could affect model performance")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🧪 Test Scenarios")
        
        # Quick test buttons
        if st.button("✅ Test Normal Data", use_container_width=True):
            test_data = [
                {"claim_amount": 15000, "days_to_report": 5, "claimant_age": 35, "prior_claims": 1},
                {"claim_amount": 25000, "days_to_report": 10, "claimant_age": 45, "prior_claims": 2},
                {"claim_amount": 8000, "days_to_report": 3, "claimant_age": 28, "prior_claims": 0}
            ]
            st.session_state.drift_data = test_data
            st.session_state.drift_test = "normal"
        
        if st.button("🚨 Test Extreme Data", use_container_width=True, type="primary"):
            test_data = [
                {"claim_amount": 150000, "days_to_report": 90, "claimant_age": 85, "prior_claims": 15},
                {"claim_amount": 200000, "days_to_report": 120, "claimant_age": 90, "prior_claims": 20},
                {"claim_amount": 180000, "days_to_report": 100, "claimant_age": 82, "prior_claims": 18}
            ]
            st.session_state.drift_data = test_data
            st.session_state.drift_test = "extreme"
        
        # Display test data
        if 'drift_data' in st.session_state:
            st.markdown("**Test Data:**")
            df_test = pd.DataFrame(st.session_state.drift_data)
            st.dataframe(df_test, use_container_width=True)
    
    with col2:
        st.subheader("📊 Drift Analysis Results")
        
        if 'drift_data' in st.session_state:
            with st.spinner("Running drift detection..."):
                try:
                    response = requests.post(
                        f"{API_URL}/drift/check",
                        json=st.session_state.drift_data
                    )
                    
                    if response.status_code == 200:
                        drift_result = response.json()
                        
                        # Status indicator
                        if drift_result['status'] == 'ALERT':
                            st.markdown(f"""
                            <div class="fraud-alert">
                                <h3>🚨 DRIFT DETECTED</h3>
                                <p><strong>Status:</strong> Critical - Model retraining recommended</p>
                                <p><strong>Samples Analyzed:</strong> {drift_result['n_samples_analyzed']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="safe-alert">
                                <h3>✅ NO DRIFT DETECTED</h3>
                                <p><strong>Status:</strong> Model performance stable</p>
                                <p><strong>Samples Analyzed:</strong> {drift_result['n_samples_analyzed']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Feature drift scores
                        st.markdown("### Feature Drift Scores")
                        
                        features = list(drift_result['feature_drift'].keys())
                        drift_scores = [drift_result['feature_drift'][f]['drift_score'] for f in features]
                        drift_detected = [drift_result['feature_drift'][f]['drift_detected'] for f in features]
                        
                        # Create drift score chart
                        colors = ['red' if d else 'green' for d in drift_detected]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=features,
                                y=drift_scores,
                                marker_color=colors,
                                text=[f"{score:.4f}" for score in drift_scores],
                                textposition='auto',
                            )
                        ])
                        
                        fig.update_layout(
                            title="Drift Scores by Feature (Lower = More Drift)",
                            xaxis_title="Features",
                            yaxis_title="Drift Score (p-value)",
                            height=400,
                            showlegend=False
                        )
                        
                        # Add threshold line at 0.05
                        fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                                     annotation_text="Alert Threshold (p=0.05)")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Distribution comparison
                        st.markdown("### Distribution Comparison")
                        
                        # Create comparison metrics
                        comparison_data = []
                        for feature in features:
                            comparison_data.append({
                                'Feature': feature,
                                'Current Mean': f"{drift_result['feature_drift'][feature]['current_mean']:.2f}",
                                'Baseline Mean': f"{drift_result['feature_drift'][feature]['reference_mean']:.2f}",
                                'Drift': '🚨' if drift_result['feature_drift'][feature]['drift_detected'] else '✅'
                            })
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, use_container_width=True)
                        
                        # Recommendations
                        st.markdown("### 💡 Recommendations")
                        for rec in drift_result['recommendations']:
                            if '⚠️' in rec:
                                st.error(rec)
                            else:
                                st.success(rec)
                        
                    else:
                        st.error(f"Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        else:
            st.info("👈 Click a test button to run drift detection")                

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit | Insurance IQ v1.0 | 
    <a href="http://localhost:8000/docs" target="_blank">API Docs</a> | 
    <a href="http://localhost:3000" target="_blank">Grafana</a> | 
    <a href="http://localhost:5000" target="_blank">MLflow</a>
</div>
""", unsafe_allow_html=True)