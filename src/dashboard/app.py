"""
Streamlit dashboard for prompt recommendation system.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Prompt Recommender",
    page_icon="🎯",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

st.title("🎯 LLM Prompt Recommender System")
st.markdown("Find optimal prompt templates with cost optimization and safety scoring")

# Sidebar configuration
st.sidebar.header("Configuration")

domain_options = [
    "customer_support",
    "code_generation",
    "creative_writing",
    "data_analysis",
    "general"
]

domain = st.sidebar.selectbox("Domain", options=domain_options, index=0)
max_cost = st.sidebar.slider("Max Cost (USD)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
min_safety = st.sidebar.slider("Min Safety Score", min_value=0, max_value=100, value=70, step=5)
num_recommendations = st.sidebar.slider("Number of Results", min_value=1, max_value=10, value=5)
model_preference = st.sidebar.selectbox("Model Preference", options=["gpt-4", "claude-3-opus", "gemini-1.5-pro"])

# Main input
st.header("Enter Your Task Description")
task_description = st.text_area(
    "Describe what you want the LLM to do:",
    placeholder="e.g., Handle customer refund request for defective product",
    height=100
)

# Recommend button
if st.button("🔍 Get Recommendations", type="primary"):
    if not task_description:
        st.warning("Please enter a task description")
    else:
        with st.spinner("Finding best templates..."):
            try:
                # Call API
                response = requests.post(
                    f"{API_URL}/recommend",
                    json={
                        "task_description": task_description,
                        "domain": domain,
                        "max_cost_usd": max_cost,
                        "min_safety_score": min_safety,
                        "num_recommendations": num_recommendations,
                        "model_preference": model_preference
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    
                    if not recommendations:
                        st.info("No templates found matching your criteria. Try adjusting filters.")
                    else:
                        st.success(f"Found {len(recommendations)} recommendations!")
                        
                        # Display metrics overview
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_quality = sum(r["predicted_quality"] for r in recommendations) / len(recommendations)
                            st.metric("Avg Quality", f"{avg_quality:.2f}")
                        with col2:
                            avg_cost = sum(r["estimated_cost_usd"] for r in recommendations) / len(recommendations)
                            st.metric("Avg Cost", f"${avg_cost:.4f}")
                        with col3:
                            avg_safety = sum(r["safety_score"] for r in recommendations) / len(recommendations)
                            st.metric("Avg Safety", f"{avg_safety:.0f}/100")
                        
                        # Display recommendations
                        st.header("Recommended Templates")
                        
                        for idx, rec in enumerate(recommendations, 1):
                            with st.expander(f"#{idx} - Quality: {rec['predicted_quality']:.2f} | Cost: ${rec['estimated_cost_usd']:.4f}", expanded=(idx==1)):
                                # Safety badge
                                safety_score = rec["safety_score"]
                                if safety_score >= 90:
                                    st.success(f"✅ Safety Score: {safety_score:.0f}/100 (Excellent)")
                                elif safety_score >= 70:
                                    st.warning(f"⚠️ Safety Score: {safety_score:.0f}/100 (Good)")
                                else:
                                    st.error(f"❌ Safety Score: {safety_score:.0f}/100 (Review Required)")
                                
                                # Template text
                                st.subheader("Template")
                                st.code(rec["template_text"], language="text")
                                
                                # Metadata
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Predicted Quality:**", f"{rec['predicted_quality']:.3f}")
                                    st.write("**Combined Score:**", f"{rec['combined_score']:.3f}")
                                    st.write("**Domain:**", rec.get("domain", "N/A"))
                                
                                with col2:
                                    cost_info = rec["itemized_cost"]
                                    st.write("**Cost Breakdown:**")
                                    st.write(f"- Input tokens: {cost_info['input_tokens']}")
                                    st.write(f"- Output tokens: {cost_info['output_tokens']}")
                                    st.write(f"- Total cost: ${cost_info['cost_usd']:.4f}")
                        
                        # Visualization
                        st.header("Score Comparison")
                        
                        df = pd.DataFrame([
                            {
                                "Template": f"Template {i+1}",
                                "Quality": r["predicted_quality"],
                                "Safety": r["safety_score"] / 100,
                                "Cost": r["estimated_cost_usd"]
                            }
                            for i, r in enumerate(recommendations)
                        ])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name="Quality",
                            x=df["Template"],
                            y=df["Quality"],
                            marker_color="lightblue"
                        ))
                        fig.add_trace(go.Bar(
                            name="Safety (normalized)",
                            x=df["Template"],
                            y=df["Safety"],
                            marker_color="lightgreen"
                        ))
                        
                        fig.update_layout(
                            barmode="group",
                            title="Quality vs Safety Scores",
                            yaxis_title="Score",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Cost chart
                        fig_cost = px.bar(
                            df,
                            x="Template",
                            y="Cost",
                            title="Cost Comparison",
                            labels={"Cost": "Cost (USD)"},
                            color="Cost",
                            color_continuous_scale="Reds"
                        )
                        fig_cost.update_layout(height=300)
                        st.plotly_chart(fig_cost, use_container_width=True)
                
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the server is running at http://localhost:8000")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Health check
with st.sidebar:
    st.markdown("---")
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                st.success(f"✅ API Status: {health_data['status']}")
                st.json(health_data)
            else:
                st.error("❌ API Unhealthy")
        except:
            st.error("❌ Cannot reach API")

# Footer
st.markdown("---")
st.markdown("**LLM Prompt Recommender System** | Built with FastAPI + Streamlit")
