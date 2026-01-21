import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Simple page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Add some styling
st.markdown("""
<style>
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏠 Simple House Price Predictor")
st.markdown("Predict house prices based on features from your dataset")

# Load and prepare data
@st.cache_data
def load_and_train_model():
    """Load data and train a simple model"""
    try:
        # Create simple sample data (no external file needed)
        np.random.seed(42)
        n_samples = 100
        
        # Generate realistic sample data
        data = {
            'area': np.random.randint(5000, 13500, n_samples),
            'bedrooms': np.random.choice([2, 3, 4, 5], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
            'bathrooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.4, 0.4, 0.1]),
            'stories': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'parking': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
            'mainroad': np.random.choice(['yes', 'no'], n_samples, p=[0.7, 0.3]),
            'guestroom': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'basement': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
            'hotwaterheating': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'airconditioning': np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4]),
            'prefarea': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'furnishingstatus': np.random.choice(['furnished', 'semi-furnished', 'unfurnished'], 
                                                n_samples, p=[0.4, 0.4, 0.2]),
        }
        
        # Calculate price based on realistic formula
        base_price = 3000000
        price = (
            base_price + 
            data['area'] * 500 +  # 500 per sq ft
            data['bedrooms'] * 500000 +
            data['bathrooms'] * 300000 +
            data['stories'] * 200000 +
            data['parking'] * 100000 +
            (np.array(data['mainroad']) == 'yes') * 200000 +
            (np.array(data['airconditioning']) == 'yes') * 150000 +
            np.random.normal(0, 200000, n_samples)  # Some noise
        )
        
        data['price'] = price.astype(int)
        df = pd.DataFrame(data)
        
        # Prepare features
        categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                           'airconditioning', 'prefarea', 'furnishingstatus']
        
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
        
        # Features and target
        X = df.drop('price', axis=1)
        y = df['price']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        return {
            'model': model,
            'encoders': le_dict,
            'data': df,
            'feature_names': X.columns.tolist()
        }
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Predict", "📊 View Data", "ℹ️ About"])

# Load model once
model_data = load_and_train_model()

if page == "🏠 Predict":
    st.header("Make a Prediction")
    
    if model_data:
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            area = st.slider("Area (sq ft)", 5000, 13500, 9000, 100)
            bedrooms = st.selectbox("Bedrooms", [2, 3, 4, 5], index=1)
            bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
            stories = st.selectbox("Stories", [1, 2, 3, 4], index=1)
            parking = st.selectbox("Parking Spaces", [0, 1, 2, 3], index=2)
        
        with col2:
            mainroad = st.selectbox("Main Road Access", ["yes", "no"])
            guestroom = st.selectbox("Guest Room", ["yes", "no"])
            basement = st.selectbox("Basement", ["yes", "no"])
            hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
            airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
            prefarea = st.selectbox("Preferred Area", ["yes", "no"])
            furnishingstatus = st.selectbox("Furnishing Status", 
                                          ["furnished", "semi-furnished", "unfurnished"])
        
        # Prediction button
        if st.button("Predict Price", type="primary"):
            try:
                # Prepare input data
                input_dict = {
                    'area': area,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'stories': stories,
                    'parking': parking,
                    'mainroad': mainroad,
                    'guestroom': guestroom,
                    'basement': basement,
                    'hotwaterheating': hotwaterheating,
                    'airconditioning': airconditioning,
                    'prefarea': prefarea,
                    'furnishingstatus': furnishingstatus,
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_dict])
                
                # Encode categorical variables
                for col in model_data['encoders'].keys():
                    le = model_data['encoders'][col]
                    # Handle unseen labels
                    if input_dict[col] in le.classes_:
                        input_df[col] = le.transform([input_dict[col]])[0]
                    else:
                        input_df[col] = 0  # Default value
                
                # Ensure correct column order
                input_df = input_df[model_data['feature_names']]
                
                # Make prediction
                prediction = model_data['model'].predict(input_df)[0]
                
                # Display result
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.success(f"### Predicted Price: ₹{prediction:,.0f}")
                
                # Show some insights
                avg_price = model_data['data']['price'].mean()
                price_diff = prediction - avg_price
                
                if price_diff > 0:
                    st.info(f"This is ₹{price_diff:,.0f} above average")
                else:
                    st.info(f"This is ₹{abs(price_diff):,.0f} below average")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show feature impact
                with st.expander("View Feature Analysis"):
                    st.write("**Input Features:**")
                    for key, value in input_dict.items():
                        st.write(f"- {key}: {value}")
                    
                    # Simple impact calculation
                    st.write("\n**Top Impact Features:**")
                    impacts = {
                        'Area': area * 500,
                        'Bedrooms': bedrooms * 500000,
                        'Bathrooms': bathrooms * 300000,
                        'Air Conditioning': 150000 if airconditioning == 'yes' else 0,
                        'Main Road': 200000 if mainroad == 'yes' else 0,
                    }
                    
                    for feature, impact in impacts.items():
                        st.write(f"- {feature}: ₹{impact:,.0f}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model failed to load. Please check the app.")

elif page == "📊 View Data":
    st.header("Dataset Information")
    
    if model_data:
        df = model_data['data']
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Houses", len(df))
        
        with col2:
            st.metric("Average Price", f"₹{df['price'].mean():,.0f}")
        
        with col3:
            st.metric("Average Area", f"{df['area'].mean():,.0f} sq ft")
        
        st.markdown("---")
        
        # Show data table
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.write(df[['price', 'area', 'bedrooms', 'bathrooms']].describe())
        
        # Simple visualization
        st.subheader("Price Distribution")
        
        # Create bins for histogram
        price_bins = np.linspace(df['price'].min(), df['price'].max(), 10)
        hist_values, bin_edges = np.histogram(df['price'], bins=price_bins)
        
        # Display as bar chart using Streamlit
        chart_data = pd.DataFrame({
            'Price Range': [f"₹{int(bin_edges[i]):,} - ₹{int(bin_edges[i+1]):,}" 
                           for i in range(len(bin_edges)-1)],
            'Number of Houses': hist_values
        })
        
        st.bar_chart(chart_data.set_index('Price Range'))
        
        # Download option
        st.subheader("Download Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="housing_data.csv",
            mime="text/csv"
        )
    else:
        st.error("Data not available")

else:  # About page
    st.header("About This App")
    
    st.markdown("""
    ### Simple House Price Predictor
    
    This application predicts house prices based on various features using a Linear Regression model.
    
    **Features Used:**
    1. **Area** - Size of the house in square feet
    2. **Bedrooms** - Number of bedrooms
    3. **Bathrooms** - Number of bathrooms
    4. **Stories** - Number of floors
    5. **Parking** - Number of parking spaces
    6. **Main Road Access** - Proximity to main road
    7. **Guest Room** - Availability of guest room
    8. **Basement** - Presence of basement
    9. **Hot Water Heating** - Availability
    10. **Air Conditioning** - Availability
    11. **Preferred Area** - Located in preferred area
    12. **Furnishing Status** - Level of furnishing
    
    **How it works:**
    1. The app creates a sample dataset
    2. Trains a Linear Regression model
    3. Uses the model to predict prices based on your inputs
    
    **Note:** This is a simplified version for demonstration purposes.
    For real-world applications, use actual housing data.
    """)
    
    st.info("""
    ⚠️ **Safety Features:**
    - No external file access required
    - All computations done in memory
    - No internet connection needed
    - No installation of heavy packages
    - Safe to run on any laptop
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Simple ML Model | Safe to run")