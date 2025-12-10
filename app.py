import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# -------------------------------------------------

LOCAL_FILENAMES = [
    "India housing prices.csv",      # primary name
    "india_housing_prices.csv",      # backup name
]

@st.cache_data
def load_data(uploaded_file=None):
    """
    Load dataset from:
    1) Uploaded file (if provided)
    2) Local CSV in same folder as app.py

    Returns:
        df (DataFrame), city_medians (Series)
    """
    # 1) If user uploaded a file, use that
    if uploaded_file is not None:
        # read directly from the uploaded file
        df = pd.read_csv(uploaded_file)
    else:
        # 2) Try local files in current working directory
        df = None
        for name in LOCAL_FILENAMES:
            candidate = Path(name)
            if candidate.exists():
                df = pd.read_csv(candidate)
                break

        if df is None:
            st.error(
                "❌ Could not find the dataset.\n\n"
                "Upload a CSV using the file uploader in the sidebar, "
                "or place `India housing prices.csv` next to `app.py`."
            )
            st.stop()

    # ---- Feature engineering (same logic as your EDA notebook) ----
    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]
    df["Location"] = df["City"] + " - " + df["Locality"]

    city_medians = df.groupby("City")["Price_per_SqFt"].median()

    df["Relative_Price_To_City"] = df["Price_per_SqFt"] / df["City"].map(city_medians)

    df["Good_Investment"] = np.where(
        (df["Relative_Price_To_City"] <= 0.90)
        & (df["BHK"] >= 2)
        & (df["Parking_Space"] == "Yes")
        & (df["Public_Transport_Accessibility"].isin(["High", "Medium"])),
        "Yes",
        "No",
    )

    return df, city_medians


def estimate_future_price(current_price_lakhs, annual_growth_rate, years=5):
    """Compound growth: Future = Current * (1 + r)^t"""
    return current_price_lakhs * ((1 + annual_growth_rate) ** years)


def investment_score(relative_price, bhk, parking, transport_access):
    """
    Simple rule-based score (0–100) as 'confidence'.
    """
    score = 50

    # Affordability vs city
    if relative_price <= 0.85:
        score += 25
    elif relative_price <= 0.95:
        score += 15
    elif relative_price <= 1.05:
        score += 5
    else:
        score -= 10

    # BHK
    if bhk >= 3:
        score += 10
    elif bhk == 2:
        score += 5
    else:
        score -= 5

    # Parking
    if parking == "Yes":
        score += 5

    # Public transport
    if transport_access == "High":
        score += 10
    elif transport_access == "Medium":
        score += 5
    else:
        score -= 5

    return int(np.clip(score, 0, 100))


# -------------------------------------------------
# 2. STREAMLIT APP
# -------------------------------------------------

def main():
    st.set_page_config(
        page_title="Real Estate Investment Advisor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏠 Real Estate Investment Advisor")
    st.write(
        "EDA-driven app to explore the housing dataset and get rule-based investment suggestions."
    )

    # ---- Sidebar file uploader ----
    st.sidebar.header("📂 Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset CSV (optional)", type=["csv"]
    )
    st.sidebar.caption(
        "If you don't upload, the app will try to load "
        "`India housing prices.csv` from the app folder."
    )

    # Load data (from upload or local file)
    df, city_medians = load_data(uploaded_file)

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.header("🔍 Filters")

    states = ["All"] + sorted(df["State"].unique().tolist())
    selected_state = st.sidebar.selectbox("State", states)

    if selected_state != "All":
        df_filtered = df[df["State"] == selected_state].copy()
        city_options = ["All"] + sorted(df_filtered["City"].unique().tolist())
    else:
        df_filtered = df.copy()
        city_options = ["All"] + sorted(df["City"].unique().tolist())

    selected_city = st.sidebar.selectbox("City", city_options)

    if selected_city != "All":
        df_filtered = df_filtered[df_filtered["City"] == selected_city]

    bhk_options = sorted(df_filtered["BHK"].unique())
    selected_bhk = st.sidebar.multiselect(
        "BHK", bhk_options, default=bhk_options
    )

    price_min, price_max = st.sidebar.slider(
        "Price range (Lakhs)",
        float(df["Price_in_Lakhs"].min()),
        float(df["Price_in_Lakhs"].max()),
        (
            float(df["Price_in_Lakhs"].min()),
            float(df["Price_in_Lakhs"].max()),
        ),
    )

    df_filtered = df_filtered[
        (df_filtered["BHK"].isin(selected_bhk))
        & (df_filtered["Price_in_Lakhs"].between(price_min, price_max))
    ]

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Filtered properties:** {len(df_filtered):,}")

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(
        ["📊 Market Overview", "📋 Investment Advisor", "📈 Visual Insights"]
    )

    # -----------------------------
    # Tab 1 – Market Overview
    # -----------------------------
    with tab1:
        st.subheader("📊 Market Overview (Filtered Dataset)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Properties", f"{len(df_filtered):,}")
        with col2:
            st.metric(
                "Avg Price (Lakhs)",
                f"{df_filtered['Price_in_Lakhs'].mean():.1f}",
            )
        with col3:
            st.metric(
                "Avg Price per SqFt",
                f"{df_filtered['Price_per_SqFt'].mean():.0f}",
            )

        st.markdown("### Sample of Filtered Properties")
        st.dataframe(
            df_filtered[
                [
                    "State",
                    "City",
                    "Locality",
                    "Property_Type",
                    "BHK",
                    "Size_in_SqFt",
                    "Price_in_Lakhs",
                    "Price_per_SqFt",
                    "Good_Investment",
                ]
            ].head(50)
        )

        st.markdown("### 📥 Download Filtered Dataset")
        st.download_button(
            label="Download filtered data as CSV",
            data=df_filtered.to_csv(index=False),
            file_name="filtered_india_housing_prices.csv",
            mime="text/csv",
        )

    # -----------------------------
    # Tab 2 – Investment Advisor
    # -----------------------------
    with tab2:
        st.subheader("📋 Property Investment Advisor")

        st.write(
            "Fill in property details below to check whether it is a **Good Investment** "
            "(rule-based from EDA) and to estimate the **price after 5 years**."
        )

        col_left, col_right = st.columns(2)

        # --- Left side: basic details ---
        with col_left:
            input_city = st.selectbox("City", sorted(df["City"].unique()))
            city_subset = df[df["City"] == input_city]

            input_locality = st.selectbox(
                "Locality", sorted(city_subset["Locality"].unique())
            )

            input_property_type = st.selectbox(
                "Property Type", sorted(df["Property_Type"].unique())
            )

            input_bhk = st.number_input(
                "BHK", min_value=1, max_value=10, value=2, step=1
            )

            input_size = st.number_input(
                "Size in SqFt",
                min_value=200.0,
                max_value=10000.0,
                value=1000.0,
                step=50.0,
            )

            input_price = st.number_input(
                "Current Price (Lakhs)",
                min_value=10.0,
                max_value=1000.0,
                value=float(city_subset["Price_in_Lakhs"].median()),
                step=1.0,
            )

        # --- Right side: amenities / access ---
        with col_right:
            input_furnished = st.selectbox(
                "Furnished Status", sorted(df["Furnished_Status"].unique())
            )

            input_parking = st.selectbox(
                "Parking Space", sorted(df["Parking_Space"].unique())
            )

            input_security = st.selectbox(
                "Security", sorted(df["Security"].unique())
            )

            input_amenity = st.selectbox(
                "Primary Amenity", sorted(df["Amenities"].unique())
            )

            input_transport = st.selectbox(
                "Public Transport Accessibility",
                sorted(df["Public_Transport_Accessibility"].unique()),
            )

            annual_growth = st.slider(
                "Assumed Annual Growth Rate (%)",
                min_value=2,
                max_value=20,
                value=8,
                step=1,
            )

        submitted = st.button("🔍 Analyse Investment")

        if submitted:
            user_price_psf = (input_price * 100000) / input_size
            city_median_psf = city_medians.loc[input_city]
            relative_price = user_price_psf / city_median_psf

            good_investment = (
                (relative_price <= 0.90)
                and (input_bhk >= 2)
                and (input_parking == "Yes")
                and (input_transport in ["High", "Medium"])
            )

            score = investment_score(
                relative_price, input_bhk, input_parking, input_transport
            )

            future_price = estimate_future_price(
                input_price, annual_growth_rate=annual_growth / 100.0, years=5
            )

            st.markdown("---")
            st.markdown("### ✅ Investment Result")

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric(
                    "Good Investment?",
                    "Yes ✅" if good_investment else "No ❌",
                )
            with colB:
                st.metric("Confidence Score", f"{score} / 100")
            with colC:
                st.metric(
                    "Estimated Price after 5 Years (Lakhs)",
                    f"{future_price:.1f}",
                )

            st.markdown("#### Details")
            st.write(f"- City median price per SqFt: **{city_median_psf:,.0f}**")
            st.write(f"- Your property price per SqFt: **{user_price_psf:,.0f}**")
            st.write(
                f"- Relative price vs city median: **{relative_price:.2f}x** "
                f"({'Cheaper' if relative_price < 1 else 'Costlier'})"
            )
            st.info(
                "Note: This decision and score are **rule-based from EDA**, "
                "not from a trained machine learning model."
            )

    # -----------------------------
    # Tab 3 – Visual Insights
    # -----------------------------
    with tab3:
        st.subheader("📈 Visual Insights")

        st.markdown("### Location-wise Average Price per SqFt (Top 15 Cities)")
        top_cities_pps = (
            df.groupby("City")["Price_per_SqFt"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        st.bar_chart(top_cities_pps)

        st.markdown("### Good vs Not Good Investments (by City – Filtered Data)")
        good_counts = (
            df_filtered.groupby(["City", "Good_Investment"])["ID"]
            .count()
            .unstack(fill_value=0)
        )
        st.bar_chart(good_counts)

        st.markdown(
            "Use the **sidebar filters** to see how prices and 'Good Investment' "
            "properties vary across states, cities, BHK, and price ranges."
        )


if __name__ == "__main__":
    main()
