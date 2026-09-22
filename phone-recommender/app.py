"""
app.py
Streamlit UI for the Nigerian Smartphone Recommender.
Thin presentation layer only -- all logic lives in src/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from data_processing import load_catalog
from recommender import top_recommendations
from ml_recommender import top_recommendations_ml

st.set_page_config(page_title="Naija Phone Finder", page_icon="📱", layout="centered")

st.title("📱 Find Your Smartphone")
st.caption("Recommendations from a catalog of real, sourced Nigerian market prices "
           "(no fabricated specs or prices).")


@st.cache_data
def get_catalog():
    return load_catalog()


df = get_catalog()

with st.sidebar:
    st.header("Engine")
    engine = st.radio(
        "Recommendation engine",
        ["Machine Learning (Random Forest)", "Rule-based (transparent weights)"],
        index=0,
        help="Both are trained/derived from the same logic — the ML model was "
             "distilled from the rule-based scorer on 600 synthetic customers. "
             "Useful for comparing the two live.",
    )
    st.caption(f"Catalog: {len(df)} priced phones across {df['brand'].nunique()} brands.")

st.subheader("Your preferences")

col1, col2 = st.columns(2)
with col1:
    budget = st.number_input("Budget (₦)", min_value=50000, max_value=3000000,
                              value=250000, step=10000)
    primary_use = st.selectbox(
        "Primary use",
        ["general", "social_media", "camera", "gaming", "work", "student"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    brand_preference = st.selectbox(
        "Brand preference",
        ["Any", "TECNO", "Infinix", "Samsung", "itel", "Xiaomi", "Oppo", "Google Pixel", "Apple"],
    )

with col2:
    camera_priority = st.select_slider("Camera priority", ["low", "medium", "high"], value="medium")
    battery_priority = st.select_slider("Battery priority", ["low", "medium", "high"], value="medium")
    performance_priority = st.select_slider("Performance priority", ["low", "medium", "high"], value="medium")
    storage_priority = st.select_slider("Storage priority", ["low", "medium", "high"], value="medium")

find_clicked = st.button("🔍 Find My Phone", type="primary", use_container_width=True)

st.divider()

if find_clicked:
    customer = {
        "budget_ngn": budget,
        "primary_use": primary_use,
        "camera_priority": camera_priority,
        "battery_priority": battery_priority,
        "performance_priority": performance_priority,
        "storage_priority": storage_priority,
        "brand_preference": brand_preference,
    }

    use_ml = engine.startswith("Machine Learning")
    recs = (top_recommendations_ml if use_ml else top_recommendations)(df, customer, n=3)

    if not recs:
        st.warning(
            "No phones matched within your budget. Try raising it a bit — "
            "the cheapest phones in the catalog start around ₦98,000."
        )
    else:
        st.subheader("Top 3 matches")
        cards = st.columns(3)
        for col, rec in zip(cards, recs):
            with col:
                st.markdown(f"### {rec['brand']} {rec['model']}")
                st.metric("Match score", f"{rec['match_score']}%")
                st.write(f"**Price:** ₦{rec['price_ngn']:,.0f}")
                remaining = rec["remaining_budget"]
                if remaining >= 0:
                    st.write(f"**Remaining budget:** ₦{remaining:,.0f}")
                else:
                    st.write(f"**Over budget by:** ₦{abs(remaining):,.0f}")

                specs = rec["specs"]
                spec_lines = []
                if specs["ram_gb"] == specs["ram_gb"]:  # not NaN
                    spec_lines.append(f"RAM: {int(specs['ram_gb'])} GB")
                if specs["storage_gb"] == specs["storage_gb"]:
                    spec_lines.append(f"Storage: {int(specs['storage_gb'])} GB")
                if specs["battery_mah"] == specs["battery_mah"]:
                    spec_lines.append(f"Battery: {int(specs['battery_mah'])} mAh")
                if specs["main_camera_mp"] == specs["main_camera_mp"]:
                    spec_lines.append(f"Camera: {int(specs['main_camera_mp'])} MP")
                spec_lines.append(f"5G: {specs['five_g']}")
                st.write("  \n".join(spec_lines))

                st.write("**Why it matches:**")
                for reason in rec["why"]:
                    st.write(f"✓ {reason}")

        with st.expander("How this score was calculated"):
            if use_ml:
                st.write(
                    "This ranking came from a Random Forest model trained on 600 synthetic "
                    "customer profiles, using the transparent rule-based scores below as "
                    "training labels. It generalizes to unseen customers (R²=0.966 on held-out "
                    "data) but is not a black box you can't explain — the 'why it matches' "
                    "reasons above still come from the same transparent component scores."
                )
            else:
                st.write(
                    "Weighted blend — Budget 30%, Camera 20%, Battery 20%, Performance 15%, "
                    "Storage 10%, Brand 5% — with weights re-shaped by your primary use and "
                    "priority sliders before scoring."
                )
else:
    st.info("Set your preferences above and click **Find My Phone**.")
