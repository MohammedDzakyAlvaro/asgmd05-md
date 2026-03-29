import sys
from pathlib import Path

# =========================
# Add project paths
# =========================
ROOT_DIR = Path(__file__).resolve().parent.parent  # misal apps/.. -> root
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st
from config.config import ARTIFACT_PIPELINE
from src.utils.io import load_artifact

# =========================
# Load pipeline
# =========================
@st.cache_resource
def load_pipeline():
    """Load trained pipeline artifact"""
    try:
        return load_artifact(ARTIFACT_PIPELINE)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


# =========================
# Main App
# =========================
def main():
    st.title("Spaceship Titanic Model Deployment")

    # Load model pipeline
    model = load_pipeline()

    # =========================
    # User Inputs
    # =========================
    PassengerId = st.text_input("PassengerId", "0067_67")
    HomePlanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
    CryoSleep = st.selectbox("CryoSleep", [True, False])
    Cabin = st.text_input("Cabin", "B/67/P")
    Destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
    Age = st.number_input("Age", 0, 100, 30)
    VIP = st.selectbox("VIP", [True, False])
    RoomService = st.number_input("RoomService", 0, 10000, 67)
    FoodCourt = st.number_input("FoodCourt", 0, 10000, 67)
    ShoppingMall = st.number_input("ShoppingMall", 0, 10000, 67)
    Spa = st.number_input("Spa", 0, 10000, 67)
    VRDeck = st.number_input("VRDeck", 0, 10000, 67)
    Name = st.text_input("Name", "mpruy giluy")

    # Prepare input dataframe
    input_df = pd.DataFrame([{
        "PassengerId": PassengerId,
        "HomePlanet": HomePlanet,
        "CryoSleep": CryoSleep,
        "Cabin": Cabin,
        "Destination": Destination,
        "Age": Age,
        "VIP": VIP,
        "RoomService": RoomService,
        "FoodCourt": FoodCourt,
        "ShoppingMall": ShoppingMall,
        "Spa": Spa,
        "VRDeck": VRDeck,
        "Name": Name
    }])

    # =========================
    # Prediction
    # =========================
    if st.button("Predict"):
        pred = model.predict(input_df)[0]

        if pred == 1:
            st.success("Passenger was Transported")
        else:
            st.error("Passenger was NOT Transported")


if __name__ == "__main__":
    main()