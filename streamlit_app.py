import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lgboost import LGBMClassifier
from catboost import CatBoostClassifier
import json


def load_lottiefile(filepath:str):
    with open(filepath,'r') as f:
        lottie_json = json.load(f)
    return lottie_json

filename = load_lottiefile("doctor_animation.json")
st_lottie(filename, speed=1)

st.title('Parkinsons Severity Prediction')



