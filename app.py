#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 21:15:52 2022

@author: shashankdwivedi
"""

import streamlit as st
from streamlit_option_menu import option_menu
from restaurant_recommender_system import displayToFront

st.title("Food Recommender System")
user_input = st.text_input("Enter the Name of Restaurant", 'Barbeque Nation')
st.write(displayToFront(user_input))
