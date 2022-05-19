#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:31:22 2022

@author: shashankdwivedi
"""

import streamlit as st
from PIL import Image
from restaurant_recommender_system import unique_restaurants

def Names():
    st.header("Here is the List of Restaurants In Hyderabad")
    st.dataframe(unique_restaurants(), 1000, 500)
    