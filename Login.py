#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:39:00 2022

@author: shashankdwivedi
"""

import streamlit as st

async def login():
    st.title("Loging Window")
    user_input = st.text_input("Enter Your profile")
    if user_input == "Shashank":
        return True
    else:
        return False