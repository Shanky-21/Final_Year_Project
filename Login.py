#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:39:00 2022

@author: shashankdwivedi
"""

import streamlit as st
import asyncio
import time

def login():
    st.title("Loging Window")
    user_input = st.text_input("Enter Your UserName")
    password = st.text_input("Enter your password")
    #
    time.sleep(3)
    with st.spinner("Loading..."):
        time.sleep(2)
    st.success("Authenticated")
    if user_input == "Shashank" and password == "123":
        return True
    else:
        st.title("Auth Failed")
        return False
    