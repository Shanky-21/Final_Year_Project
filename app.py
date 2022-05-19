#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 21:15:52 2022

@author: shashankdwivedi
"""

import streamlit as st
from streamlit_option_menu import option_menu
from restaurant_recommender_system import displayToFront, plot,top_10
from multiPage import MultiPage
from Login import login
from Test import Names
import asyncio
import time
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(
     page_title="Ex-stream-ly Cool App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    #text = link.split('=')[1]
    return f'<a target="_blank" href="{link}">{"Visit"}</a>'


def main():
    #login page
    #res = True
    
    with st.spinner("Recommending..."):
        time.sleep(3)
    st.success("Results!")
    user_input = st.text_input("Enter the Name of Restaurant", 'Barbeque Nation')
    user_input2 = st.number_input("Enter the Number of Recommendations", 5)
    df = displayToFront(user_input, user_input2)
    st.write("This is the recommendations")
    df3 = df.copy()
    df3["Links"] = df3["Links"].apply(make_clickable)
    df3 = df3.to_html(escape=False)
    st.write(df3, unsafe_allow_html=True)
    st.write("This is the Line_Chart For Mean Rating")
    st.line_chart(df['Mean Rating'])
    st.write("Following is the visualization of Mean Rating of Recommended Restaurants")
    df = plot(df)
    st.plotly_chart(df)
    fig2 = top_10()
    st.plotly_chart(fig2)
    
    
        
        

def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

#Create an instance of the app 


app = MultiPage()
#header("Recommendation System")

# Title of the main page
#st.title("Recommendation System")
image = Image.open('/Users/shashankdwivedi/Documents/Final_Year_Project/Recommender System.jpeg')

st.image(image, caption='Recommender System')
st.title("Restaurant Recommendation Application")

app.add_page("Restaurant Recommendation", main)
app.add_page("Restaurant Names", Names)

# The main app
app.run()

#main()