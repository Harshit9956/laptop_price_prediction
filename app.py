import streamlit as st
import pickle
import numpy as np
from sklearn.pipeline import Pipeline

pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))


st.title('Laptop Price Predictor')

company=st.selectbox('Brand',df['Company'].unique())
type=st.selectbox('TypeName',df['TypeName'].unique())

ram=st.selectbox('Ram (in GB)',[2,4,6,8,12,16,24,64])
weight=st.number_input('weight in kg')
touchscreen=st.selectbox('Touchscreen',['Yes','No'])
ips=st.selectbox('Ips Panel',['Yes','No'])
full_hd=st.selectbox('Full HD',['Yes','No'])

screen_size=st.number_input('screen size in Inches')
resolution=st.text_input('Resolution (1000x1000) format')
cpu=st.selectbox('Cpu brand',df['Cpu'].unique())
hdd=st.selectbox('HDD(in GB)',[0,32,128,256,516,1024,2048])
ssd=st.selectbox('SSD(in GB)',[0,8,16,32,128,256,516,1024,2048])

gpu=st.selectbox('Gpu',df['Gpu_brand'].unique())
os=st.selectbox('OS',df['OpSys'].unique())

if st.button('predict price'):
    if touchscreen=='Yes':
        touchscreen= 1
    else:
        touchscreen=0
    
    if ips=='Yes':
        ips= 1
    else:
        ips=0

    if full_hd=='Yes':
        full_hd= 1
    else:
        full_hd=0

    x_res=int(resolution.split('x')[0])
    y_res=int(resolution.split('x')[1])
    ppi=((x_res**2)+(y_res**2))**0.5/screen_size
    
    query=np.array([company,type,cpu,ram,os,weight,ips,full_hd,touchscreen,ssd,hdd,gpu,ppi])
    # ['Company', 'TypeName', 'Cpu', 'Ram', 'OpSys', 'Weight', 'Ips panel',
    #    'full_hd', 'touchscreen', 'SSD', 'HDD', 'Gpu_brand', 'ppi'],
    query=query.reshape(1,13)

    st.title(np.exp(pipe.predict(query)))

