import streamlit as st
from prophet import Prophet
import pandas as pd
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from prophet.plot import plot_plotly
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# #MainMenu {visibility: hidden;}
hide = """
<style>
footer {visibility: hidden;}
</style> 
"""
st.markdown(hide, unsafe_allow_html=True)

st.title("Stock Forecasting")

stocks = st.text_input("Enter Stock Name")

@st.cache(allow_output_mutation=True)
def load_dataset(filename):
    df = pd.read_json(filename)
    return df
df = load_dataset('https://raw.githubusercontent.com/Vatsal2251/stock-forecasting/main/stocks.json')
    
    
if stocks != "":
    df1 = df.loc[df["company_name"].str.contains(stocks, case=False, na=False)]
    df3 = df1["company_name"]+" - "+df1["code_name"]
    selected_stock = st.selectbox("List of Matching Stocks",df3)
    if df1.empty:
        st.write("No Match Found")
    else:
        codeName = selected_stock.split(" - ")[1]
        #st.write(codeName)
    
        START  = "2005-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")
        #st.write(TODAY)
        @st.cache()
        def download_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data
    
        data = download_data(codeName)
    
        st.subheader("Real Data")
        st.write(data.tail())

        st.cache() 
        def plot_real_data_line():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = data['Date'],y = data['Open'], name = 'open'))
            fig.add_trace(go.Scatter(x = data['Date'],y = data['Close'], name = 'close'))
            fig.layout.update( xaxis_rangeslider_visible = True)
            st.plotly_chart(fig)

        st.cache()
        def plot_real_data_candlestic():
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.03,row_width=[0.2, 0.7])
            fig.add_trace(go.Candlestick(x=data["Date"], open=data["Open"], high=data["High"],
                low=data["Low"], close=data["Close"], name="OHLC"), 
                row=1, col=1)
            #fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], showlegend=False), row=2, col=1)
            fig.update(layout_xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        chart = st.radio("Select Your Chart",('Line Chart', 'Candlestic Chart'))

        if chart == 'Line Chart':
            plot_real_data_line()
        if chart == 'Candlestic Chart':
            plot_real_data_candlestic()
        
        #fbprophet:
        st.cache(allow_output_mutation=True)
        def fbprophet_model():
            train_data = data.drop(['Open','High','Low','Adj Close','Volume'],axis=1)
            train_data = train_data.rename( columns={"Date":"ds", "Close":"y"})
            m = Prophet()
            m.fit(train_data)
            return m


        def fbprophet_impl():
            m = fbprophet_model()
            period = 120
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)
            fig_fb = plot_plotly(m,forecast)
            st.subheader("Prediction for 1 year")
            st.write(forecast.tail())
            st.plotly_chart(fig_fb)
            fig_comp = m.plot_components(forecast)
            st.write(fig_comp)

        def lstm_multi_model(trainX,trainY):
            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(trainY.shape[1]))
            model.compile(optimizer='adam', loss='mse')
            es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
            rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
            model.fit(trainX, trainY, shuffle=True, epochs=30, callbacks=[es, rlr], validation_split=0.2, verbose=-1, batch_size=256)
            return model

        def lstm_multi_impl():
            df = data
            train_dates = pd.to_datetime(df['Date'])
            cols = list(df)[1:6]
            
            df_for_training = df[cols].astype(float)
            scaler = StandardScaler()
            scaler = scaler.fit(df_for_training)
            df_for_training_scaled = scaler.transform(df_for_training)

            trainX = []
            trainY = []
            n_future = 1 
            n_past = 14 

            for i in range(n_past, len(df_for_training_scaled) - n_future +1):
                trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
                trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

            trainX, trainY = np.array(trainX), np.array(trainY)

            model = lstm_multi_model(trainX,trainY)

            us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
            n_past = 1
            n_days_for_prediction=120

            predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()

            prediction = model.predict(trainX[-n_days_for_prediction:]) 

            prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
            y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]

            forecast_dates = []
            for time_i in predict_period_dates:
                forecast_dates.append(time_i.date())
    
            df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
            df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

            original = df[['Date', 'Open']]
            original['Date']=pd.to_datetime(original['Date'])
            original = original.loc[original['Date'] >= '2010-1-1']

            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(x = original['Date'],y = original['Open'], name = 'Original'))
            fig_lstm.add_trace(go.Scatter(x = df_forecast['Date'],y = df_forecast['Open'], name = 'Predicted'))
            fig_lstm.layout.update( xaxis_rangeslider_visible = True)
            st.write(df_forecast.tail())
            st.plotly_chart(fig_lstm)



        model = st.radio("Select Your model",('LSTM Multi Variable','FB Prophet'))

        # if model == 'LSTM Single Variable':
        #     st.subheader("Prediction using LSTM Single Variable Model:")
        if model == 'LSTM Multi Variable':
            st.subheader("Prediction using LSTM Multi Variable Model:")
            lstm_multi_impl()
        if model == 'FB Prophet':
            st.subheader("Prediction using FB Prophet Model:")
            fbprophet_impl()
