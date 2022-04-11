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
import matplotlib.pyplot as plt


hide = """
<style>
#MainMenu {visibility: hidden;}
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
        
        if data.empty:
            st.write("Data Not Available")
        else:
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
        
            ##fbprophet:
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
                n_days_for_prediction=365

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
                ori = original[-1:]['Open'].values
                fore = df_forecast[0:1]['Open'].values
                diff = ori - fore
                incr = []
                for i in range(len(df_forecast)):
                    incr.append(df_forecast['Open'][i]+diff)

                df_inc = pd.DataFrame(incr)
                df_incr = pd.concat([df_forecast['Date'],df_inc],axis=1)

                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(x = original['Date'],y = original['Open'], name = 'Original'))
                fig_lstm.add_trace(go.Scatter(x = df_incr['Date'],y = df_incr[0], name = 'Predicted'))
                fig_lstm.layout.update( xaxis_rangeslider_visible = True)
                st.write(df_forecast.tail())
                st.plotly_chart(fig_lstm)
            
            
            plt.style.use('fivethirtyeight')

            st.cache()
            def SMA(data_frame,period=30,column = 'Close'):
                return data_frame[column].rolling(window=period).mean() 

            st.cache()
            def plot_sma(sma_df):
                plt.figure(figsize= (16, 8))
                plt.title('Close Price History', fontsize=1)
                plt.plot(data['Date'],data['Close'],alpha=0.5,label='Close')
                plt.plot(data['Date'],sma_df,alpha=0.5)
                plt.xlabel ('Date', fontsize=18)
                plt.ylabel('Close Price', fontsize=18)
                plt.legend(loc='upper left')
                st.pyplot(plt)
            
            def rsi():
                delta = data['Close'].diff()
                up = delta.clip(lower=0)
                down = -1*delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()   # ewm - Exponential weighted mass
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up/ema_down
                df_rsi = 100 - (100/(1 + rs))
                return df_rsi

            
            def atr():
                high_low = data['High'] - data['Low']
                high_cp = np.abs(data['High'] - data['Close'].shift())
                low_cp =  np.abs(data['Low'] - data['Close'].shift())
                df_temp = pd.concat([high_low, high_cp, low_cp],axis = 1)
                true_range = np.max(df_temp, axis = 1)
                average_true_range = true_range.rolling(13).mean()
                return average_true_range

            def plot_rsi_atr(rsi_atr):
                plt.figure(figsize= (16, 8))
                plt.title('Close Price History', fontsize=1)
                plt.plot(rsi_atr,alpha=0.5,)
                plt.legend(loc='upper right')
                st.pyplot(plt)
    
            options = st.radio("Select From the options:",('Analysis','Prediction'))

            if options == 'Analysis':
                analysis = st.radio("Simple Moving Average",('SMA9','SMA20','SMA50','SMA100','SMA200','RSI - Relative Strength Index','ATR - Average True Range'))
                if analysis == 'SMA9':
                    st.subheader("Simple Moving average for 9 days:")
                    plot_sma(SMA(data,9))
                if analysis == 'SMA20':
                    st.subheader("Simple Moving average for 9 days:")
                    plot_sma(SMA(data,20))
                if analysis == 'SMA50':
                    st.subheader("Simple Moving average for 9 days:")
                    plot_sma(SMA(data,50))
                if analysis == 'SMA100':
                    st.subheader("Simple Moving average for 9 days:")
                    plot_sma(SMA(data,100))
                if analysis == 'SMA200':
                    st.subheader("Simple Moving average for 9 days:")
                    plot_sma(SMA(data,200))
                if analysis == 'RSI - Relative Strength Index':
                    st.subheader("RSI - Relative Strength Index:")
                    plot_rsi_atr(rsi())
                if analysis == 'ATR - Average True Range':
                    st.subheader("ATR - Average True Range:")
                    plot_rsi_atr(atr())
            
            if options == 'Prediction':
                model = st.radio("Select Your model",('LSTM Multi Variable','FB Prophet'))

                if model == 'LSTM Multi Variable':
                    st.subheader("Prediction using LSTM Multi Variable Model:")
                    lstm_multi_impl()
                if model == 'FB Prophet':
                    st.subheader("Prediction using FB Prophet Model:")
                    fbprophet_impl()
