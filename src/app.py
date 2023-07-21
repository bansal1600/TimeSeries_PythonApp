import dash
import pandas as pd
import numpy as np
from dash import html, dcc, callback, Output, Input
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

def plot_seasonal_decompose(result: DecomposeResult, dates: pd.Series = None, title: str = "Seasonal Decomposition"):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name='Observed'),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name='Trend'),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Seasonal'),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="lines", name='Residual'),
            row=4,
            col=1,
        )
        .update_layout(
            height=900, title=f'<b>{title}</b>', margin={'t': 100}, title_x=0.5, showlegend=False
        )
    )


def generate_stock_data(stock_name, start_date, end_date, initial_price, volatility):
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    price = initial_price
    stock_data = {
        'Date': date_range,
        'Stock': [stock_name] * len(date_range),
        'Price': [price],
    }

    for i in range(1, len(date_range)):
        price += np.random.normal(0, volatility)
        stock_data['Price'].append(max(0, price))

    return pd.DataFrame(stock_data)


# Generate synthetic stock price data for two imaginary stocks: "Stock A" and "Stock B"
start_date = '2023-01-01'
end_date = '2023-07-01'
initial_price_a = 100.0
initial_price_b = 200.0
volatility_a = 2.0
volatility_b = 3.0

stock_data_a = generate_stock_data('Stock A', start_date, end_date, initial_price_a, volatility_a)
stock_data_b = generate_stock_data('Stock B', start_date, end_date, initial_price_b, volatility_b)

# Concatenate the dataframes
df = pd.concat([stock_data_a, stock_data_b], ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'])


########### Initiate the app
app = dash.Dash(__name__)
server = app.server

########### Set up the layout
# Set up the app layout
app.layout = html.Div([
    html.H1("Stock Price Prediction and Seasonal Decomposition"),
    html.Label("Select a stock"),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[
            {'label': 'Stock A', 'value': 'Stock A'},
            {'label': 'Stock B', 'value': 'Stock B'},
        ],
        value='Stock A'
    ),
    html.Label("Select the number of forecast steps:"),
    dcc.Slider(
        id='steps-slider',
        min=1,
        max=60,
        step=1,
        value=30,
        marks={i: str(i) for i in range(1, 61)},
    ),
    html.Label("ARIMA Model Parameters:"),
    html.Label("p (order of the autoregressive part)"),
    dcc.Slider(
        id='p-slider',
        min=0,
        max=5,
        step=1,
        value=1,
        marks={i: str(i) for i in range(6)},
    ),
    html.Label("d (order of differencing)"),
    dcc.Slider(
        id='d-slider',
        min=0,
        max=2,
        step=1,
        value=0,
        marks={i: str(i) for i in range(3)},
    ),
    html.Label("q (order of the moving average part)"),
    dcc.Slider(
        id='q-slider',
        min=0,
        max=5,
        step=1,
        value=0,
        marks={i: str(i) for i in range(6)},
    ),
    html.Label("Select Model:"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'ARIMA', 'value': 'arima'},
            {'label': 'SARIMA', 'value': 'sarima'},
            {'label': 'Prophet', 'value': 'prophet'},
        ],
        value='arima'
    ),
    dcc.Graph(id='stock-graph'),
    dcc.Graph(id='seasonal-decompose-graph'),
    dcc.Graph(id='acf-graph'),
    dcc.Graph(id='pacf-graph')
])

# Define the callback function for updating the graphs
@app.callback(
    [Output('stock-graph', 'figure'), Output('seasonal-decompose-graph', 'figure'),
     Output('acf-graph', 'figure'), Output('pacf-graph', 'figure')],
    [Input('stock-dropdown', 'value'), Input('steps-slider', 'value'),
     Input('p-slider', 'value'), Input('d-slider', 'value'), Input('q-slider', 'value'),
     Input('model-dropdown', 'value')]
)
def update_graphs(stock, steps, p, d, q, model_selected):
    # Filter the data based on the selected stock
    filtered_df = df[df['Stock'] == stock]

    # Perform ARIMA, SARIMA, or Prophet model fitting and forecasting
    if model_selected == 'arima' or model_selected == 'sarima':
        model = ARIMA(filtered_df['Price'], order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
    elif model_selected == 'prophet':
        prophet_df = filtered_df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)['yhat'].tail(steps)

    # Create the figure object for stock price prediction
    fig_stock = go.Figure()

    # Add historical stock price data to the figure
    fig_stock.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Price'],
        name='Historical Data'
    ))

    # Add forecasted stock prices to the figure
    future_dates = pd.date_range(start=filtered_df['Date'].iloc[-1], periods=steps).tolist()
    fig_stock.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        name='Forecasted Data'
    ))

    # Set the figure layout for stock price prediction
    fig_stock.update_layout(
        title=f"Stock Price for {stock} - {model_selected.upper()} Model",
        xaxis_title='Date',
        yaxis_title='Price'
    )

    # Perform seasonal decomposition
    decomposition_result = seasonal_decompose(filtered_df['Price'], model='multiplicative', period=5)
    fig_decompose = plot_seasonal_decompose(decomposition_result, dates=filtered_df['Date'])

    # Compute ACF and PACF values
    acf_values, conf_int_acf = acf(filtered_df['Price'], nlags=20, alpha=0.05)
    pacf_values, conf_int_pacf = pacf(filtered_df['Price'], nlags=20, alpha=0.05)

    # Create the figure objects for ACF and PACF plots
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(
        x=np.arange(len(acf_values)),
        y=acf_values,
        name='ACF'
    ))
    fig_acf.add_trace(go.Scatter(
        x=np.arange(len(conf_int_acf)),
        y=conf_int_acf[:, 0],
        mode='lines',
        line=dict(dash='dash'),
        name='Confidence Interval Lower'
    ))
    fig_acf.add_trace(go.Scatter(
        x=np.arange(len(conf_int_acf)),
        y=conf_int_acf[:, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Confidence Interval Upper'
    ))
    fig_acf.update_layout(
        title=f"Autocorrelation Function (ACF) for {stock}",
        xaxis_title='Lags',
        yaxis_title='ACF Value'
    )

    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Bar(
        x=np.arange(len(pacf_values)),
        y=pacf_values,
        name='PACF'
    ))
    fig_pacf.add_trace(go.Scatter(
        x=np.arange(len(conf_int_pacf)),
        y=conf_int_pacf[:, 0],
        mode='lines',
        line=dict(dash='dash'),
        name='Confidence Interval Lower'
    ))
    fig_pacf.add_trace(go.Scatter(
        x=np.arange(len(conf_int_pacf)),
        y=conf_int_pacf[:, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Confidence Interval Upper'
    ))
    fig_pacf.update_layout(
        title=f"Partial Autocorrelation Function (PACF) for {stock}",
        xaxis_title='Lags',
        yaxis_title='PACF Value'
    )

    return fig_stock, fig_decompose, fig_acf, fig_pacf

if __name__ == '__main__':
    app.run_server()
