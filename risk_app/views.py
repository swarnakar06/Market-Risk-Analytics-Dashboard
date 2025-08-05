from django.core.mail import EmailMessage
import io
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from django.shortcuts import render
from django.http import HttpResponse
from plotly.offline import plot
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ======== Home View (Fetch + Calculate + Visualize) =========
def home(request):
    risk_label = None
    accuracy = None
    suggestion = None

    data, columns, metrics, price_chart, daily_return_chart, cumulative_return_chart = [], [], {}, '', '', ''

    if request.method == 'POST':
        ticker = request.POST.get('ticker')
        start_date = request.POST.get('start')
        end_date = request.POST.get('end')

        if ticker and start_date and end_date:
            df = yf.download(ticker, start=start_date, end=end_date)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)
            if not df.empty:
                df['Daily Return'] = df['Close'].pct_change()
                df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
                df['High Risk'] = df['Daily Return'].apply(lambda x: 1 if abs(x) > 0.02 else 0)

                # SMA Signals
                df['SMA10'] = df['Close'].rolling(window=10).mean()
                df['SMA50'] = df['Close'].rolling(window=50).mean()
                df['Signal'] = 0
                df.loc[df['SMA10'] > df['SMA50'], 'Signal'] = 1
                df['Position'] = df['Signal'].diff()

                # Today's Buy/Sell/Hold Suggestion
                latest_signal = df['Signal'].iloc[-1]
                previous_signal = df['Signal'].iloc[-2] if len(df['Signal']) > 1 else 0
                if previous_signal == 0 and latest_signal == 1:
                    suggestion = "📈 BUY"
                elif previous_signal == 1 and latest_signal == 0:
                    suggestion = "🔻 SELL"
                else:
                    suggestion = "⏳ HOLD"

                features = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)
                labels = df['High Risk']

                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                latest_row = features.iloc[-1].values.reshape(1, -1)
                pred = model.predict(latest_row)[0]
                risk_label = "High Risk" if pred == 1 else "Low Risk"

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                metrics = {
                    'volatility': round(df['Daily Return'].std(), 4),
                    'avg_daily_return': round(df['Daily Return'].mean(), 4),
                    'cumulative_return': round(df['Cumulative Return'].iloc[-1], 4) if not df['Cumulative Return'].empty else 0.0
                }

                request.session['data'] = df.to_json()
                request.session['ticker'] = ticker
                request.session['accuracy'] = round(accuracy * 100, 2)

                data = df.round(6).fillna('').values.tolist()
                columns = df.columns.tolist()

                price_fig = go.Figure()
                price_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
                buy_signals = df[df['Position'] == 1]
                sell_signals = df[df['Position'] == -1]
                price_fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', marker=dict(color='green', size=8), name='Buy'))
                price_fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', marker=dict(color='red', size=8), name='Sell'))
                price_fig.update_layout(title='Stock Close Price Over Time with Buy/Sell Signals', xaxis_title='Date', yaxis_title='Close')
                price_chart = plot(price_fig, output_type='div')

                return_fig = go.Figure()
                return_fig.add_trace(go.Scatter(x=df['Date'], y=df['Daily Return'], mode='lines', name='Daily Return'))
                return_fig.update_layout(title='Daily Return Over Time', xaxis_title='Date', yaxis_title='Daily Return')
                daily_return_chart = plot(return_fig, output_type='div')

                cumulative_fig = go.Figure()
                cumulative_fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative Return'], mode='lines', name='Cumulative Return'))
                cumulative_fig.update_layout(title='Cumulative Return Over Time', xaxis_title='Date', yaxis_title='Cumulative Return')
                cumulative_return_chart = plot(cumulative_fig, output_type='div')

    return render(request, 'home.html', {
        'data': data,
        'columns': columns,
        'volatility': metrics.get('volatility'),
        'avg_daily_return': metrics.get('avg_daily_return'),
        'cumulative_return': metrics.get('cumulative_return'),
        'price_chart': price_chart,
        'daily_return_chart': daily_return_chart,
        'cumulative_return_chart': cumulative_return_chart,
        'risk_label': risk_label,
        'accuracy': round(accuracy * 100, 2) if accuracy is not None else None,
        'suggestion': suggestion
    })

# ===== CSV Export Function =====
def export_csv(request):
    if 'data' not in request.session:
        return HttpResponse("No data available for CSV export.")

    df = pd.read_json(request.session['data'])
    csv_data = df.to_csv(index=False)

    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="stock_data.csv"'
    return response

# ===== PDF Report Generation Function =====
def generate_pdf_report(df, ticker, accuracy):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        plt.title(f'{ticker} Close Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(df['Date'], df['Daily Return'], label='Daily Return', color='green')
        plt.title('Daily Return Over Time')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(df['Date'], df['Cumulative Return'], label='Cumulative Return', color='purple')
        plt.title('Cumulative Return Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 1))
        plt.axis('off')
        color = 'green' if accuracy and float(accuracy) >= 75 else 'red'
        if accuracy:
            plt.text(0, 0.5, f'📈 Prediction Accuracy: {accuracy}%', fontsize=12, color=color)
        else:
            plt.text(0, 0.5, 'Prediction Accuracy: Not Available', fontsize=12, color='black')
        plt.tight_layout()
        plt.savefig(pdf, format='pdf')
        plt.close()

    buffer.seek(0)
    return buffer

# ===== PDF Export View =====
def export_pdf(request):
    if 'data' not in request.session:
        return HttpResponse("No data available for PDF export.")

    df = pd.read_json(request.session['data'])
    ticker = request.session.get('ticker', 'Stock')
    accuracy = request.session.get('accuracy')

    buffer = generate_pdf_report(df, ticker, accuracy)

    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="stock_report.pdf"'
    return response

# ===== Email Report View =====
def email_report(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        if not email or 'data' not in request.session:
            return HttpResponse("Missing email or data.")

        df = pd.read_json(request.session['data'])
        ticker = request.session.get('ticker', 'Stock')
        accuracy = request.session.get('accuracy')

        buffer = generate_pdf_report(df, ticker, accuracy)

        email_msg = EmailMessage(
            subject=f"Market Report for {ticker}",
            body="📊 Please find attached your PDF market report with prediction accuracy and trading signals.",
            to=[email]
        )
        email_msg.attach("market_report.pdf", buffer.read(), 'application/pdf')
        email_msg.send()

        return HttpResponse("Email sent successfully!")
