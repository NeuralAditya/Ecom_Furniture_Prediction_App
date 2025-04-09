from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io
import matplotlib
matplotlib.use('Agg')
import pandas as pd

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        title = request.form['productTitle']
        original_price = float(request.form['originalPrice'])
        price = float(request.form['price'])
        tag = request.form['tagText']

        tag_map = {'Free shipping': 0, '+Shipping: $5.09': 1, 'others': 2}
        tag_encoded = tag_map.get(tag, 2)

        discount = ((original_price - price) / original_price) * 100 if original_price != 0 else 0

        title_tfidf = tfidf.transform([title]).toarray()
        features = np.concatenate([[price, original_price, discount, tag_encoded], title_tfidf[0]])

        feature_df = pd.DataFrame([features], columns=model.feature_names_in_)
        prediction = model.predict(feature_df)[0]

        # Dynamic Graph 1: Prediction Breakdown
        importances = model.feature_importances_
        top_n = 10
        top_indices = np.argsort(importances)[-top_n:][::-1]

        top_features = [model.feature_names_in_[i] for i in top_indices]
        contributions = [features[i] * importances[i] * 1000 for i in top_indices]

        plt.figure(figsize=(6, 4))
        plt.barh(top_features[::-1], contributions[::-1], color='orange')
        plt.xlabel("Feature Contribution")
        plt.title("Prediction Breakdown")
        plt.tight_layout()
        plt.savefig("static/graphs/prediction_breakdown.png")
        plt.close()

        # Dynamic Graph 2: Price Impact Visualization
        price_range = np.linspace(price * 0.8, price * 1.2, 20)
        simulated_preds = []

        for p in price_range:
            d = ((original_price - p) / original_price) * 100 if original_price != 0 else 0
            f = np.concatenate([[p, original_price, d, tag_encoded], title_tfidf[0]])
            df = pd.DataFrame([f], columns=model.feature_names_in_)
            pred = model.predict(df)[0]
            simulated_preds.append(pred)

        plt.figure(figsize=(6, 4))
        plt.plot(price_range, simulated_preds, marker='o', color='green')
        plt.xlabel("Price")
        plt.ylabel("Predicted Sales")
        plt.title("Price Impact on Predicted Sales")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("static/graphs/price_impact.png")
        plt.close()

        return render_template('result.html', prediction=int(prediction), graph='predicted_this_product.png')

    except Exception as e:
        return f"Error occurred during prediction: {e}"

@app.route('/download_pdf')
def download_pdf():
    try:
        prediction = request.args.get('prediction', 'N/A')

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 800, "🪑 Furniture Sales Prediction Report")

        # Prediction
        c.setFont("Helvetica", 12)
        c.drawString(100, 770, f"Predicted Items Sold: {prediction}")

        # Embedding graphs
        graphs = [
            ('Feature Importance', 'static/graphs/feature_importance.png'),
            ('Predicted vs Actual', 'static/graphs/predicted_vs_actual.png'),
            ('Price vs Predicted Sales', 'static/graphs/price_vs_predicted.png'),
            ('Distribution of Sold', 'static/graphs/distribution_sold.png'),
            ('Sales by Shipping Tag', 'static/graphs/sales_by_tag.png'),
            ('Discount % vs Sold', 'static/graphs/discount_vs_sold.png'),
            ('Top Product Keywords', 'static/graphs/top_keywords.png'),
            ('Price Distribution', 'static/graphs/price_distribution.png'),
            ('Sold by Price Bracket', 'static/graphs/sold_by_price_bracket.png'),
            ('Sold by Discount Bin', 'static/graphs/sales_by_discount_bin.png'),
            ('Tag Text Breakdown', 'static/graphs/tag_text_breakdown.png')
        ]

        y = 720
        for title, path in graphs:
            c.drawImage(path, 100, y - 200, width=300, height=180)
            y -= 220
            if y < 150:
                c.showPage()
                y = 750

        if y < 100:
            c.showPage()

        # Footer copyright
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(
            100,
            40,
            u"\u00A9 2025 Aditya Arora. All rights reserved. https://www.linkedin.com/in/NeuralAditya"
        )

        c.save()
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name='prediction_report.pdf', mimetype='application/pdf')

    except Exception as e:
        return f"Error occurred during PDF generation: {e}"


if __name__ == '__main__':
    app.run(debug=True)
