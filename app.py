from flask import Flask, render_template, request, jsonify
from sentiment_analyzer import SentimentAnalyzer
import os

app = Flask(__name__)
analyzer = SentimentAnalyzer()


if not analyzer.load_model():
    print("UWAGA: Model nie został znaleziony!")
    print("Uruchom najpierw sentiment_analyzer.py aby wytrenować model.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        review_text = request.form.get('review', '').strip()
        
        if not review_text:
            return jsonify({
                'error': 'Proszę wprowadzić tekst recenzji'
            })
        
        if len(review_text) < 10:
            return jsonify({
                'error': 'Recenzja musi mieć co najmniej 10 znaków'
            })
        
        
        result = analyzer.predict_sentiment(review_text)
        
        if result is None:
            return jsonify({
                'error': 'Błąd podczas analizy tekstu. Sprawdź czy model został wytrenowany.'
            })
        
       
        sentiment_pl = 'pozytywny' if result['sentiment'] == 'positive' else 'negatywny'
        confidence_percent = round(result['confidence'] * 100, 1)
        
        response = {
            'sentiment': result['sentiment'],
            'sentiment_pl': sentiment_pl,
            'confidence': confidence_percent,
            'positive_prob': round(result['probabilities']['positive'] * 100, 1),
            'negative_prob': round(result['probabilities']['negative'] * 100, 1)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Wystąpił błąd: {str(e)}'
        })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint dla zewnętrznych aplikacji"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Brak wymaganego pola "text" w JSON'
            }), 400
        
        review_text = data['text'].strip()
        
        if not review_text:
            return jsonify({
                'error': 'Tekst nie może być pusty'
            }), 400
        
        result = analyzer.predict_sentiment(review_text)
        
        if result is None:
            return jsonify({
                'error': 'Błąd modelu'
            }), 500
        
        return jsonify({
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Utworzono katalog 'templates'")
    
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Utworzono katalog 'static'")
    
    app.run(debug=True, host='0.0.0.0', port=5000)