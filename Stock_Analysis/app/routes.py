from flask import Blueprint, render_template, request, jsonify
from app.model_loader import (
    get_companies, 
    get_latest_stock_data, 
    get_stock_history, 
    predict_close,
    get_company_name
)
from app.database import (
    add_prediction, 
    get_history, 
    delete_history_item, 
    clear_all_history
)
from datetime import datetime

bp = Blueprint('main', __name__)

# --- HTML Page Routes ---

@bp.route('/')
def index():
    return render_template('landing.html')

@bp.route('/about')
def about():
    return render_template('about.html')

@bp.route('/dashboard')
def dashboard():
    companies = get_companies()
    return render_template('dashboard.html', companies=companies)

@bp.route('/analytics')
def analytics():
    companies = get_companies()
    return render_template('analytics.html', companies=companies)

@bp.route('/model')
def model_info():
    return render_template('model_info.html')

# --- API Endpoints ---

@bp.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json() or request.form
        
        company = data.get('company')
        open_val = float(data.get('open', 0))
        high_val = float(data.get('high', 0))
        low_val = float(data.get('low', 0))
        adj_close = float(data.get('adj_close', 0))
        volume = float(data.get('volume', 0))
        
        if not company:
            return jsonify({'error': 'Company ticker is required.'}), 400
            
        # Run prediction
        predicted_close = predict_close(company, open_val, high_val, low_val, adj_close, volume)
        
        # Calculate Trend
        trend = "BULLISH" if predicted_close >= open_val else "BEARISH"
        
        # Calculate dynamic confidence level based on relative error to adj_close
        if adj_close > 0:
            rel_error = abs(predicted_close - adj_close) / adj_close
            confidence = min(99.99, max(95.0, 100.0 - (rel_error * 100.0)))
        else:
            confidence = 98.50
            
        # Log to Database
        add_prediction(
            company_name=company,
            open_price=open_val,
            high_price=high_val,
            low_price=low_val,
            adj_close=adj_close,
            volume=volume,
            predicted_close=predicted_close,
            trend=trend,
            confidence=confidence
        )
        
        return jsonify({
            'success': True,
            'company': company,
            'open': open_val,
            'high': high_val,
            'low': low_val,
            'adj_close': adj_close,
            'volume': volume,
            'predicted_close': round(predicted_close, 2),
            'trend': trend,
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except ValueError as ve:
        return jsonify({'error': f'Invalid input format. Ensure numeric fields are numbers: {ve}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@bp.route('/api/stocks/<company>')
def api_stock_data(company):
    latest = get_latest_stock_data(company)
    history = get_stock_history(company, limit=100)
    
    if not latest:
        return jsonify({'error': f'No data found for company: {company}'}), 404
        
    return jsonify({
        'latest': latest,
        'history': history
    })

@bp.route('/api/history', methods=['GET'])
def api_get_history():
    history = get_history(limit=50)
    return jsonify({'history': history})

@bp.route('/api/history/<int:item_id>', methods=['DELETE'])
def api_delete_history(item_id):
    try:
        delete_history_item(item_id)
        return jsonify({'success': True, 'message': 'History item deleted.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    try:
        clear_all_history()
        return jsonify({'success': True, 'message': 'History cleared successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Error Handlers ---

@bp.app_errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@bp.app_errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500
