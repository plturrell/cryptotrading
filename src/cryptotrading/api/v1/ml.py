"""
ML API v1 - Machine Learning endpoints
"""
from flask import request
from flask_restx import Namespace, Resource, fields
from ...services.ml_service import EnhancedMLService

ml_ns = Namespace('ml', description='Machine Learning operations')

# Models for documentation
prediction_model = ml_ns.model('MLPrediction', {
    'predicted_price': fields.Float(description='Predicted price'),
    'current_price': fields.Float(description='Current price'),
    'price_change_percent': fields.Float(description='Predicted price change %'),
    'confidence': fields.Float(description='Prediction confidence'),
    'model_type': fields.String(description='Model type used'),
    'horizon': fields.String(description='Prediction horizon')
})

training_model = ml_ns.model('TrainingStatus', {
    'status': fields.String(description='Training status'),
    'message': fields.String(description='Status message'),
    'symbols': fields.List(fields.String, description='Symbols being trained'),
    'model_types': fields.List(fields.String, description='Model types being trained')
})

# Initialize service
ml_service = EnhancedMLService()


@ml_ns.route('/predict/<string:symbol>')
class MLPrediction(Resource):
    @ml_ns.doc('get_prediction')
    @ml_ns.marshal_with(prediction_model)
    @ml_ns.param('horizon', 'Prediction horizon (default: 24h)')
    @ml_ns.param('model_type', 'Model type (lstm, transformer, xgboost, ensemble)')
    def get(self, symbol):
        """Get ML price prediction for a cryptocurrency"""
        horizon = request.args.get('horizon', '24h')
        model_type = request.args.get('model_type', None)
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                ml_service.get_prediction(symbol, horizon, model_type)
            )
            return result
        except Exception as e:
            ml_ns.abort(500, f"Prediction failed: {str(e)}")
        finally:
            loop.close()


@ml_ns.route('/predict/batch')
class MLBatchPrediction(Resource):
    @ml_ns.doc('get_batch_predictions')
    @ml_ns.expect(ml_ns.model('BatchRequest', {
        'symbols': fields.List(fields.String, required=True, description='List of symbols'),
        'horizon': fields.String(description='Prediction horizon'),
        'model_type': fields.String(description='Model type')
    }))
    def post(self):
        """Get ML predictions for multiple cryptocurrencies"""
        data = request.get_json() or {}
        symbols = data.get('symbols', ['BTC', 'ETH'])
        horizon = data.get('horizon', '24h')
        model_type = data.get('model_type', None)
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                ml_service.get_batch_predictions(symbols, horizon, model_type)
            )
            return results
        except Exception as e:
            ml_ns.abort(500, f"Batch prediction failed: {str(e)}")
        finally:
            loop.close()


@ml_ns.route('/performance/<string:symbol>')
class MLModelPerformance(Resource):
    @ml_ns.doc('get_model_performance')
    @ml_ns.param('horizon', 'Prediction horizon')
    def get(self, symbol):
        """Get ML model performance metrics"""
        horizon = request.args.get('horizon', '24h')
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                ml_service.get_model_performance(symbol, horizon)
            )
            return result
        except Exception as e:
            ml_ns.abort(500, f"Performance retrieval failed: {str(e)}")
        finally:
            loop.close()


@ml_ns.route('/train')
class MLTraining(Resource):
    @ml_ns.doc('trigger_training')
    @ml_ns.marshal_with(training_model)
    @ml_ns.expect(ml_ns.model('TrainingRequest', {
        'symbols': fields.List(fields.String, description='Symbols to train'),
        'model_types': fields.List(fields.String, description='Model types to train')
    }), validate=False)
    def post(self):
        """Trigger ML model training"""
        data = request.get_json() or {}
        symbols = data.get('symbols', None)
        model_types = data.get('model_types', None)
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                ml_service.trigger_training(symbols, model_types)
            )
            return result
        except Exception as e:
            ml_ns.abort(500, f"Training trigger failed: {str(e)}")
        finally:
            loop.close()


@ml_ns.route('/features/<string:symbol>')
class MLFeatures(Resource):
    @ml_ns.doc('get_feature_importance')
    @ml_ns.param('model_type', 'Model type for feature importance')
    def get(self, symbol):
        """Get ML feature importance for interpretability"""
        model_type = request.args.get('model_type', 'xgboost')
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                ml_service.get_feature_importance(symbol, model_type)
            )
            return result
        except Exception as e:
            ml_ns.abort(500, f"Feature importance retrieval failed: {str(e)}")
        finally:
            loop.close()


@ml_ns.route('/retrain/auto')
class AutoRetraining(Resource):
    @ml_ns.doc('run_auto_retrain')
    def post(self):
        """Run automated model retraining check"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(ml_service.run_automated_retraining())
            return result
        except Exception as e:
            ml_ns.abort(500, f"Auto retraining failed: {str(e)}")
        finally:
            loop.close()
