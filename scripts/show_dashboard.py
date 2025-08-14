import requests
import json

def show_dashboard():
    try:
        response = requests.get('http://localhost:5001/observability/dashboard?hours=1')
        data = response.json()
        
        print('🎯 REX TRADING OBSERVABILITY DASHBOARD')
        print('=' * 50)
        print(f'📊 Health Status: {data["health"]["tracer"]["status"]}')
        print(f'📈 Total Metrics: {data["metrics_summary"]["total_metrics"]}')  
        print(f'❌ Total Errors: {data["errors_summary"]["total_errors"]}')
        print('\n📋 System Components:')
        for component, status in data['health'].items():
            if component != 'timestamp':
                print(f'   {component}: {status.get("status", "unknown")}')
        
        if data['metrics_summary']['total_metrics'] > 0:
            print('\n📊 Recent Metrics:')
            for name, metric in list(data['metrics_summary']['metrics'].items())[:5]:
                print(f'   {name}: {metric.get("count", 0)} points')
        
        print('\n🌐 Dashboard Available At:')
        print('   HTML: http://localhost:5001/observability/dashboard.html')
        print('   JSON: http://localhost:5001/observability/dashboard')
        print('   Health: http://localhost:5001/observability/health')
        
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == '__main__':
    show_dashboard()