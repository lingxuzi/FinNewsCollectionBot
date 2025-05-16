from api import get_financial_metrics
from api import get_financial_statements


if __name__  == '__main__':
    metrics = get_financial_metrics('300024')

    print(metrics)