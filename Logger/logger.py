import logging
from time import asctime
from functools import wraps

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(filename='Logger/processing_logs.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f'{func.__name__} started')
        try:
            result = func(*args, **kwargs)
            logger.info(f'{func.__name__} finished successfully')
            return result
        except Exception as e:
            logger.error(f'Error in {func.__name__}: {str(e)}')
            raise
    return wrapper