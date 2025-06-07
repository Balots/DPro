import logging
from time import asctime

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(filename='Logger/processing_logs.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def decorator(func):
    message_1 = f'{func.__name__} started.'
    message_2 = f'{func} finished.'
    log_file = open('processing_logs.log', 'w')
    def wrapper(*args):
        logging.info(message_1)
        log_file.write(f'{asctime()}\t{message_1}\n')

        result = func(*args)
        log_file.write(f'{asctime()}\t{func} result: {result}\n')

        logging.info(message_2)
        log_file.write(f'{asctime()}\t{message_2}\n\n\n')
        log_file.close()
        return result
    return wrapper