import uvicorn
import logging
from webserver import *

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(filename='logs.txt', level=logging.INFO)
    uvicorn.run('main:app', reload=True)
    logger.info('stared')

if __name__ == '__main__':
    main()