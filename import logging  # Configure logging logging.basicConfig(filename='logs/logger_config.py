import logging

# Configure logging
logging.basicConfig(filename='logs/project.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(message)s')
