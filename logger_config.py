import logging

def setup_logger():
    logging.basicConfig(filename='logs/project.log', level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(message)s')
