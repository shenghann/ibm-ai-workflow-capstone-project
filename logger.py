from asyncio import run
import logging
from logging import handlers
from pathlib import Path

LOG_ROOT = Path('log')
LOG_FILES = {
    'pred' : LOG_ROOT / 'predict.log',
    'train' : LOG_ROOT / 'train.log'
}

def setup_logging():
    """Enable logging, set date format, logging level, log file paths, and file mode."""
    loggers = []
    for logger_name in ['train','pred']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        handler = logging.handlers.RotatingFileHandler(LOG_FILES[logger_name], mode='a', maxBytes=10485760, backupCount=10)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        loggers.append(logger)
    
    return loggers

train_logger, pred_logger = setup_logging()

def log_train(model_name, data_shape,eval_test, runtime,
                         model_version, model_version_note):
    text = f'Training completed for {model_name}. Data shape {data_shape}, Eval test {eval_test}, Runtime {runtime:.2}, ModelVer {model_version}, Note {model_version_note}'
    train_logger.info(text)

def log_pred(y_pred, query, runtime, model_version):
    text = f'Prediction completed. Predicted: {y_pred}, Query {query}, Runtime: {runtime:.2}, ModelVer: {model_version}'
    pred_logger.info(text)