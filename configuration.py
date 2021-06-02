import configparser

def ini_parser(file_name):
    ''' parser for configuration.ini file'''
    config = configparser.ConfigParser()
    config.read(file_name)
    sections = config.sections()
    configuration = {}
    for section in sections:
        configuration[section] = dict(config[section])
    return configuration