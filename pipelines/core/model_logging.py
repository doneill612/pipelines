def info(*message) -> None:
    print('[INFO]', *message)

def fatal(exception_type: str, *message) -> None:
    print('[FATAL]', 'A fatal exception occured.')
    if exception_type == 'assertion':
        raise AssertionError(message)
    elif exception_type == 'ni':
        raise NotImplementedError(message)
    else:
        raise Exception(message)
