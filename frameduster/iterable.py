import inspect


def _correct_user(func):
    if inspect.isfunction(func):
        if not inspect.isgeneratorfunction(func):
            def corrected_func(*args, **kwargs):
                func(*args, **kwargs)
                yield None

            return corrected_func
        else:
            return func
    elif hasattr(func, '__iter__'):
        if not inspect.isgeneratorfunction(func.__iter__):
            def corrected_func(*args, **kwargs):
                func.__iter__(*args, **kwargs)
                yield None

            return corrected_func
        else:
            return func.__iter__
    else:
        raise Exception("Invalid service : cannot find __iter__ method in user service class.")


def _flush(func, *args, **kwargs):
    for _ in _generator(func, *args, **kwargs):
        pass


def _generator(func, *args, **kwargs):
    return _correct_user(func)(*args, **kwargs)
