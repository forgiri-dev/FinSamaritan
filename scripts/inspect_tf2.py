import importlib
try:
    import tensorflow as tf
    print('tensorflow repr:', tf)
    print('__path__:', list(getattr(tf, '__path__', [])))
    try:
        import pkgutil
        for finder, name, ispkg in pkgutil.iter_modules(tf.__path__):
            print('module in tensorflow package:', name, 'ispkg=', ispkg)
    except Exception as e:
        print('pkgutil error:', e)
except Exception as e:
    print('import error:', e)
