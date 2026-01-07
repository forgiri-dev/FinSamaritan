import sys
try:
    import tensorflow as tf
    print('tensorflow module repr:', tf)
    print('__file__:', getattr(tf, '__file__', None))
    try:
        print('__version__:', tf.__version__)
    except Exception as e:
        print('__version__ access error:', e)
    print('keras in dir:', 'keras' in dir(tf))
    try:
        import tensorflow.keras as tkeras
        print('import tensorflow.keras ok; type:', type(tkeras))
    except Exception as e:
        print('import tensorflow.keras failed:', e)
except Exception as e:
    print('import tensorflow failed:', e)
    sys.exit(1)
print('sys.path top entries:')
for p in sys.path[:5]:
    print(' -', p)
print('site-packages path found in sys.path?:', any('site-packages' in str(p) for p in sys.path))
