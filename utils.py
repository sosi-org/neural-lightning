



def is_linux():
    import platform
    s = platform.system()
    return {
        'Linux': True,
        'Darwin': False,
        'Windows': False,
    }[s]

def is_mac():
    import platform
    s = platform.system()
    return {
        'Linux': False,
        'Darwin': True,
        'Windows': False,
    }[s]

def linux_plot_issue():
    if is_linux():
        import matplotlib
        matplotlib.use('TkAgg')
        # matplotlib.use('agg')
        print('backend:', matplotlib.get_backend())
        # matplotlib.hold(true) # deprecated



# [1] From https://github.com/sosi-org/neural-networks-sandbox/blob/4cba7254b52551c9bd4235e2f6d41feb3e1c8447/glyphnet/utils/pcolor.py
