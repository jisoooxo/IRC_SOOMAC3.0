import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/pc/irc_ws/irc_ws/install/irc_control_pkg'
