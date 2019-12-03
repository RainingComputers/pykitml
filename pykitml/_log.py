# =====================================================
# = This module contains classes and helper functions =
# = for printing and plotting logged logs.            =
# =====================================================

def beginlog():
    '''
    Goes to a new line and hides cursor.
    '''
    # Hide cursor
    print('\033[?25l', flush=True)

def progressbar(completed, total, name='Progress', length=40):
    '''
    Display a progress bar.
    Parameters
    ----------
    completed : int
        Amount of tasks completed.
    total : int
        Total number of tasks to complete.
    name : str
        Name for the progress bar.
    length : int
        Length of the progrss bar.
    '''
    # Create bar string
    hashes = int((completed/total)*length)
    dots = length-hashes
    bar = ' [' + '#'*hashes + '.'*dots + '] '
    # Create progress string
    progress = '{}/{} '.format(completed, total)
    # Print the progress bar
    pstr = name + ':' + bar + progress
    print(pstr, flush=True)

def cursorup(lines):
    print('\033[{}A'.format(lines), end='', flush=True)

def endlog(lines):
    '''
    Goes to a newline and shows cursor.
    '''
    # Show cursor
    print('\033[?25h', end='', flush=True)
    # Goto new line
    print('\033[{}B'.format(lines), end='', flush=True)
    