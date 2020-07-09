import pandas as pd

def enlarge(n):
    '''This function will multiply the input by 100'''
    return n*100

if __name__ == '__main__':
    y= int(input('choose a number: '))
    print(y,enlarge(y))