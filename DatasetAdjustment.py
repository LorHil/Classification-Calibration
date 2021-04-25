import numpy as np
import os, sys


if __name__ == '__main__':
    files = os.listdir("C:\\Users\\loreh\\Downloads\\Train")
    with open('train.csv', 'w') as csvfile:
        # given titles to the columns
        csvfile.write('Label')
        csvfile.write(',')
        csvfile.write('EmailText')
        csvfile.write('\n')

        for f in files:
            with open("C:\\Users\\loreh\\Downloads\\Train\\{}".format(f),'r') as txtfile:  
                if 'ham' in f:
                    csvfile.write('ham')
                else:
                    csvfile.write('spam')
                csvfile.write(',')

                tekst = txtfile.read()
                tekst = tekst.replace('\n',' ') 
                tekst = tekst.replace('"',"\'")
                
                csvfile.write('\"')
                csvfile.write(tekst)
                csvfile.write('\"')
                csvfile.write('\n')
            txtfile.close()
        csvfile.close()

    files2 = os.listdir("C:\\Users\\loreh\\Downloads\\Test")
    with open('test.csv', 'w') as csvfile:
        # given titles to the columns
        csvfile.write('EmailText')
        csvfile.write('\n')

        for f in files2:
            with open("C:\\Users\\loreh\\Downloads\\Test\\{}".format(f),'r') as txtfile:  
                tekst = txtfile.read()
                tekst = tekst.replace('\n',' ')
                tekst = tekst.replace('"',"\'")
                
                csvfile.write('\"')
                csvfile.write(tekst)
                csvfile.write('\"')
                csvfile.write('\n')
            txtfile.close()
        csvfile.close()
