from qmpy import *
import pandas as pd 

def red_vasp():
	fe = Element.objects.get(symbol='Fe')
	path = 'analysis/vasp/files/convergence'
	calc = Calculation.read(INSTALL_PATH+'/'+path)
	print calc.read_energies()

	calc.get_outcar()
	print len(calc.outcar)


