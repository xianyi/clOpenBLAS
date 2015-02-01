#!/usr/bin/python

import time
import numpy
from numpy.random import randn

def run_dgemm():

	N=3840*2
	
	print("dgemm %dx%d" % ( N, N ));

	A = randn(N,N).astype('float64')
	B = randn(N,N).astype('float64')

	start = time.time();
	ref = numpy.dot(A,B)
	end = time.time()
	
	timediff = (end -start) 
	gflops = ( 2*N*N*N) / timediff
	gflops *= 1e-9

	print("Time:\t\t%f" % timediff);
	print("GFlops:\t\t%f" % gflops);


if __name__ == "__main__":
	run_dgemm()

