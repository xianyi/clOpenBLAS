#!/usr/bin/python

import time
import numpy
from numpy.random import randn

def run_sgemm():

	N=3840*2
	
	print("sgemm %dx%d" % ( N, N ));

	A = randn(N,N).astype('float32')
	B = randn(N,N).astype('float32')

	start = time.time();
	ref = numpy.dot(A,B)
	end = time.time()
	
	timediff = (end -start) 
	gflops = ( 2*N*N*N) / timediff
	gflops *= 1e-9

	print("Time:\t\t%f" % timediff);
	print("GFlops:\t\t%f" % gflops);


if __name__ == "__main__":
	run_sgemm()

