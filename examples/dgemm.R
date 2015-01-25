a <- matrix(1:16, ncol = 2048, nrow = 2048)
b <- matrix(1:16, ncol = 2048, nrow = 2048)
c <- matrix(1:16, ncol = 2048, nrow = 2048)

start <- proc.time()[3]
c <- a %*% b
end <- proc.time()[3]

end - start

g <- 2*2048*2048*2048/(end - start)
"Gflops"
g*1e-9
