clr <- function() cat("\014");
library(stats)
library(ggplot2)
library(dplyr)
library(utils)

##Numeric differentiation
{
  install.packages('numDeriv')
}


## Symoblic differentation
{
  ##D
  {
    f <- expression(sin(x))
    df <- D(f, 'x'); df
    x <- 0
    eval(df)
  }
  
  ##deriv
  {
      f <- expression(x^3)
      dfdx <- deriv(f, 'x'); dfdx
      x <- -1:2
      eval(dfdx)
      
      f2 <- expression(factorial(x))
      df2dx <- deriv(f2, 'x'); df2dx
      x <- 1:10
      dx <- eval(df2dx); dx
      
      clr()
      
      f3 <- expression(cos(x) * exp(x/y) * (x/y))
      df3dx <- deriv(f3, c('x','y')); df3dx
      x <- 1:10
      y <- 11:20
      eval(df3dx)
      
      hf3dx <- deriv(f3, c('x', 'y'), hessian = TRUE)
      x <- 1
      y <- 1
      eval(hf3dx)
      
      f4 <- expression( exp(cos(sin(x*y/z))) +
                        log(z^(y^z)) +
                        log(sqrt(x^y - z^x)) +
                        tan(cos(x/y))
      )
      x <- 4; y <- 1; z <- 1
      times = 0
      for(i in 1:10){
        ptm <- Sys.time()
        hf4dx <- deriv(f4, c('x'));
        hf4dy <- deriv(f4, c('y'));
        hf4dz <- deriv(f4, c('x'));
        invisible(eval(hf4dx));
        invisible(eval(hf4dy));
        invisible(eval(hf4dz));
        time <- (Sys.time() - ptm)
        times <- time + times
      }; times/10
  }
  
  # format(): format expressions from strings
  # Simplify()
  # Deriv package add aditional functionalities to deriv
}

# Automatic differentiation
{
  #Julia
  {
    #install.packages("devtools")
    #library(devtools)
    #devtools::install_github("Non-Contradiction/autodiffr")
    #devtools::install_github("Non-Contradiction/JuliaCall")
    library(JuliaCall)
    library(autodiffr)
    
    #Setting Julia 
    ad_setup()
    f <- function(x) sum(x^2L)
    
    #gradient of the function
    ad_grad(f, c(2, 3), mode = c("reverse"))
    ad_grad(f, c(2, 3), mode = "forward")
    g <- makeGradFunc(f)
    g(c(2, 3))
    
    #Hessian calculation
    ad_hessian(f, c(2, 3))
    h <- makeHessianFunc(f)
    h(c(2, 3))
    
    #Jacobian calculation 
    f <- function(x) x^2
    ad_jacobian(f, c(2, 3))
    j <- makeJacobianFunc(f)
    j(c(2,10))
    
    
    #Working with multiple arguments
    f <- function(a, b, c){
      return(a * b^2 * c^3 + sin(a*c) - cos(b/c))
    }
    ad_grad(f, 2, b = 1, c = 1)
    g <- makeGradFunc(f, mode="reverse")
    g(1,2,3)
    
    
    ad_hessian(f, b = 2,c = 3)
    g <- makeHessianFunc(f)
    g(1)
    
    g <- makeJacobianFunc(f)
    g(c(1,1,0))
  }
}