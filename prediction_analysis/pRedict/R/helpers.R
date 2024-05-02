check_scaling <- function(df, set_eps=1e-10) { 
    stopifnot(df %>% colMeans() %>% sort() %>% isZero(eps=set_eps)) 
    
    stopifnot( df %>% 
          apply(2, sd) %>% 
          sapply( function(x){ x -1 })  %>%  
          isZero(eps=set_eps) ) 
    
    } 