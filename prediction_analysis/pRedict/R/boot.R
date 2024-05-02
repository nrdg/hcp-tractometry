boot <- function(df, n_boots) { 
    
    df %>% nest(data = everything(), .by=Family_ID) %>%
              dplyr::slice_sample(n = n_boots, replace=TRUE) %>% 
              unnest( cols=c(data), names_repair = "minimal") 
    
    return(df) } 