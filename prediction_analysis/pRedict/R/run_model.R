run_model <- function(function_name='lasso', 
                      x_train, y_train, 
                      x_test, y_test, 
                      alpha=1, theta=1, ...) { 

    if (function_name == 'pcr') { 
        
        pca_results <- prcomp(x_train, 
                              scale.=FALSE, 
                              center=FALSE) 
        
        pca_x_train <- pca_results$x
        pca_x_test <- as.matrix(x_test) %*% pca_results$rotation 
        #sub in predict and check 
        
        pcr_model <- cv.glmnet(x=as.matrix(pca_x_train),
                               y=as.matrix(y_train), 
                               alpha=alpha)


        rsq <- R2(predict(pcr_model, 
                          newx=as.matrix(pca_x_test),
                           s="lambda.min"), 
                   as.matrix(y_test))

        return(list(model=pcr_model, rsq=rsq) )

    } 

    else if (function_name == 'lasso') {
        lasso_model <- cv.glmnet(x=as.matrix(x_train),
                                 y=as.matrix(y_train), 
                                 alpha=1)


        r2 <- R2(predict(lasso_model, 
                         newx=as.matrix(x_test),
                         s="lambda.min"), 
                 as.matrix(y_test))

        return(list(model=lasso_model, rsq=r2) )
        
    }

    else if (function_name == 'pcLasso') { 

        pcL_model <- cv.pcLasso(x=as.matrix(x_train), 
                           y=as.matrix(y_train), 
                           theta=theta) 

        r2 <- R2(predict(pcL_model, 
                         xnew=as.matrix(x_test),
                         s="lambda.min"), 
                 as.matrix(y_test))
        
        return(list(model=pcL_model, rsq=r2) ) 
    }
    
    else if (function_name == 'sgl') { 

        groups=c()
        
        if (length(sel_metrics) > 1) { 
            num_groups = 96 } #num_groups = n tracts x n metrics
        else { 
            num_groups = 24 } 
        for (ii in 1:num_groups) { 
            groups <- append(groups, rep(ii, 100)) #100 = num_nodes
        } 

        sgl_model <- cv.sparsegl(y = as.matrix(y_train), 
                                 x = as.matrix(x_train),
                                 group=groups) 

        r2 <- R2(predict(sgl_model, 
                         newx=as.matrix(x_test), 
                         s='lambda.min'), 
                 as.matrix(y_test))

        return(list(model=sgl_model, rsq=r2) ) } 

    } 