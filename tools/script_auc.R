MWWStatVar=function(y,x,y0=NA,x0=NA){
         # This function returns the Mann-Whitney statistic and its unbiased, 
         # non-negative variance estimate. Arguments y and x are the two samples.
         # If y0 and x0 are provided, the difference of two Mann-Whitney
         # statistics and the variance estimate of that difference is output.
         k <- function(x,y) (sign(y-x)+1)/2;  # The indicator kernel function
         m <- length(y); n <- length(x);  X <- outer(x,y,k);
         if (length(y)==length(y0) && length(x)==length(x0)) X<- X-outer(x0,y0,k)
         MSA <- m*var(rowMeans(X)) ;  MSB <- n*var(colMeans(X)) ;  
         MST <- var(as.vector(X))
         ev <- ((n*m-1)*MST-(n-1)*MSA-(m-1)*MSB)/((n-1)*(m-1))
         return(c(mean(X), (MSA+MSB-ev)/(m*n)))
}

auc_cal = function(scanner_results){
          # This function return the auc value and its SD.
          # Input: the ground truth and predicted results
          y.scanner_results <- scanner_results[scanner_results[,1]==1,2];
          x.scanner_results <- scanner_results[scanner_results[,1]==0,2]; 
          roc.scanner_results <- MWWStatVar(y.scanner_results, x.scanner_results); 
          auc.scanner_results <- roc.scanner_results[1]; sd.scanner_results <- sqrt(roc.scanner_results[2]); 
          return(auc.scanner_results, sd.scanner_results) }

auc_diff = function(auc_with_sd_1, auc_with_sd_2){
          # This function return the difference of two auc and it CI
          # Input: auc and its SD from the function auc_cal
          auc.diff <- auc_with_sd_1[1] - auc_with_sd_2[1]; 
          dauc.low <- auc.diff - 1.96*sqrt(auc_with_sd_1[2]*auc_with_sd_1[2]+auc_with_sd_2[2]*auc_with_sd_2[2]); 
          dauc.high <- auc.diff + 1.96*sqrt(auc_with_sd_1[2]*auc_with_sd_1[2]+auc_with_sd_2[2]*auc_with_sd_2[2]); 
          return(c(auc.diff, dauc.low, dauc.high))}

SD_cal <- function(upper_ci,lower_ci){sd <- (upper_ci - lower_ci)/1.96; return (sd)}

CI_cal <- function(mean, sd){upper_ci <- mean + 1.96*sd;lower_ci <- mean - 1.96*sd; return (c(lower_ci, upper_ci))}

setwd("/Users/liw17/Documents/WSI")
scanner <- read.csv("WSI_scanner_test.txt", header=F)
dnn.out <- read.csv("RF/RF_results/reference_all_final_results_updated0517.csv")

# subgroup analysis for Method_I
## read in the data
### read in the data: overall data
Method_I <- dnn.out[as.integer(scanner$V2)==1, c(7,8)]
### read in the data: histech scanner
histech_Method_I <- dnn.out[as.integer(scanner$V2)==1, c(7,8)]
### read in the data: hamamstu scanner
hamamatsu_Method_I <- dnn.out[as.integer(scanner$V2)==2,c(7,8)]

## calculate the auc with its sd
### calculate the auc with sd for the overall data
auc_with_sd_I <- auc_cal(Method_I)
### calculate the auc with sd for the histech scanner
auc_with_sd_histech_I <- auc_cal(histech_Method_I)
### calculate the auc with sd for the hamamastu scanner
auc_with_sd_hamamatsu_I <- auc_cal(hamamatsu_Method_I)
### calculate the auc difference between histech and hamamastu scanners
auc_diff_Method_I <- auc_diff(auc_with_sd_histech_I, auc_with_sd_hamamatsu_I)

## The confidence interval (CI) of the overall dataset in Method I
CI_method_I <- CI_cal(0.916, 0.03)
### The confidence interval (CI) of histech scanner in Method I
CI_histech_method_I <- CI_cal(0.924, 0.03)
### The confidence interval (CI) of hamamastu scanner in Method I
CI_hamamatsu_method_I <- CI_cal(0.9133, 0.039)


# subgroup analysis for Model with color noise only
## read in the data
### read in the data: overall data
Method_II_noise_only <- dnn.out[c(7,10)]
### read in the data: histech scanner
histech_Method_II_noise_only <- dnn.out[as.integer(scanner$V2)==1, c(7,10)]
### read in the data: hamamstu scanner
hamamatsu_Method_II_noise_only <- dnn.out[as.integer(scanner$V2)==2,c(7,10)]

## calculate the auc with its sd
### calculate the auc with sd for the overall data
auc_with_sd_noise_only <- auc_cal(Method_II_noise_only)
### calculate the auc with sd for the histech scanner
auc_with_sd_histech_II_noise_only <- auc_cal(histech_Method_II_noise_only)
### calculate the auc with sd for the hamamastu scanner
auc_with_sd_hamamatsu_II_noise_only <- auc_cal(hamamatsu_Method_II_noise_only)
### calculate the auc difference between histech and hamamastu scanners
auc_diff_Method_II_noise_only <- auc_diff(auc_with_sd_hamamatsu_II_noise_only, auc_with_sd_histech_II_noise_only)

## calculate the confidence interval
### The confidence interval (CI) of the overal dataset in Model with color noise only
CI_method_II <- CI_cal(0.94209184, 0.02184416)
### The confidence interval (CI) of histech scanner for Model with color noise only
CI_histech_method_II_noise_only <- CI_cal(0.93125000, 0.03717711)
### The confidence interval (CI) of hamamastu scanner for Model with color noise only
CI_hamamatsu_method_II_noise_only <- CI_cal(0.967, 0.019)

## check the performance improvement
### histech scanner improvement from Model with no color noise
auc_diff_histech <- auc_diff(auc_with_sd_histech_II_noise_only, auc_with_sd_histech)
### hamamatsu scanner improvement from Model with no color noise
auc_diff_hamamatsu <- auc_diff(auc_with_sd_hamamatsu_II_noise_only, auc_with_sd_hamamatsu)

# subgroup analysis for the Model with color noise and color normalization
## read in the data
### read in the data: overall data
Method_II_noise_norm <- dnn.out[c(7,9)]
### read in the data: histech scanner
histech_Method_II_noise_norm <- dnn.out[as.integer(scanner$V2)==1, c(7,9)]
### read in the data: hamamastu scanner
hamamatsu_Method_II_noise_norm <- dnn.out[as.integer(scanner$V2)==2, c(7,9)]

## calculate the auc with its sd
### calculate the auc with sd for the overall data
auc_with_sd_II_noise_norm <- auc_cal(Method_II_noise_norm)
### calculate the auc with sd for the histech scanner
auc_with_sd_histech_II_noise_norm <- auc_cal(histech_Method_II_noise_norm)
### calculate the auc with sd for the hamamastu scanner
auc_with_sd_hamamatsu_II_noise_norm <- auc_cal(hamamatsu_Method_II_noise_norm)
### calculate the auc difference between histech and hamamastu scanners
auc_diff_Method_II_noise_norm <- auc_diff(auc_with_sd_hamamatsu_II_noise_norm, auc_with_sd_histech_II_noise_norm)

## calculate the confidence interval
### The overall confidence interval (CI)
CI_method_II_norm_noise <- CI_cal(0.98418367, 0.01184759)
### the confidence interval (CI) of hamamatsu scanner in Model with color noise and color normalization
CI_histech_method_II_noise_norm <- CI_cal(0.992500000, 0.005708622)
### the confidence interval (CI) of hamamatsu scanner in Model with color noise and color normalization
CI_hamamatsu_method_II_noise_norm <- CI_cal(0.98133333, 0.01866667)

## check the performance improvement
### the difference of auc between histech scanner and hamamatsu scanner in Model with color noise and color normalization
auc_diff_method_II_norm_noise <- auc_diff(auc_with_sd_histech_II_noise_norm, auc_with_sd_hamamatsu_II_noise_norm)
### improvement of performance compared to the Model with color noise only
auc_diff_histech_noise_norm <- auc_diff(auc_with_sd_histech_II_noise_norm, auc_with_sd_histech_II_noise_only)
auc_diff_hamamtsu_noise_norm <- auc_diff(auc_with_sd_hamamatsu_II_noise_norm, auc_with_sd_hamamatsu_II_noise_only)
