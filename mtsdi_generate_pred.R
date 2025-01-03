args <- commandArgs(trailingOnly = TRUE)
full_path <- args[1]
full_path <- normalizePath(full_path, winslash="/")

# Parse hyperparameters from args
params <- list()
i <- 2
while (i <= length(args)) {
    if (substr(args[i], 1, 2) == "--") {
        param_name <- substr(args[i], 3)
        if (i + 1 <= length(args)) {
            param_value <- args[i + 1]
            # Convert parameter to appropriate type
            if (param_value == "NULL") {
                params[[param_name]] <- NULL
            } else if (grepl("^\\d+$", param_value)) {
                params[[param_name]] <- as.integer(param_value)
            } else if (grepl("^\\d+\\.\\d+$", param_value)) {
                params[[param_name]] <- as.numeric(param_value)
            } else {
                params[[param_name]] <- param_value
            }
            i <- i + 2
        } else {
            params[[param_name]] <- NULL
            i <- i + 1
        }
    } else {
        i <- i + 1
    }
}

print(paste("full_path:", full_path))
library(mtsdi)
print("Reading data...")
data <- read.csv(file.path(full_path, "gaps.csv"))
f <- ~P1_VWC+P2_VWC+P3_VWC+P4_VWC+P5_VWC+P6_VWC+P7_VWC+P8_VWC+P9_VWC+P10_VWC+P11_VWC+P12_VWC+P13_VWC+P14_VWC+P15_VWC+P16_VWC

# Use hyperparameters in mnimput
model <- mnimput(f, data, method=params[['method']], m=params[['m']], maxiter=params[['maxiter']], tol=params[['tol']])
pred <- predict(model)
write.csv(pred, file.path(full_path, "pred.csv"))