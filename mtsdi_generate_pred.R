args <- commandArgs(trailingOnly = TRUE)
full_path <- args[1]
full_path <- normalizePath(full_path, winslash="/")

print(paste("full_path:", full_path))
library(mtsdi)
print("Reading data...")
data <- read.csv(file.path(full_path, "gaps.csv"))
f <- ~P1_VWC+P2_VWC+P3_VWC+P4_VWC+P5_VWC+P6_VWC+P7_VWC+P8_VWC+P9_VWC+P10_VWC+P11_VWC+P12_VWC+P13_VWC+P14_VWC+P15_VWC+P16_VWC
model <- mnimput(f,data,method="spline")
pred <- predict(model)
write.csv(pred,file.path(full_path, "pred.csv"))