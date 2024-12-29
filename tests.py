
import rpy2.robjects as ro
r = ro.r
r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')
print(r('Sys.getlocale()'))  # Verify the locale settings