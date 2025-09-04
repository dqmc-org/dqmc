library(ggplot2)

data <- read.table("data.csv", sep = ",", header=TRUE)

ggplot(data, aes(x = inv_beta, y = x, color = factor(onsite_U))) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = x - abs_err, ymax = x + abs_err), width = 0.01) +
  scale_x_log10() +
  labs(x = "inv_beta", y = "X", color = "onsite_U") +
  theme_minimal()
