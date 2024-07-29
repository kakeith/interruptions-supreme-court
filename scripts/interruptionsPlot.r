# interruptionsPlot.r
#
# 1.27.23
# =-=-=-=-=-=-
library(ggplot2)

data <- read.csv("./data/df_figure5.csv") 

data$justices <- data$justice_last_name
data$genderEffect <- as.numeric(data$Gender.Effect)
data$ideology <- as.numeric(data$justice_ideology_scores)
data$se <- as.numeric(data$Gender.Effect..1.96.Std)


p <- ggplot(data, aes(x=ideology, y=genderEffect)) + 
  geom_hline(yintercept=0, linetype = 2) +
  geom_segment(aes(x=ideology, xend=ideology, y= genderEffect + se, yend= genderEffect - se), color = "grey", alpha = 0.75, linewidth = 1.2, data=data) +
  geom_point(size = 1.5) + 
  geom_text(aes(label = justices), nudge_x = .2, nudge_y = .2) +
  ylim(-3.7,3.7) +
  xlim(-3.3, 3.3) +
  ylab("Interrupt Men More                  Interrupt Women More") +
  xlab("More Liberal                  . . .         More Conservative") +
  theme_bw()


ggsave(filename = './figs/fig5-gendereffect-ideologyscore.pdf', plot = p, width = 8, height = 6)