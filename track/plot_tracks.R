require(tidyverse)
require(viridis)
require(emoGG)

args <- commandArgs(TRUE)
csv_name <- args[1]

if(! csv_name %in% list.files(getwd())){
  stop(paste0("need ", csv_name, " in the current directory"))
}

found <- read_csv(csv_name) %>% filter(fish == "found")

read_csv(csv_name) %>%
  ggplot(aes(x =  x, y = y)) +
  add_emoji("1f41f") +
  geom_path(aes(color = frame)) +
  # geom_point(data = found, aes(color = frame), size = 1) +
  scale_color_viridis(option = "A") +
  theme_minimal() +
  ggtitle(paste0(csv_name))
ggsave("out.pdf", height = 5, width = 7)
