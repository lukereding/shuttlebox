require(tidyverse)
require(viridis)

if(! "footprints.csv" %in% list.files(getwd())){
  stop("need `footprints.csv` in the current directory")
}

read_csv("footprints_interpolated.csv") %>%
  ggplot(aes(x = x, y = y)) +
  geom_path(aes(color = frame)) +
  scale_color_viridis(option = "A") +
  theme_minimal()
ggsave("out.pdf", height = 5, width = 7)
