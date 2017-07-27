require(tidyverse)
require(viridis)
#require(emoGG)
require(viridis)

args <- commandArgs(TRUE)
csv_name <- args[1]
print(csv_name)

if(! csv_name %in% list.files(getwd())){
  stop(paste0("need ", csv_name, " in the current directory"))
}

df <- read_csv(csv_name)

found <- df %>% filter(fish == "found")


name <- csv_name %>%
  stringr::str_split("_") %>%
  magrittr::extract2(1) %>%
  magrittr::extract(length(.) - 1)

df %>%
  ggplot(aes(x =  x, y = y)) +
  geom_path(aes(color = frame)) +
  scale_color_viridis(option = "A") +
  # geom_point(data = found, aes(color = frame), size = 1) +
  theme_minimal() +
  ggtitle(paste0(csv_name))
ggsave(paste0(name, ".pdf"), height = 5, width = 7)


df %>%
  mutate(interval = cut_number(frame, 6)) %>%
  filter(! is.na(interval)) %>%
  ggplot(., aes(x = x, y = y)) +
  geom_density2d(data = df, aes(color = ..level..)) +
  geom_path(alpha = 0.6) +
  scale_color_viridis(option = "C", guide = F) +
  facet_wrap(~interval) +
  theme_minimal() +
  ggtitle(name, subtitle = "each plot represents ~10 minutes of the trial")
ggsave(paste0(name, "_contours.pdf"), height = 5, width = 7)

df %>%
  mutate(interval = cut_number(frame, 6)) %>%
  filter(! is.na(interval)) %>%
  ggplot(., aes(x = x, y = y)) +
  geom_hex(data = df) +
  geom_path(color = "grey80") +
  scale_fill_viridis(option = "C", guide = F) +
  facet_wrap(~interval) +
  theme_minimal() +
  ggtitle(name, subtitle = "each plot represents ~10 minutes of the trial")
ggsave(paste0(name, "_hex.pdf"), height = 5, width = 7)
