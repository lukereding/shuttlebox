source("~/Documents/random_scripts/plotting_functions.R")

args <- commandArgs(TRUE)
csv_name <- args[1]
print(csv_name)

#if(! csv_name %in% list.files(getwd())){
#  stop(paste0("need ", csv_name, " in the current directory"))
#}

df <- read_csv(csv_name)

# if is temporary; only use if I'm tracking every 10th frame
df <- slice(df, 0:10800)

found <- df %>% filter(fish == "found")

# get the name of the fish for the trial
name <- csv_name %>%
  stringr::str_split("_") %>%
  purrr::pluck(1) %>%
  purrr::pluck(length(.) - 1)

# OR, generate a nonsense name
# name <- Reduce(paste0, sample(letters, 10, replace = T))

# or use the full path name
name <- csv_name

# path of the fish
df %>%
  ggplot(aes(x =  x, y = y)) +
  geom_path(aes(color = frame)) +
  scale_color_viridis(option = "A") +
  geom_point(data = found, aes(color = frame), size = 1) +
  theme_minimal() +
  ggtitle(paste0(csv_name))
ggsave(paste0(name, ".png"), height = 5, width = 7)

# #
# df %>%
#   mutate(interval = cut_number(frame, 6)) %>%
#   filter(! is.na(interval)) %>%
#   ggplot(., aes(x = x, y = y)) +
#   geom_density2d(data = df, aes(color = ..level..)) +
#   geom_path(alpha = 0.6) +
#   scale_color_viridis(option = "C", guide = F) +
#   facet_wrap(~interval) +
#   theme_minimal() +
#   ggtitle(name, subtitle = "each plot represents ~10 minutes of the trial")
# ggsave(paste0(name, "_contours.pdf"), height = 5, width = 7)

# hex plot by interval
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
ggsave(paste0(name, "_hex.png"), height = 5, width = 7)

# get frames spend in each one by interval
df %>%
  mutate(interval = cut_number(frame, 6)) %>%
  group_by(interval, zone) %>%
  count %>%
  filter(! is.na(interval)) %>%
  ggplot(aes(x = zone, y = n)) +
  geom_col(aes(fill = zone)) +
  theme_minimal() +
  facet_wrap(~interval) +
  scale_fill_carto(guide = F) +
  ggtitle(name, subtitle = "each plot represents ~10 minutes of the trial")
ggsave(paste0(name, "_by_zone.png"), height = 5, width = 7)

# joyplot
library(ggridges)
df %>%
  mutate(interval = cut_number(frame, 30)) %>%
  filter(! is.na(interval)) %>%
  ggplot(aes(x = x, y = interval)) + 
  geom_density_ridges2(color = "white") +
  theme_ridges()
ggsave(paste0(name, "_ridgeplot.png"), height = 5, width = 7)
