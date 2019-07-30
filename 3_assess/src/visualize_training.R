# gif and png visualizations of PGDL training process for Jordan's CUAHSI 2019
# poster (with iPad)

library(tidyverse)

#### DATA PREP ####
library(scipiper)
library(reticulate)
np <- import("numpy")

lake_dirs <- dir('2_model/tmp/cuahsi19/', full.names = TRUE)

label_lake_files <- function(lake_path=lake_dirs[1]) {
  lake_id <- basename(lake_path)
  file_list <- list(
    lake_id = lake_id,
    input = list(
      R_object = sprintf('R_data_%s', lake_id)
    ),
    output = lapply(setNames(nm=c('pretrain', 'train')), function(job_phase) {
      job_path <- file.path(lake_path, job_phase)
      job_files <- dir(job_path, full.names=TRUE)
      list(preds = grep('preds', job_files, value=TRUE),
           stats = grep('stats', job_files, value=TRUE))
    })
  )
  return(file_list)
}
lake_files <- label_lake_files(lake_dirs[1])

extract_dims <- function(lake_files) {
  inputs <- scmake(lake_files$input$R_object, '1_format_tasks.yml')
  depths <- inputs$geometry$Depth
  dates <- inputs$sequences$predict[[1]] %>%
    lapply(function(seq) { dimnames(seq)$date }) %>%
    unlist() %>% unique() %>%
    as.Date(format='%Y-%m-%d') %>%
    sort()
  date_df <- tibble(date = dates) %>%
    mutate(date_id = sprintf('V%d', 1:n()))
  return(list(depths=depths, dates=dates, date_df=date_df))
}
dims <- extract_dims(lake_files)

extract_preds <- function(lake_files) {
  huge_preds <- lapply(setNames(nm=names(lake_files$output)), function(job_phase) {
    preds_file <- lake_files$output[[job_phase]][['preds']]
    message(job_phase, ":")
    message('  Opening predictions file...')
    preds_npz <- np$load(preds_file, allow_pickle=TRUE)
    preds_array <- preds_npz$f[["train_preds"]]
    epochs <- seq_len(dim(preds_array)[1])
    message('  Extracting by epoch...')
    pred_sets <- epochs %>% {setNames(., nm=sprintf('%s_ep%03d', job_phase, .))}
    # pred_sets <- seq(1,91,by=10) %>% {setNames(., nm=sprintf('%s_ep%03d', job_phase, .))}
    preds_list <- lapply(seq_along(pred_sets), function(set) {
      epoch <- pred_sets[set]
      if(epoch %% 10 == 0) message(if(epoch %% 100 == 0) epoch else '*', appendLF=FALSE)
      preds_mat <- preds_array[epoch, , ]
      rownames(preds_mat) <- dims$depths
      preds_df <- preds_mat %>%
        t() %>%
        as_tibble() %>%
        mutate(date = dims$dates) %>%
        gather(depth_m, temp_C, -date) %>%
        mutate(depth_m = as.numeric(depth_m))
      names(preds_df)[3] <- names(epoch)
      if(set == 1) preds_df else preds_df[3]
    })
    message('')
    preds_list
  }) %>% purrr::flatten()
  big_preds <- bind_cols(huge_preds) %>%
    select(-date1, -depth_m1)

  return(big_preds)
}
preds <- extract_preds(lake_files)
saveRDS(preds, '3_assess/tmp/preds.rds')

extract_obs <- function(lake_files, phase='pretrain') {
  inputs <- scmake(lake_files$input$R_object, '1_format_tasks.yml')
  lapply(inputs$sequences[[phase]]$labels, function(obs_mat) {
    obs_mat %>%
      as_tibble() %>%
      mutate(depth_m = as.numeric(rownames(obs_mat))) %>%
      gather(date, temp_C, -depth_m) %>%
      mutate(date = as.Date(date, format='%Y-%m-%d')) %>%
      filter(!is.na(temp_C))
  }) %>%
    bind_rows() %>%
    distinct()
}
glm <- extract_obs(lake_files, phase='pretrain')
saveRDS(glm, '3_assess/tmp/glm.rds')
obs <- extract_obs(lake_files, phase='train')
saveRDS(obs, '3_assess/tmp/obs.rds')

extract_stats <- function(lake_files) {
  bind_rows(lapply(names(lake_files$output), function(job_phase) {
    stats_file <- lake_files$output[[job_phase]][['stats']]
    stats_npz <- np$load(stats_file, allow_pickle=TRUE)
    stats <- lapply(setNames(nm=stats_npz$files), function(npz_file) stats_npz$f[[npz_file]])
    crossing(epoch = 1:dim(stats$train_stats)[1], batch = 1:dim(stats$train_stats)[2]) %>%
      mutate(
        phase = job_phase,
        total = c(t(stats$train_stats[,,1])),
        RMSE = c(t(stats$train_stats[,,2])),
        EC = c(t(stats$train_stats[,,3])),
        L1 = c(t(stats$train_stats[,,4])),
        test = ifelse(batch == max(batch), stats$test_loss_rmse[epoch], NA)
      ) %>%
      gather(loss_type, loss, total, RMSE, EC, L1, test) %>%
      mutate(loss_type = ordered(loss_type, levels=c('RMSE', 'EC', 'L1', 'total', 'test'))) %>%
      select(phase, everything())
  })) %>%
    mutate(phase = ordered(phase)) %>%
    group_by(loss_type) %>%
    arrange(phase) %>%
    mutate(
      iter = 1:n(),
      era = ceiling(10 * iter / max(iter)) # break iterations into "eras" of 1 through 10
    )
}
train_stats_df <- extract_stats(lake_files)
saveRDS(train_stats_df, '3_assess/tmp/stats.rds')

#### PLOTS ####
# reload data prepped above
preds <- readRDS('3_assess/tmp/preds.rds')
glm <- readRDS('3_assess/tmp/glm.rds')
obs <- readRDS('3_assess/tmp/obs.rds')
train_stats_df <- readRDS('3_assess/tmp/stats.rds')

plot_preds <- function(pred_df, glm_df, obs_df, frame='pretrain_ep001', ylims, save_dir) {
  # parse the frame name
  phase <- strsplit(frame, '_', fixed=TRUE)[[1]][[1]]
  epoch <- as.integer(gsub('ep', '', strsplit(frame, '_', fixed=TRUE)[[1]][[2]]))
  plot_title <- sprintf('%s%sing Epoch %d', toupper(substr(phase, 1, 1)), substring(phase, 2), epoch)

  # make the plot
  g <- (if(phase == 'pretrain') glm_df else obs_df) %>%
    ggplot(aes(x=date, y=temp_C, color=depth_m, group=depth_m)) +
    geom_line(data=pred_df) +
    (if(phase == 'pretrain') geom_line(linetype='dotted') else geom_point(size=0.8)) +
    viridis::scale_color_viridis('Depth (m)', discrete=FALSE, direction=-1, guide=guide_colorbar(reverse=TRUE)) +#, palette = 'RdYlBu'
    ylim(ylims) +
    xlab('Date') + ylab('Temperature (degrees C)') +
    theme_classic() +
    ggtitle(plot_title)

  # save the plot if requested
  if(!missing(save_dir)) {
    ggsave(file.path(save_dir, sprintf('%s.png', frame)), plot=g, height=4.5, width=8)
  }

  # return the plot for previewing
  return(g)
}
pred_df <- preds[c('date','depth_m','pretrain_ep011')] %>% filter(
  dplyr::between(date, as.Date('2013-01-01'), as.Date('2014-12-31')),
  depth_m %% 2 == 0) %>%
  rename(temp_C = pretrain_ep011)
glm_df <- glm %>% filter(
  dplyr::between(date, as.Date('2013-01-01'), as.Date('2014-12-31')),
  depth_m %% 2 == 0)
obs_df <- obs %>% filter(
  dplyr::between(date, as.Date('2013-01-01'), as.Date('2014-12-31')),
  depth_m %% 2 == 0)
plot_preds(pred_df, glm_df, obs_df, frame='pretrain_ep011', ylims=c(0,25))

#### ANIMATION ####

create_animation_frames <- function(preds, glm, obs, save_dir='3_assess/tmp/png') {
  # Filter all of the observations and predictions to smaller values
  filter_for_plot <- function(df) {
    df %>% filter(
      dplyr::between(date, as.Date('2013-01-01'), as.Date('2014-12-31')),
      depth_m %% 1.5 == 0)
  }
  preds_subset <- filter_for_plot(preds)
  glm_subset <- filter_for_plot(glm)
  obs_subset <- filter_for_plot(obs)

  # Unpack the big preds tibble back into a list of tibbles
  preds_names <- names(preds_subset)[-c(1:2)]
  preds_list <- lapply(setNames(nm=preds_names), function(pred_col) {
    pred_df <- preds_subset[c('date','depth_m',pred_col)]
    names(pred_df)[3] <- 'temp_C'
    pred_df
  })

  # Compute y limits for this animation
  ylims <- range(c(
    range(preds_subset[-(1:2)]),
    range(glm_subset[-(1:2)]),
    range(obs_subset[-(1:2)])))

  # Create and save the animation frames
  if(!dir.exists(save_dir)) dir.create(save_dir, recursive=TRUE)
  lapply(names(preds_list), function(frame_name) {
    message(frame_name)
    plot_preds(preds_list[[frame_name]], glm_subset, obs_subset, frame_name, ylims, save_dir=save_dir)
  })

  invisible()
}
create_animation_frames(preds, glm, obs)

combine_animation_frames <- function(
  png_files=dir('3_assess/tmp/png', pattern='png', full.names=TRUE),
  gif_file='3_assess/tmp/training_preds.gif',
  min_delay_cs=10, max_delay_cs=200, decay_exp=0.9) {

  # run imageMagick convert to build a gif
  tmp_dir <- file.path(dirname(gif_file), 'magick')
  if(!dir.exists(tmp_dir)) dir.create(tmp_dir)
  magick_command <- sprintf(
    'convert -define registry:temporary-path=%s -limit memory 24GiB -delay %d -loop 0 %s %s',
    tmp_dir, max_delay_cs, paste(png_files, collapse=' '), gif_file)
  if(Sys.info()[['sysname']] == "Windows") {
    magick_command <- sprintf('magick %s', magick_command)
  }
  system(magick_command)

  # for use with gifsicle below, create a string defining the delay for each
  # frame individually. frame_nums is converted from 1-indexed to 0-indexed
  format_decaying_delays <- function(min_delay_cs=10, max_delay_cs=30, decay_exp=0.9, frame_nums) {
    delays <- pmax(round(min_delay_cs + (max_delay_cs-min_delay_cs)*seq_len(length(frame_nums))^-decay_exp), 1)
    sprintf('-d%s "#%s"', delays, frame_nums-1) %>%
      paste(collapse=' ')
  }
  seq_pretrain <- as.integer(gsub('pretrain_ep', '', tools::file_path_sans_ext(basename(grep('/pretrain', png_files, value=TRUE)))))
  seq_train <- max(seq_pretrain) + as.integer(gsub('train_ep', '', tools::file_path_sans_ext(basename(grep('/train', png_files, value=TRUE)))))
  delay_string_p1 <- format_decaying_delays(min_delay_cs, max_delay_cs, decay_exp, seq_pretrain[-(-2:0 + length(seq_pretrain))])
  delay_string_p2 <- format_decaying_delays(max_delay_cs/2, max_delay_cs/2, decay_exp=1, seq_pretrain[-2:0 + length(seq_pretrain)])
  delay_string_t1 <- format_decaying_delays(min_delay_cs, max_delay_cs, decay_exp, seq_train[-(-2:0 + length(seq_train))])
  delay_string_t2 <- format_decaying_delays(max_delay_cs/2, max_delay_cs/2, decay_exp=1, seq_train[-2:0 + length(seq_train)])
  delay_string <- paste(delay_string_p1, delay_string_p2, delay_string_t1, delay_string_t2)

  # simplify the gif with gifsicle - cuts size by about 50%
  # for now, at least, --colors 256 does nothing
  gifsicle_command <- sprintf('gifsicle -b -O3 %s %s', gif_file, delay_string)
  system(gifsicle_command)
}
combine_animation_frames(
  png_files=dir('3_assess/tmp/png', pattern='png', full.names=TRUE),
  gif_file='3_assess/tmp/training_preds.gif',
  min_delay_cs=10, max_delay_cs=200, decay_exp=0.9)

#### REFERENCE CODE ####

plot_train_stats <- function(train_stats_df, lake_id) {
  g <- train_stats_df %>%
    filter(!is.na(loss)) %>%
    ggplot(aes(x=iter, y=loss, color=phase, group=phase)) +
    geom_line() +
    facet_grid(loss_type ~ ., scales='free_y') +
    scale_color_manual(values=c('pretrain'='#1b9e77', 'train'='#7570b3')) +
    xlab('Iteration') + ylab('Loss') +
    ggtitle(sprintf('Loss components for Trout Lake, WI (%s)', lake_id)) +
    theme_bw() +
    theme(plot.title=element_text(size=9))
  ggsave('3_assess/tmp/losses.png', g, width=7, height=6)

}
plot_train_stats(train_stats_df, lake_files$lake_id)

# ggplot(plot_dat, aes(x = doy, y = value)) +
#   geom_line(aes(group = depth_bin, color = depth_bin), alpha = 0.7, size = 1.1) +
#   geom_point(aes(group = depth_bin, color = depth_bin),
#              alpha = 0.7, size = 0.9, fill = 'white') +
#   geom_point(data = plot_dat, aes(x = doy, y = source_obs), shape = '|', color = 'darkgray', size = 2) +
#   geom_hline(data = dummy_hline, aes(yintercept = int), linetype = 2) +
#   viridis::scale_color_viridis(discrete = TRUE, direction = -1) + #, palette = 'RdYlBu'
#   #scale_shape_manual(values = c(21, 22, 23)) +
#   scale_x_continuous(breaks = date_breaks,
#                      labels = date_labels, limits = c(min(plot_dat$doy, na.rm = TRUE)-5, max(plot_dat$doy, na.rm = TRUE)+5))+
#   #facet_wrap(Depth_cat~year, ncol = 3, scales = 'free_y') +
#   facet_grid(rows = vars(variable),  scales = 'free_y') +
#   coord_cartesian(xlim = c(min_date-5, max_date+5)) +
#   theme_bw()+
#   theme(#strip.text = element_blank(),
#     strip.background = element_blank(),
#     legend.position = 'right',
#     legend.direction="horizontal",
#     legend.key.size = unit(1.5, 'lines'),
#     panel.grid = element_blank(),
#     legend.background = element_rect(fill = 'transparent', color = NA),
#     legend.box.background = element_rect(color = NA),
#     panel.spacing = unit(0.8, 'lines'),
#     legend.spacing.y = unit(2, 'lines'),
#     text = element_text(size = 24),
#     axis.text.y = element_text(size = 18)
#   ) +
#   labs( y = 'Observed Temperature or Bias (deg C)', x = '', title = paste0(target_name, ', ', top_year)) +
#   guides(color = guide_legend(title = 'Depth (m)', title.position = 'top', ncol = 1,
#                               label.position = 'left', direction = "vertical"))
