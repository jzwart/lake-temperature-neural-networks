# This file creates gif and png visualizations of the PGDL training process for
# the training of a single model.

# The script is currently set up to be run interactively rather than with
# scipiper. Edit the directory paths to point to the project of interest.

library(tidyverse)
library(scipiper)
library(reticulate)
np <- import("numpy")

#### DATA PREP ####

lake_dirs <- dir('tmp/190814_WMA_TED_Talk/2_model/tmp/', full.names = TRUE)

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
      list(params = grep('params', job_files, value=TRUE),
           preds = grep('preds', job_files, value=TRUE),
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
    epochs <- seq_len(dim(preds_array)[1]) - 1
    message('  Extracting by epoch...')
    pred_sets <- epochs %>% {setNames(., nm=sprintf('%s_ep%03d', job_phase, .))}
    preds_list <- lapply(seq_along(pred_sets), function(set) {
      epoch <- pred_sets[set]
      if(epoch %% 10 == 0) message(if(epoch %% 100 == 0) epoch else '*', appendLF=FALSE)
      preds_mat <- preds_array[epoch+1, , ]
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
saveRDS(preds, 'tmp/190814_WMA_TED_Talk/3_assess/tmp/preds.rds')

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
saveRDS(glm, 'tmp/190814_WMA_TED_Talk/3_assess/tmp/glm.rds')
obs <- extract_obs(lake_files, phase='train')
saveRDS(obs, 'tmp/190814_WMA_TED_Talk/3_assess/tmp/obs.rds')

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
train_stats <- extract_stats(lake_files)
saveRDS(train_stats, 'tmp/190814_WMA_TED_Talk/3_assess/tmp/stats.rds')

extract_params <- function(lake_files) {
  lapply(setNames(nm=names(lake_files$output)), function(job_phase) {
    params_file <- lake_files$output[[job_phase]][['params']]
    params_npz <- np$load(params_file, allow_pickle=TRUE)
    params_list <- params_npz$f[['train_params']][[1]]
  })
}
params <- extract_params(lake_files)
saveRDS(params, 'tmp/190814_WMA_TED_Talk/3_assess/tmp/params.rds')


#### ANIMATION FRAMES ####

# reload data prepped above
preds <- readRDS('tmp/190814_WMA_TED_Talk/3_assess/tmp/preds.rds')
glm <- readRDS('tmp/190814_WMA_TED_Talk/3_assess/tmp/glm.rds')
obs <- readRDS('tmp/190814_WMA_TED_Talk/3_assess/tmp/obs.rds')
params <- readRDS('tmp/190814_WMA_TED_Talk/3_assess/tmp/params.rds')

plot_params <- function(params, frame='pretrain_ep001', clims=c(-1,1), save_dir) {
  # parse the frame name
  phase <- strsplit(frame, '_', fixed=TRUE)[[1]][[1]]
  epoch <- as.integer(gsub('ep', '', strsplit(frame, '_', fixed=TRUE)[[1]][[2]]))
  plot_title <- sprintf('%s%sing Epoch %d', toupper(substr(phase, 1, 1)), substring(phase, 2), epoch)
  params_list <- params[[phase]]

  # determine the dimensions of the params matrix "parmat" (which will have a
  # gap between each set of parameters)
  gapsize <- 1
  biases_per_node <- 1
  n_gates <- 4
  n_hidden_states <- dim(params_list$lstm_weights)[3] / n_gates
  n_drivers <- dim(params_list$lstm_weights)[2] - n_hidden_states
  parmat_width <- dim(params_list$lstm_weights)[3] + gapsize + dim(params_list$pred_weights)[3]
  parmat_height <- dim(params_list$lstm_weights)[2] + gapsize + biases_per_node

  # get the data for this plot
  parmat <- matrix(NA, nrow=parmat_height, ncol=parmat_width)
  # put the lstm_weights in the top left
  parmat[1:dim(params_list$lstm_weights)[2], 1:dim(params_list$lstm_weights)[3]] <- params_list$lstm_weights[epoch+1,,]
  # put the lstm_biases in the bottom left
  parmat[parmat_height, 1:dim(params_list$lstm_biases)[2]] <- params_list$lstm_biases[epoch+1,]
  # put the pred_weights in the middle right
  parmat[(1 + dim(params_list$lstm_weights)[2] - dim(params_list$pred_weights)[2]):dim(params_list$lstm_weights)[2], parmat_width] <-
    params_list$pred_weights[epoch+1,,]
  # put the pred_biases in the bottom right
  parmat[parmat_height, parmat_width] <- params_list$pred_biases[epoch+1]

  # melt the parameter matrix into a tidy tibble
  dimnames(parmat) <- list(as.character(seq_len(dim(parmat)[1])), sprintf('C%d', seq_len(dim(parmat)[2])))
  parmelt <- parmat %>%
    as_tibble() %>%
    mutate(row=as.integer(rownames(parmat))) %>%
    tidyr::gather(col, value, -row) %>%
    mutate(col=as.integer(gsub('C', '', col)))

  # make the geom_tile plot
  base_size <- 9
  g <- ggplot(parmelt, aes(x=col, y=row)) +
    geom_tile(aes(fill = value), colour = NA) +
    scale_fill_gradient2(
      '', breaks=c(clims[1], 0, clims[2]), labels=c('Low', '0', 'High'),
      low = "midnightblue", mid = "gray95", high = "darkred", midpoint = 0, na.value = 'white', lim=clims) +
    scale_x_continuous(
      expand = c(0, 0),
      breaks=c( n_hidden_states/2+n_hidden_states*(0:3), parmat_width),
      labels=c('Input Gate', 'New Input Gate', 'Forget Gate', 'Output Gate', 'Prediction')) +
    scale_y_reverse(
      expand = c(0, 0),
      breaks=c(n_drivers/2, n_drivers+n_hidden_states/2, parmat_height),
      labels=c('Driver\nWeights', 'Hidden\nState\nWeights', 'Biases')) +
    coord_fixed() +
    labs(x = "", y = "") +
    theme_grey(base_size = base_size) +
    theme(
      legend.position = 'bottom',
      legend.key.width = unit(0.6, units='inches'),
      legend.key.height = unit(0.08, units='inches'),
      axis.ticks = element_blank(),
      plot.margin = margin(r=25)) +
    geom_vline(xintercept=dim(params_list$lstm_weights)[3] * (1:4)/4 + 0.5, color='white') +
    geom_hline(yintercept=dim(params_list$lstm_weights)[2]-dim(params_list$pred_weights)[2] + 0.5, color='white') +
    ggtitle(plot_title)

  # Save if requested
  if(!missing(save_dir)) {
    if(!dir.exists(save_dir)) dir.create(save_dir, recursive=TRUE)
    ggsave(file.path(save_dir, sprintf('%s.png', frame)), plot=g, height=3.5, width=6)
  }

  return(g)
}
# # tests
# maxes <- list(
#   lstm_weights = max(abs(c(params[[1]]$lstm_weights, params[[2]]$lstm_weights))),
#   lstm_biases  = max(abs(c(params[[1]]$lstm_biases,  params[[2]]$lstm_biases ))),
#   pred_weights = max(abs(c(params[[1]]$pred_weights, params[[2]]$pred_weights))),
#   pred_biases  = max(abs(c(params[[1]]$pred_biases,  params[[2]]$pred_biases ))))
# params_norm <- lapply(params, function(params_list) {
#   list(
#     lstm_weights = params_list$lstm_weights / maxes$lstm_weights,
#     lstm_biases = params_list$lstm_biases / maxes$lstm_biases,
#     pred_weights = params_list$pred_weights / maxes$pred_weights,
#     pred_biases = params_list$pred_biases / maxes$pred_biases)
# })
# clims <- c(-1,1)*max(abs(range(c(params_norm[['pretrain']]$lstm_weights, params_norm[['train']]$lstm_weights))))
# plot_params(params_norm, frame='pretrain_ep000', clims=clims)
# plot_params(params_norm, frame='pretrain_ep050', clims=clims)
# plot_params(params_norm, frame='pretrain_ep100', clims=clims)
# plot_params(params_norm, frame='train_ep000', clims=clims)
# plot_params(params_norm, frame='train_ep050', clims=clims)
# plot_params(params_norm, frame='train_ep100', clims=clims)

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
# # tests
# pred_df <- preds[c('date','depth_m','pretrain_ep011')] %>% filter(
#   dplyr::between(date, as.Date('2013-01-01'), as.Date('2014-12-31')),
#   depth_m %% 2 == 0) %>%
#   rename(temp_C = pretrain_ep011)
# glm_df <- glm %>% filter(
#   dplyr::between(date, as.Date('2013-01-01'), as.Date('2014-12-31')),
#   depth_m %% 2 == 0)
# obs_df <- obs %>% filter(
#   dplyr::between(date, as.Date('2013-01-01'), as.Date('2014-12-31')),
#   depth_m %% 2 == 0)
# plot_preds(pred_df, glm_df, obs_df, frame='pretrain_ep011', ylims=c(0,25))


#### ANIMATION GIFS ####

create_param_animation_frames <- function(params, save_dir='tmp/190814_WMA_TED_Talk/3_assess/tmp/params/png') {
  # normalize weights in each matrix so they range from -1 to 1 within that
  # matrix over both phases and all epochs
  maxes <- list(
    lstm_weights = max(abs(c(params[[1]]$lstm_weights, params[[2]]$lstm_weights))),
    lstm_biases  = max(abs(c(params[[1]]$lstm_biases,  params[[2]]$lstm_biases ))),
    pred_weights = max(abs(c(params[[1]]$pred_weights, params[[2]]$pred_weights))),
    pred_biases  = max(abs(c(params[[1]]$pred_biases,  params[[2]]$pred_biases ))))
  params_norm <- lapply(params, function(params_list) {
    list(
      lstm_weights = params_list$lstm_weights / maxes$lstm_weights,
      lstm_biases = params_list$lstm_biases / maxes$lstm_biases,
      pred_weights = params_list$pred_weights / maxes$pred_weights,
      pred_biases = params_list$pred_biases / maxes$pred_biases)
  })

  # Set color limits for this animation: we know they're (-1, 1) because of the
  # normalization above
  clims <- c(-1,1)

  # Create and save the animation frames
  n_epochs <- lapply(params_norm, function(params_list) dim(params_list$lstm_weights)[1] ) # includes the 0 epoch (before any training)
  frame_names <- unlist(lapply(names(n_epochs), function(phase) {
    sprintf('%s_ep%03d', rep(phase, times=n_epochs[[phase]]), seq_len(n_epochs[[phase]]) - 1)
  }))
  if(!dir.exists(save_dir)) dir.create(save_dir, recursive=TRUE)
  lapply(frame_names, function(frame_name) {
    message(frame_name)
    plot_params(params_norm, frame=frame_name, clims=clims, save_dir=save_dir);
  })

  invisible()
}
create_param_animation_frames(params, 'tmp/190814_WMA_TED_Talk/3_assess/tmp/params/png')


create_pred_animation_frames <- function(preds, glm, obs, save_dir) {
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
create_pred_animation_frames(preds, glm, obs, save_dir='tmp/190814_WMA_TED_Talk/3_assess/tmp/preds/png')


combine_animation_frames <- function(png_files, gif_file, min_delay_cs=10, max_delay_cs=200, decay_exp=0.9) {

  # run imageMagick convert to build a gif
  tmp_dir <- file.path(dirname(unique(dirname(png_files))), 'magick')
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
  seq_train <- length(seq_pretrain) + as.integer(gsub('train_ep', '', tools::file_path_sans_ext(basename(grep('/train', png_files, value=TRUE)))))
  delay_string_p1 <- format_decaying_delays(min_delay_cs, max_delay_cs, decay_exp, seq_pretrain[-(-2:0 + length(seq_pretrain))])
  delay_string_p2 <- format_decaying_delays(max_delay_cs/2, max_delay_cs/2, decay_exp=1, seq_pretrain[-2:0 + length(seq_pretrain)])
  delay_string_t1 <- format_decaying_delays(min_delay_cs, max_delay_cs, decay_exp, seq_train[-(-2:0 + length(seq_train))])
  delay_string_t2 <- format_decaying_delays(max_delay_cs/2, max_delay_cs/2, decay_exp=1, seq_train[-2:0 + length(seq_train)])
  delay_string <- paste(delay_string_p1, delay_string_p2, delay_string_t1, delay_string_t2)

  # simplify the gif with gifsicle - cuts size by about 50%
  # for predictions, --colors 256 does nothing, but it's useful for params
  gifsicle_command <- ifelse(
    grepl('params', gif_file),
    sprintf('gifsicle --colors 256 -b -O3 %s %s', gif_file, delay_string),
    sprintf('gifsicle -b -O3 %s %s', gif_file, delay_string))
  system(gifsicle_command)
}
combine_animation_frames(
  png_files=dir('tmp/190814_WMA_TED_Talk/3_assess/tmp/params/png', pattern='png', full.names=TRUE),
  gif_file='tmp/190814_WMA_TED_Talk/3_assess/tmp/training_params.gif',
  min_delay_cs=10, max_delay_cs=200, decay_exp=0.9)
combine_animation_frames(
  png_files=dir('tmp/190814_WMA_TED_Talk/3_assess/tmp/preds/png', pattern='png', full.names=TRUE),
  gif_file='tmp/190814_WMA_TED_Talk/3_assess/tmp/training_preds.gif',
  min_delay_cs=10, max_delay_cs=200, decay_exp=0.9)

#### MODEL DIAGNOSTICS PLOTS ####

train_stats <- readRDS('tmp/190814_WMA_TED_Talk/3_assess/tmp/stats.rds')

plot_train_stats <- function(train_stats, lake_id, save_path) {
  g <- train_stats %>%
    filter(!is.na(loss)) %>%
    ggplot(aes(x=iter, y=loss, color=phase, group=phase)) +
    geom_line() +
    facet_grid(loss_type ~ ., scales='free_y') +
    scale_color_manual(values=c('pretrain'='#1b9e77', 'train'='#7570b3')) +
    xlab('Iteration') + ylab('Loss') +
    ggtitle(sprintf('Loss components for Trout Lake, WI (%s)', lake_id)) +
    theme_bw() +
    theme(plot.title=element_text(size=9))
  ggsave(file.path(save_path, 'losses.png'), g, width=7, height=6)

}
plot_train_stats(train_stats, lake_files$lake_id, save_path='tmp/190814_WMA_TED_Talk/3_assess/tmp')
