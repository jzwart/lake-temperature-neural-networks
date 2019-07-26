library(tidyverse)
library(scipiper)
library(reticulate)
np <- import("numpy")

#train_config <- readr::read_tsv('2_model/out/train_config.tsv')
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
  return(list(depths=depths, dates=dates))
}
dims <- extract_dims(lake_files)

extract_preds <- function(lake_files) {
  lapply(setNames(nm=names(lake_files$output)), function(job_phase) {
    preds_file <- lake_files$output[[job_phase]][['preds']]
    message(job_phase, ":")
    message('  Opening predictions file...')
    preds_npz <- np$load(preds_file, allow_pickle=TRUE)
    preds_array <- preds_npz$f[["train_preds"]]
    epochs <- seq_len(dim(preds_array)[1])
    message('  Extracting by epoch...')
    preds_list <- lapply(setNames(epochs, nm=sprintf('%s_ep%03d', job_phase, epochs)), function(epoch) {
      if(epoch %% 10 == 0) message(if(epoch %% 100 == 0) epoch else '*', appendLF=FALSE)
      preds_array[epoch, , ]
    })
    message('')
    preds_list
  }) %>% purrr::flatten()
}
preds <- extract_preds(lake_files)

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

plot_train_stats <- function(train_stats_df, lake_id) {
  train_stats_df %>%
    filter(!is.na(loss)) %>%
    ggplot(aes(x=iter, y=loss, color=phase, group=phase)) +
    geom_line() +
    facet_grid(loss_type ~ ., scales='free_y') +
    scale_color_manual(values=c('pretrain'='#1b9e77', 'train'='#7570b3')) +
    xlab('Iteration') + ylab('Loss') +
    ggtitle(lake_id) +
      theme_bw() +
      theme(plot.title=element_text(size=9))
  # ggsave(save_path) save_path='3_assess/out/losses.png'
}
plot_train_stats(train_stats_df, lake_files$lake_id)
