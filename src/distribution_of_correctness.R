# Library
library(dplyr)
library(ggplot2)

# Read the dataset
df = read.csv("./data/entropy_data/hypothesis_data.csv",header=TRUE)

# Plot overlapping histograms
plot_logprob_histogram <- function(df, logprob_col, label_col, 
                                   title = "Distribution of Seq-logprob by Prediction Correctness", 
                                   xlab = "Seq-logprob", 
                                   output_file = "histogram.pdf") {
  
  # Extract raw vectors
  logprob <- as.numeric(df[[logprob_col]])
  label_raw <- df[[label_col]]
  
  # Combine into one data frame early
  raw_df <- data.frame(logprob = logprob, label_raw = label_raw)
  
  # Drop rows with NA in either column
  raw_df <- na.omit(raw_df)
  
  
  # Ensure correct label formatting
  # Convert string labels ("False", "True") into factors
  #label <- factor(raw_label, levels = c("False", "True"), labels = c("Incorrect", "Correct"))
  if (is.logical(raw_df$label_raw)) {
    raw_df$label <- factor(raw_df$label_raw, levels = c(FALSE, TRUE), labels = c("Incorrect", "Correct"))
  } else {
    raw_df$label <- factor(as.character(raw_df$label_raw), levels = c("False", "True"), labels = c("Incorrect", "Correct"))
  }
  
  # Combine safely
  #if (length(logprob) != length(label)) stop("logprob and label columns have mismatched lengths.")
  
  # Final cleaned frame
  plot_df <- data.frame(logprob = raw_df$logprob, label = raw_df$label)
  
  # Build plot
  p <- ggplot(plot_df, aes(x = logprob, fill = label)) +
    geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
    scale_fill_manual(values = c("Incorrect" = "#e41a1c", "Correct" = "#4daf4a")) +
    labs(
      title = title,
      x = xlab,
      y = "Frequency",
      fill = "Prediction"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid = element_blank()
    )
  
  # Save as PDF
  ggsave(filename = output_file,
         plot = p,
         width = 8,
         height = 4,
         units = "in",
         dpi = 300)
}

# Roleplay
plot_logprob_histogram(df,
                       logprob_col = "roleplay_prob",
                       label_col = "label_match_roleplay",
                       title = "Seq-logprob Distribution for Roleplay Prompt Type",
                       xlab = "Seq-logprob",
                       output_file = "./images/roleplay_histogram.pdf")

# Content
plot_logprob_histogram(df,
                       logprob_col = "content_prob",
                       label_col = "label_match_content",
                       title = "Seq-logprob Distribution for Content Prompt Type",
                       xlab = "Seq-logprob",
                       output_file = "./images/content_histogram.pdf")

# Zero-shot
plot_logprob_histogram(df,
                       logprob_col = "zero.shot_prob",
                       label_col = "label_match_zero",
                       title = "Seq-logprob Distribution for Zero-shot Prompt Type",
                       xlab = "Seq-logprob",
                       output_file = "./images/zero_histogram.pdf")

# Few-shot
plot_logprob_histogram(df,
                       logprob_col = "few.shot_prob",
                       label_col = "label_match_few",
                       title = "Seq-logprob Distribution for Few-shot Prompt Type",
                       xlab = "Seq-logprob",
                       output_file = "./images/few_histogram.pdf")


