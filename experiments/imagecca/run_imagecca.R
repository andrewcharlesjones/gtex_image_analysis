library(readr)
library(tidyr)
library(dplyr)
library(forcats)
library(magrittr)

if (Sys.info()["user"] == "andrewjones") {
  DATA_DIR <- "/Users/andrewjones/Documents/beehive/gtex_image_analysis/data/imagecca"
  SAVE_DIR <- "/Users/andrewjones/Documents/beehive/gtex_image_analysis/experiments/imagecca/out"
  PENALTY_X = 1
  PENALTY_Z = 1
} else {
  DATA_DIR <- "/tigress/aj13/gtex_image_analysis/imagecca/data"
  SAVE_DIR <- "/tigress/aj13/gtex_image_analysis/imagecca/out"
  PENALTY_X = 0.1
  PENALTY_Z = 0.15
}

## Load data
expression_data <- read.csv(file.path(DATA_DIR, "expression_data_for_cca.csv"), row.names = 1)
img_data <- read.csv(file.path(DATA_DIR, "img_data_for_cca.csv"), row.names = 1)

## Save gene names
colnames(expression_data) %>% as.data.frame() %>% write_tsv(file.path(SAVE_DIR, "all_genes.tsv"))


# Remove columns with 0 variance
expression_data <- expression_data[, sapply(1:ncol(expression_data), function(x) sd(expression_data[, x]) > 0)]
img_data <- img_data[, sapply(1:ncol(img_data), function(x) sd(img_data[, x]) > 0)]

print("Expression data shape:")
print(dim(expression_data))
print("Image data shape:")
print(dim(img_data))

# Sparse CCA
SCCA <- function(datasets, penalty_x, penalty_z, K) {
  datasets[[1]] <- scale(datasets[[1]])
  datasets[[2]] <- scale(datasets[[2]])
  p <- dim(datasets[[1]])[2]
  q <- dim(datasets[[2]])[2]
  x <- PMA::CCA(datasets[[1]], datasets[[2]], penaltyx = penalty_x, penaltyz = penalty_z, K = min(K, min(p, q)))
  rownames(x$u) <- colnames(datasets[[1]])
  rownames(x$v) <- colnames(datasets[[2]])
  x$group_names <- names(datasets)
  x$CCA_var1 <- datasets[[1]] %*% x$u
  x$CCA_var2 <- datasets[[2]] %*% x$v
  return(x)
}

# Used to get coefficients from sparse CCA object
SCCA_coefs <- function(cca) {
  data_frame(CCA_var = 1:cca$K) %>%
    group_by(CCA_var) %>%
    do({
      k <- .$CCA_var
      nonzero_1 <- which(cca$u[, k] != 0)
      nonzero_2 <- which(cca$v[, k] != 0)
      bind_rows(
        data_frame(
          type = cca$group_names[1],
          name = rownames(cca$u)[nonzero_1],
          coefficient = cca$u[nonzero_1, k]
        ),

        data_frame(
          type = cca$group_names[2],
          name = rownames(cca$v)[nonzero_2],
          coefficient = cca$v[nonzero_2, k]
        )
      )
    }) %>%
    ungroup()
}

SCCA_cors <- function(cca) data_frame(k = 1:length(cca$d), d = cca$d)


##### Run ImageCCA

# Number of components
k <- 10

# Put datasets into list
datasets <- list(gene = expression_data %>% scale(), image = img_data)

# Run sparse CCA
cca <- SCCA(datasets, K = k, penalty_x = PENALTY_X, penalty_z = PENALTY_Z)

# Save CCA variables and coefficients
cca$CCA_var1 %>%
  as_tibble() %>%
  as.matrix() %>%
  as.data.frame() %>%
  set_rownames(rownames(cca$CCA_var1)) %>%
  write.csv(file.path(SAVE_DIR, "imagecca_expression_vars.txt"))
cca$CCA_var2 %>%
  as_tibble() %>%
  as.matrix() %>%
  as.data.frame() %>%
  set_rownames(rownames(cca$CCA_var2)) %>%
  write.csv(file.path(SAVE_DIR, "imagecca_image_vars.txt"))
cca_coeffs <- cca %>% SCCA_coefs()
cca_coeffs %>% write_tsv(file.path(SAVE_DIR, "imagecca_coeffs.txt"))
cca$d %>% as.data.frame() %>% write_tsv(file.path(SAVE_DIR, "imagecca_d_vals.txt"), col_names = F)



