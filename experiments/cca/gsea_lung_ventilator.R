library(magrittr)
library(dplyr)
library(readr)
library(ggplot2)


GSEA_PATH <- "~/Documents/beehive/multimodal_bio/gsea/gsea.R"
CCA_COEFFS_PATH <- "/Users/andrewjones/Documents/beehive/gtex_image_analysis/experiments/cca/out"
GENE_NAME_PATH <- "~/Documents/beehive/multimodal_bio/util/ensembl_to_gene_symbol.R"

GO_BP_FILE <- "~/Documents/beehive/gtex_data_sample/gene_set_collections/GO_biological_process.gmt"
HALLMARK_FILE <- "~/Documents/beehive/gtex_data_sample/gene_set_collections/h.all.v7.1.symbols.gmt"

SAVE_PATH <- "/Users/andrewjones/Documents/beehive/gtex_image_analysis/experiments/cca/out/lung_ventilator/"


## ----- Load files for GSEA -------
source(GSEA_PATH)
source(GENE_NAME_PATH)

gsc_bp <- piano::loadGSC(GO_BP_FILE)
gsc_hallmark <- piano::loadGSC(HALLMARK_FILE)


## ----- Load CCA results --------

cca_coeffs <- read.csv(file.path(CCA_COEFFS_PATH, "pcca_exp_coeffs_shared.csv"), header = 1, row.names = 1)
all_genes_ensembl <- rownames(cca_coeffs)
all_genes <- all_genes_ensembl %>% 
  lapply(function(x) { strsplit(x, "[.]")[[1]][1] }) %>% as.character() %>% 
  ensembl_to_gene_symbol()

all_genes_unique <- all_genes %>% unique()

## ----- Run GSEA -------

VAR_IDX <- 38
gene_stats <- cca_coeffs[,VAR_IDX] %>% set_names(all_genes)
gene_stats <- gene_stats[!is.na(names(gene_stats))]
gene_stats <- gene_stats[all_genes_unique]


# Hallmark
gsea_out <- run_permutation_gsea(gsc_file = HALLMARK_FILE, gene_stats = gene_stats, gsc = gsc_hallmark, nperm = 1000)
colnames(gsea_out)[which(colnames(gsea_out) == "padj")] <- "adj_pval"
gsea_out$pathway <- lapply(gsea_out$pathway, function(x) {gsub("HALLMARK_", "", x)}) %>% as.character()
gsea_out$pathway <- lapply(gsea_out$pathway, function(x) {paste(strsplit(x, "_")[[1]], collapse = " ") }) %>% as.character()

print(gsea_out[gsea_out$adj_pval < 0.05,] %>%
        as.data.frame() %>%
        dplyr::select("pathway", "pval", "adj_pval", "NES") %>%
        dplyr::arrange(-abs(NES)) %>%
        head())

write.csv(x = gsea_out %>% dplyr::select("pathway", "pval", "padj", "NES"), 
          file = file.path(SAVE_PATH, sprintf("gsea_comp%s_shared.csv", VAR_IDX)))

  
# Down genes
hit_genes <- names(gene_stats)[order(gene_stats)[1:50]]
gsea_out <- run_fisher_exact_gsea(gsc_file = geneset_file, gsc = gsc_hallmark, hit_genes = hit_genes %>% unique(), all_genes = names(gene_stats) %>% unique())
gsea_out$pathway <- lapply(gsea_out$pathway, function(x) {gsub("HALLMARK_", "", x)}) %>% as.character()

print(gsea_out %>% dplyr::arrange(adj_pval) %>% head(10))


# Up genes
hit_genes <- names(gene_stats)[order(-gene_stats)[1:50]]
gsea_out <- run_fisher_exact_gsea(gsc_file = geneset_file, gsc = gsc_hallmark, hit_genes = hit_genes %>% unique(), all_genes = names(gene_stats) %>% unique())
gsea_out$pathway <- lapply(gsea_out$pathway, function(x) {gsub("HALLMARK_", "", x)}) %>% as.character()

print(gsea_out %>% dplyr::arrange(adj_pval) %>% head(10))

