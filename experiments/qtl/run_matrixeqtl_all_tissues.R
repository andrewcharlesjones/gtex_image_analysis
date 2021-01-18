library(MatrixEQTL)
library(magrittr)


# Load small subset of data

if (Sys.info()["user"] == "andrewjones") {
  DATA_DIR <- "/Users/andrewjones/Documents/beehive/gtex_image_analysis/data_processing/qtl/data/tissues"
  SAVE_DIR <- "/Users/andrewjones/Documents/beehive/gtex_image_analysis/experiments/qtl/out"
  PVAL_THRESHOLD = 0.05
} else {
  DATA_DIR <- "/tigress/aj13/gtex_image_analysis/qtl/data/tissues"
  SAVE_DIR <- "/tigress/aj13/gtex_image_analysis/qtl/out"
  PVAL_THRESHOLD = 1e-5
}

all_tissues <- list.files(DATA_DIR) # c("Thyroid")

for (curr_tissue in all_tissues) {
  
  data_files <- list.files(file.path(DATA_DIR, curr_tissue))
  
  comp_nums_seen = c()
  for (curr_file in data_files) {
    split_file <- strsplit(curr_file, "_")[[1]]
    curr_comp_num = as.integer(strsplit(split_file[3], '[.]')[[1]][1])
    
    if (curr_comp_num %in% comp_nums_seen) {
      next
    }
    
    comp_nums_seen <- c(comp_nums_seen, curr_comp_num)
    if (split_file[1] == "phenotype") {
      phenotype_path <- file.path(DATA_DIR, curr_tissue, curr_file)
      genotype_path <- file.path(DATA_DIR, curr_tissue, paste(c("genotype", split_file[2:3]), collapse = "_"))
    } else {
      genotype_path <- file.path(DATA_DIR, curr_tissue, curr_file)
      phenotype_path <- file.path(DATA_DIR, curr_tissue, paste(c("phenotype", split_file[2:3]), collapse = "_"))
    }
    
    phenotype_data <- read.csv(phenotype_path, row.names = 1)
    
    sprintf("Number of samples: %s", ncol(phenotype_data))
    
    
    ## Do eQTL analysis
    
    snps = SlicedData$new()
    snps$fileDelimiter = ",";
    snps$fileOmitCharacters = "NA";
    snps$fileSkipRows = 1;
    snps$fileSkipColumns = 1;       # one column of row labels; MAYBE WE CAN CHANGE THIS to align things properly
    snps$fileSliceSize = 2000;      # read file in pieces of 2,000 rows
    snps$LoadFile(genotype_path);  #takes a very long time ~ 15 min
    
    # CREATE THE GENEEXPRESSION file
    gene = SlicedData$new()
    gene$CreateFromMatrix(as.matrix(phenotype_data));
    
    # OTHER PARAMS
    useModel = modelLINEAR;
    
    dir.create(file.path(SAVE_DIR, curr_tissue), showWarnings = FALSE)
    output_file <- file.path(SAVE_DIR, curr_tissue, paste(c("qtl_output_", curr_comp_num, ".txt"), collapse = ""))
    
    me = Matrix_eQTL_engine(snps = snps, 
                            gene = gene, 
                            cvrt = SlicedData$new(), 
                            output_file_name = output_file, 
                            pvOutputThreshold = PVAL_THRESHOLD, 
                            useModel = useModel, 
                            errorCovariance = numeric(), 
                            verbose = TRUE,
                            pvalue.hist = 100,
                            min.pv.by.genesnp = FALSE,
                            noFDRsaveMemory = FALSE)

      }
  
  
}
