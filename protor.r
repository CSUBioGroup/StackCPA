
# install.packages('protr')
#obtain data
library(protr)
library(protr)

args <- commandArgs(trailingOnly = TRUE)
data <- read.csv(args[1],header=TRUE, encoding="UTF-8")

# data <- read.csv("~/PocketInfo/test_project/datasets/PDBbind_input_data.csv",header=TRUE, encoding="UTF-8")
data <-na.omit(data)
id = subset(data,select=c(protein_id))
# id = subset(data,select=c(pdbid))

fasta = subset(data,select=c(pocket_seq))

# load FASTA files
ctdc = t(sapply(as.character(fasta[ ,1]), extractCTDC))

ctdt = t(sapply(as.character(fasta[ ,1]), extractCTDT))
ctdd = t(sapply(as.character(fasta[ ,1]), extractCTDD))
ctraid = t(sapply(as.character(fasta[ ,1]), extractCTriad))
socn = t(sapply(as.character(fasta[ ,1]),extractSOCN)) 

temp_test = cbind(ctdc,ctdt,ctdd,ctraid,socn)

# write.csv(temp_test,file='C:/Users/Dell/Desktop/kiba_protein30_protr.csv',quote=F)
write.csv(temp_test,file=args[2],quote=F)
