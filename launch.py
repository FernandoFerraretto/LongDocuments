import os

#list bucket files
#os.system('gsutil ls gs://neuralresearcher_data/fernando_ferraretto/bkp')
#os.system('gsutil ls gs://neuralresearcher_data/fernando_ferraretto')
#os.system('gsutil ls gs://neuralresearcher_data/fernando_ferraretto/Datasets')
os.system('gsutil ls gs://neuralresearcher_data/fernando_ferraretto/Datasets/dataset_msmarco/V04')

#download to VM    
#os.system('gsutil cp -r gs://neuralresearcher_data/fernando_ferraretto/Datasets/dataset_harvardnews .')
#os.system('gsutil cp -r gs://neuralresearcher_data/fernando_ferraretto/HarvardNews .')
#os.system('gsutil cp -r gs://neuralresearcher_data/fernando_ferraretto/Datasets/dataset_msmarco/V04/msmarco_longnoise200.csv .')

#os.system('gsutil cp -r gs://neuralresearcher_data/fernando_ferraretto/MSMarco .')

#upload to GBUCKET
os.system('gsutil cp -r dataset_harvardnews/V03/NewsArticles_bm25_t3_600tk.csv gs://neuralresearcher_data/fernando_ferraretto/Datasets/dataset_harvardnews')

#os.system('gsutil cp -r MSMarco gs://neuralresearcher_data/fernando_ferraretto/bkp')

#delete no GBUCKET
#os.system('gsutil rm -r gs://neuralresearcher_data/fernando_ferraretto/bkp')
# os.system('gsutil rm -r gs://neuralresearcher_data/fernando_ferraretto/Model_AAN')

#move no GBUCKET
#os.system('gsutil mv gs://neuralresearcher_data/fernando_ferraretto/Datasets/annotations_dataset.txt gs://neuralresearcher_data/fernando_ferraretto/Datasets/dataset_gwikimatch/annotations_dataset.txt')