Students grades prediction project 

Authors: Bugra Duman, Kamil Matuszelanski

This project is aimed at prediciton of final grade in mathematics of students in Portuguese schools. 

Data can be downloaded from the site under this link: https://www.kaggle.com/dipam7/student-grade-prediction . 
The file "student-mat.csv" should be put in /data/ folder in order to run the notebooks.

Files structure:
- /data/ - here downloaded data should be put
- EDA.ipynb - checking the structure of the dataset, statistics and data visualisation. It also contains basic preprocessing of the dataset, so it has to be run before "Modeling" notebook.
- Modeling.ipynb - all modeling work done
- hurdle.py - implementation of hurdle model as scikit-learn estimator. Contents of this file is fully based on the codes under this link: https://geoffruddock.com/building-a-hurdle-regression-estimator-in-scikit-learn/ . 
