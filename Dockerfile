FROM heatonresearch/jupyter-python-r:latest 
COPY pca_cox_test.py /pca_cox_test.py
COPY params.npy /params.npy
COPY mean.npy /mean.npy
COPY std.npy /std.npy
COPY pca.npy /pca.npy
COPY features.npy /features.npy
COPY evaluation.py /evaluation.py
COPY score_sc2.sh /score_sc2.sh
RUN pip install pandas
RUN pip install numpy



