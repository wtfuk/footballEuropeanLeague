FROM jupyter/datascience-notebook

RUN pip install --upgrade pip

RUN pip install future ipython ipykernel lxml pandas-profiling requests sklearn tabulate torch torchvision tpot

COPY . /notebooks

WORKDIR /notebooks

EXPOSE 8080

CMD ["jupyter-lab", "--port=8080", "--ip=0.0.0.0", "--allow-root"]
