## Variational Autoencoders for Anomaly Detection With Tensorflow On SageMaker

This repository contains code for detecting anomalies using Variational Autoencoders and deploying multiple models on one SageMaker Rest API. The deep learning framework in use is Tensorflow.

## Environment Setup
First, run the following commands in your terminal to create a new conda environment named `tf2-p36` which has the required dependencies.

```bash
bash setup_env.sh
conda activate tf2-p36 # activate the environment
```

Then, run the following commands to add `tf2-p36` to IPython Kernel for Jupyter
```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=tf2-p36
```

Finally, choose `tf2-p36` as the Kernel where to run the notebooks on

## Repo Structure
```bash
+-- notebooks
|   +-- VAE_AnomalyDetection_Tensorflow.ipynb
+-- src
|   +-- config.py
|   +-- model_def.py
|   +-- train.py
+-- environment.yml
+-- README.md
+-- setup_env.sh
+-- CODE_OF_CONDUCT.md
+-- CONTRIBUTING.md
+-- LICENSE
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

