from baseline.GAE.gae import GAE
from baseline.GAMA.gama import GAMA
from baseline.GRASPED.grasped import GRASPED
from baseline.LAE.lae import LAE
from baseline.Sylvio import W2VLOF
from baseline.VAE.vae import VAE
from baseline.VAEOCSVM.vaeOCSVM import VAEOCSVM
from baseline.bezerra import NaiveAnomalyDetector, SamplingAnomalyDetector
from baseline.binet.binet import BINetv2, BINetv3
from baseline.boehmer import LikelihoodPlusAnomalyDetector
from baseline.leverage import Leverage
from novel.dae.dae import DAE


def All_original_models_finetuned():
    # RCVDB: Full Configuration
    run_name = 'Original_Finetuned'
    ads = [
        dict(ad=LikelihoodPlusAnomalyDetector),  ## Multi-perspective, attr-level    --- Multi-perspective anomaly detection in business process execution events (extended to support the use of external threshold)
        dict(ad=NaiveAnomalyDetector),  # Control flow, trace-level    ---Algorithms for anomaly detection of traces in logs of process aware information systems
        dict(ad=SamplingAnomalyDetector),  # Control flow, trace-level    ---Algorithms for anomaly detection of traces in logs of process aware information systems
        dict(ad=DAE, fit_kwargs=dict(epochs=100, batch_size=64)),  ## Multi-perspective, attr-level    ---Analyzing business process anomalies using autoencoders
        dict(ad=BINetv3, fit_kwargs=dict(epochs=20, batch_size=64)), ## Multi-perspective, attr-level  ---BINet: Multi-perspective business process anomaly classification
        dict(ad=BINetv2, fit_kwargs=dict(epochs=20, batch_size=64)), ## Multi-perspective, attr-level  ---BINet: Multivariate business process anomaly detection using deep learning
        dict(ad=GAMA,ad_kwargs=dict(n_epochs=20)), ## Multi-perspective, attr-level    ---GAMA: A Multi-graph-based Anomaly Detection Framework for Business Processes via Graph Neural Networks
        dict(ad=VAE), ## Multi-perspective, attr-level 自己修改后使其能够检测attr-level      ---Autoencoders for improving quality of process event logs
        dict(ad=LAE), ## Multi-perspective, attr-level  自己修改后使其能够检测attr-level      ---Autoencoders for improving quality of process event logs
        dict(ad=GAE), ## Multi-perspective, trace-level       ---Graph Autoencoders for Business Process Anomaly Detection
        dict(ad=GRASPED), ## Multi-perspective, attr-level    ---GRASPED: A GRU-AE Network Based Multi-Perspective Business Process Anomaly Detection Model
        dict(ad=Leverage), # Control flow, trace-level       ---Keeping our rivers clean: Information-theoretic online anomaly detection for streaming business process events
        dict(ad=W2VLOF), # Control flow, trace-level     ---Anomaly Detection on Event Logs with a Scarcity of Labels
        dict(ad=VAEOCSVM) # Control flow, trace-level   ---Variational Autoencoder for Anomaly Detection in Event Data in Online Process Mining
    ]
    return ads, run_name