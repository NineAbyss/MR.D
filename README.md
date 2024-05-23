# MR.D
This is the official implementation of paper: **Toward Fairer Review: Detection of Collusive Groups in Peer Review Bidding Network**.

# Requirements
python==3.10
torch_geometric==2.4.0
pytorch==2.1.1+cu121

# Datasets
To construct datasets, run

```
python inject.py
```
# Experiments
To reproduce the main results, run

```
bash grid_search_ours.sh

```

# Acknoledgement
This code is based on Xing Ai's [work](https://github.com/XingAi96/Group_level_Graph_Anomaly_Detection). We are appreciated for his help.