# GNN Tangency Portfolio — Time-Varying Correlation Graph Extension

> Replication and extension of:
> Liu, B., Li, H., \& Kang, L. (2026). **Tangency portfolios using graph neural networks.** Neural Networks, 193, 108043. (https://www.sciencedirect.com/science/article/pii/S0893608025009232?via%3Dihub)

## Overview

This project extends the ideas presented in the paper Tangency Portfolios Using Graph Neural Networks. The goal of the original study is to improve portfolio construction by using relationships between stocks, rather than relying only on traditional statistical methods.

Traditional portfolio models, such as mean-variance optimisation, depend heavily on estimating expected returns and covariance matrices. These estimates become unstable when dealing with many assets. The paper addresses this by using a Graph Neural Network (GNN), where stocks are treated as nodes in a network and their relationships are encoded using an industry-based graph. This allows the model to learn portfolio weights directly, without explicitly estimating the covariance matrix, while aiming to maximise the Sharpe ratio.

## What the Original Paper Does

The original paper constructs a graph of stocks based on industry relationships, the graph remains fixed over time. The graph captures how stocks are economically related
This project extends the original approach by changing how the graph is built.

Instead of using a fixed industry-based graph, this code constructs a time-varying graph based on correlations between stock returns.

Key idea:

Stock relationships are not constant — they change over time.

So in this project:

A rolling window of past returns is used to calculate correlations between stocks
These correlations define the connections in the graph
The graph is updated over time, allowing it to adapt to changing market conditions

## Conclusion 
This repository contains a single-file Google Colab implementation of the GNN-based tangency portfolio from Liu et al. (2026), extended with a **time-varying correlation graph** that replaces the original paper's static industry-chain adjacency matrix.
