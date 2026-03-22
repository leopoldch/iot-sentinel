# TODO — IoT IDS Project

## Chosen algorithms
- [ ] **Forest:** Isolation Forest
- [ ] **Deep Learning:** LSTM
- [ ] **Transformer:** TabTransformer
- [ ] **Federated Learning:** FedAvg + MLP
- [ ] **Reinforcement Learning:** DQN

## Order of work
1. [ ] Build a unified preprocessing pipeline for Edge-IIoTset, TON-IoT, and CIC IoT
2. [ ] Train and evaluate **Isolation Forest** as the unsupervised baseline
3. [ ] Train and evaluate **LSTM** as the main deep learning baseline
4. [ ] Train and evaluate **TabTransformer** for tabular attention-based learning
5. [ ] Implement **FedAvg** with a small **MLP** client model
6. [ ] Implement **DQN** as the RL exploratory branch
7. [ ] Compare all models with Accuracy, Precision, Recall, F1, ROC-AUC
8. [ ] Write the final comparison: performance, cost, complexity, IoT relevance

## Notes
- [ ] Start with **binary classification**, then extend to **multiclass**
- [ ] Keep the **same train/val/test split** for all models
- [ ] Treat **RL as exploratory**, not the core method