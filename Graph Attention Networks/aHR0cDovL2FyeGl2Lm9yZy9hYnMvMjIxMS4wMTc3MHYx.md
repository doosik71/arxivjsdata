# Exploring Explainability Methods for Graph Neural Networks
Harsh Patel, Shivam Sahni

## Problem to Solve
딥러닝 모델, 특히 복잡한 상호 연결 정보를 인코딩하는 그래프 신경망(GNN)은 "블랙 박스"로 간주됩니다. GNN의 작동 방식에 대한 투명성 부족은 신뢰와 개선을 방해합니다. 이 연구는 GNN, 특히 **그래프 어텐션 네트워크 (GAT)**의 예측을 설명하는 방법을 탐구하여, 모델이 특정 결정을 내리는 이유를 이해하는 것을 목표로 합니다. 구체적으로 **슈퍼픽셀 이미지 분류** 작업에 설명 가능성 기법을 적용하고 그 성능을 평가합니다.

## Key Contributions
*   **그래프 어텐션 네트워크 (GAT)**를 이용한 **슈퍼픽셀 이미지 분류** 작업에 널리 사용되는 설명 가능성 접근법의 적용 가능성을 시연했습니다.
*   **MNIST**, **Fashion-MNIST**, **CIFAR-10** 세 가지 데이터셋에 대해 다양한 설명 기법의 질적 및 양적 성능을 평가했습니다.
*   설명 기법의 정량적 평가를 위한 측정 지표로 **Fidelity**를 활용하고, **`Guided-Backpropagation (GBP)`**이 가장 우수한 성능을 보임을 확인했습니다.
*   이미지 내 슈퍼픽셀 수 변화가 분류 정확도와 설명 가능성 시각화에 미치는 영향을 분석했습니다.
*   GNN, 특히 GAT의 설명 가능성에 대한 새로운 통찰력을 제공했습니다.

## Methodology
*   **이미지 전처리**:
    *   `SLIC (Simple Linear Iterative Clustering)` 알고리즘을 사용하여 이미지에서 **슈퍼픽셀**을 생성합니다.
    *   각 슈퍼픽셀은 그래프의 **노드**가 되며, 인접한 슈퍼픽셀은 **엣지**를 형성하여 **지역 인접 그래프(Region Adjacency Graph)**를 구성합니다.
    *   이 그래프는 GNN 모델의 입력으로 사용됩니다.
*   **GNN 모델**:
    *   **멀티 헤드 그래프 어텐션 네트워크 (Multi-headed Graph Attention Network, GAT)**를 사용하여 슈퍼픽셀 이미지 분류를 수행합니다.
    *   GAT는 자체 어텐션 메커니즘을 통해 각 이웃의 기여도에 대한 상대적 중요성($e_{ij}$)을 학습하고, 소프트맥스($\alpha_{ij}$)를 사용하여 정규화된 어텐션 계수를 계산합니다.
    *   $$ e^{(l)}_{ij} = \text{ReLU}(\vec{a}^{(l)T}(W^{(l)}h^{(l)}_i \Vert W^{(l)}h^{(l)}_j)) $$
    *   $$ \alpha^{(l)}_{ij} = \frac{\exp(e^{(l)}_{ij})}{\sum_{k \in N(i)}\exp(e^{(l)}_{ik})} $$
    *   여러 어텐션 헤드를 사용하여 학습 안정성을 향상시킵니다.
    *   $$ h^{(l+1)}_i = \Vert_{K=1}^K \sigma\left(\sum_{j \in N(i)} \alpha^{(l)}_{ij} z^{(l)}_j\right) $$
*   **설명 가능성 방법**:
    *   `Contrastive Gradient-based Saliency Maps (CGSM)`: 입력에 대한 출력의 기울기에 ReLU를 적용하여 중요도를 측정합니다. ($L_c^{\text{Gradient}} = \Vert \text{ReLU}(\frac{\partial y_c}{\partial x}) \Vert$)
    *   `Class Activation Mapping (CAM)`: 마지막 컨볼루션 계층의 특징 맵과 전역 평균 풀링을 사용합니다 (특정 아키텍처 제약). ($L_c^{\text{CAM}}[i,j] = \text{ReLU}(\sum_k w_k^c F_{k,i,j})$)
    *   `Grad-Class Activation Mapping (Grad-CAM)`: CAM의 아키텍처 제약을 완화하여 역전파된 기울기를 가중치로 사용합니다. ($\alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y_c}{\partial F_{k,i,j}}$, $L_c^{\text{Grad-CAM}}[i,j] = \text{ReLU}(\sum_k \alpha_k^c F_{k,i,j})$)
    *   `Guided-Backpropagation (GBP)`: 역전파 과정에서 음의 기울기를 제거하여 양의 기여도에 집중합니다.
*   **정량적 평가**:
    *   **Fidelity** 점수를 사용합니다. 이는 설명 기법을 통해 식별된 중요한 특징(살리언시 값이 0.01 이상인 노드)을 가렸을 때 분류 정확도가 얼마나 감소하는지를 측정합니다.

## Results
*   **모델 정확도**:
    *   CIFAR-10: 49.73%
    *   MNIST (사전 학습): 97.8%
    *   Fashion MNIST (사전 학습): 89.6%
*   **설명 가능성 성능 (정성적 & 정량적)**:
    *   **`Guided-Backpropagation (GBP)`**가 모든 방법 중 가장 높은 **Fidelity** 점수를 보였습니다. 이는 `GBP`가 식별한 중요한 영역을 가릴 때 분류 오류가 가장 많이 발생함을 의미합니다.
    *   질적으로, `GBP`는 숫자를 형성하는 모든 노드를 가장 효과적으로 포착하며 노이즈가 적었습니다.
    *   `Vanilla Backpropagation` (CGSM)은 노이즈가 가장 많았고, `Guided Grad-CAM`은 노이즈가 적었지만 모든 중요한 노드를 완전히 포착하지 못했습니다.
    *   이미지 내 슈퍼픽셀 수가 증가할수록 (25개에서 150개로) 예측 노이즈(객체 외부 영역)가 감소하여 시각화가 더 명확해졌습니다.
    *   슈퍼픽셀 수 변화가 정확도에 미치는 영향: 슈퍼픽셀 수가 너무 적으면(예: 25개) 정보 손실로 인해 정확도가 크게 감소(MNIST 35%)하며, 너무 많으면(예: 150개) 복잡성이 증가하고 지각적 의미가 손실되어 정확도가 다시 감소했습니다.
    *   CIFAR-10 데이터셋의 시각화 결과는 설명 시각화의 실루엣이 이미지의 객체와 일치하여 설명의 품질을 보장했습니다.