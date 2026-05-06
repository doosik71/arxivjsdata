# SelfGNN: Self-supervised Graph Neural Networks without explicit negative sampling

Zekarias T. Kefato, Sarunas Girdzijauskas (2021)

## 🧩 Problem to Solve

현실 세계의 그래프 데이터는 대부분 레이블이 없거나 극소수만 레이블링되어 있다. 데이터를 수동으로 레이블링하는 작업은 매우 비용이 많이 들고 어려운 작업이므로, 지도 학습이나 준지도 학습(Semi-supervised learning)에 필적하는 성능을 낼 수 있는 강력한 비지도 학습(Unsupervised learning) 기법이 필요하다.

최근 컴퓨터 비전(CV)과 자연어 처리(NLP) 분야에서는 대조 학습(Contrastive Learning, CL) 기반의 자기지도 학습(Self-supervised Learning, SSL)이 큰 성과를 거두고 있다. 일반적인 대조 학습 프레임워크는 동일한 객체의 서로 다른 증강 뷰(Augmented view) 간의 유사성은 최대화하고, 서로 다른 객체인 부정 샘플(Negative sample) 간의 유사성은 최소화하는 방식을 취한다. 그러나 그래프 신경망(GNN)에서의 대조 학습은 명시적인 부정 샘플링(Explicit negative sampling)에 의존하는 경우가 많으며, 이는 계산 복잡도를 높이고 샘플링 전략에 따라 성능 변동이 심하다는 문제가 있다.

본 논문의 목표는 명시적인 부정 샘플 없이도 강력한 표현 학습이 가능한 자기지도 학습 GNN인 **SelfGNN**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 명시적인 부정 샘플링 없이 Batch Normalization을 통해 암시적인 부정 샘플(Implicit negative samples) 효과를 유도하여 모델의 붕괴(Collapse)를 방지하고 효율적인 학습을 달성한 점이다.

또한, 기존의 그래프 대조 학습이 주로 위상 구조 증강(Topological Augmentation, TA)에 집중했던 것과 달리, 계산 비용이 매우 낮은 네 가지의 특징 증강(Feature Augmentation, FA) 기법을 제안하고 이것이 TA와 대등한 성능을 보임을 입증하였다.

## 📎 Related Works

### Graph Neural Networks (GNNs)

GCN, GAT, GraphSage와 같은 모델들은 그래프의 위상을 따라 노드 특징을 전파하여 표현을 학습한다. GCN은 인접 행렬을 이용한 단순 집계 방식을 사용하며, GAT는 어텐션 메커니즘을 통해 가중치를 학습한다. GraphSage는 이웃 샘플링을 통해 확장성을 개선하였다. 또한 ClusterGCN이나 GraphSaint와 같은 서브그래프 샘플링 방식은 대규모 그래프에서도 효율적인 미니배치 학습을 가능하게 한다.

### Self-Supervised Learning (SSL)

SSL은 사전 학습(Pre-training) 후 미세 조정(Fine-tuning)하는 방식과 대조 학습(Contrastive Learning) 방식으로 나뉜다. 그래프 영역에서는 DGI(Deep Graph Infomax)나 MvGrl과 같은 대조 학습 기법들이 제안되었으나, 이들은 대개 명시적인 부정 샘플을 생성하여 이를 밀어내는 방식을 사용한다. 반면 본 논문은 CV 분야의 최신 경향을 반영하여 명시적 샘플링 없이도 학습 가능한 구조를 GNN에 도입하였다.

## 🛠️ Methodology

### 전체 시스템 구조

SelfGNN은 **Siamese Network** 구조를 모방하여 학생(Student) 네트워크와 교사(Teacher) 네트워크라는 두 개의 병렬 네트워크로 구성된다. 두 네트워크 모두 동일한 GNN 인코더($f_{\theta}$)를 사용하지만, 파라미터 업데이트 방식에서 차이가 있다.

1. **학생 네트워크 (Student):** 경사 하강법(Gradient Descent)을 통해 파라미터 $\theta$를 직접 업데이트한다. 또한 인코더 출력 뒤에 예측 블록(Prediction block, MLP 형태의 $g_{\theta}$)이 추가되어 교사 네트워크의 출력을 예측하도록 학습한다.
2. **교사 네트워크 (Teacher):** 기울기를 직접 계산하지 않고, 학생 네트워크 파라미터의 지수 이동 평균(Exponential Moving Average, EMA)을 통해 파라미터 $\phi$를 업데이트한다.
    $$\phi \leftarrow \tau \phi + (1-\tau)\theta$$
    여기서 $\tau$는 감쇠율(Decay rate)이다.

### 학습 목표 및 손실 함수

SelfGNN은 두 개의 서로 다른 증강 뷰 $\mathcal{G}_1, \mathcal{G}_2$를 입력으로 받는다. 학생 네트워크는 $\mathcal{G}_1$을 처리하여 예측값 $g_{\theta}(f_{\theta}(\mathcal{G}_1))$을 내놓고, 교사 네트워크는 $\mathcal{G}_2$를 처리하여 표현값 $f_{\phi}(\mathcal{G}_2)$를 생성한다. 두 벡터 사이의 평균 제곱 오차(MSE)를 최소화하는 방향으로 학습하며, 손실 함수 $L_{\theta}$는 다음과 같다.

$$L_{\theta} = 2 - 2 \cdot \frac{\langle g_{\theta}(f_{\theta}(\mathcal{G}_1)), f_{\phi}(\mathcal{G}_2) \rangle}{\|g_{\theta}(f_{\theta}(\mathcal{G}_1))\|_F \cdot \|f_{\phi}(\mathcal{G}_2)\|_F}$$

이때, 학생 네트워크의 예측 블록에 포함된 **Batch Normalization**이 배치 내의 다른 샘플들을 암시적인 부정 샘플로 활용하게 함으로써, 모든 노드가 동일한 값으로 수렴하는 trivial solution(Collapse)을 방지한다.

### 데이터 증강 (Data Augmentation)

논문은 두 가지 관점의 증강 기법을 제안한다.

1. **위상 증강 (Topological Augmentation, TA):** 그래프의 고차 구조를 활용한다.
    * **PPR (Personalized PageRank):** 루트 기반 랜덤 워크를 이용한 전파.
    * **HK (Heat-Kernel):** 열 확산 과정을 이용한 전파.
    * **Katz-index:** 모든 경로의 가중 합을 계산하며 경로 길이에 따라 페널티를 부여한다.
2. **특징 증강 (Feature Augmentation, FA):** 노드 특징 행렬 $X$를 조작한다.
    * **Split:** 특징 차원을 절반으로 나누어 두 개의 뷰를 생성한다.
    * **Standardize:** Z-score 표준화를 적용하여 스케일을 조정한다.
    * **LDP (Local Degree Profile):** 노드의 지역 차수 통계량을 이용해 새로운 특징을 생성한다.
    * **Paste:** 원래 특징과 LDP 특징을 결합한다.

### ClusterSelfGNN

TA 기법은 행렬 역행렬 계산 등으로 인해 $O(N^3)$의 시간 복잡도가 발생하고 GPU 메모리 부족 문제를 야기한다. 이를 해결하기 위해 METIS를 이용해 그래프를 클러스터링하고, 각 서브그래프 단위로 TA를 적용하는 **ClusterSelfGNN** 변형 모델을 제안한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Cora, Citeseer, Pubmed(인용 네트워크), Photo, Computers(상품 네트워크), CS, Physics(저자 네트워크) 등 7개 데이터셋을 사용하였다.
* **평가 지표:** 노드 분류(Node Classification) 작업에서의 정확도(Accuracy)를 측정하였으며, 학습 후 로지스틱 회귀 분류기를 사용하여 평가하였다.
* **비교 대상:** 준지도 학습 GNN(GCN, GAT, GraphSage) 및 자기지도 학습 GNN(DGI, MvGrl)과 비교하였다.

### 주요 결과

1. **성능:** SelfGNN은 준지도 학습 및 기존 자기지도 학습 모델보다 일관되게 높은 성능을 보였으며, 일부 데이터셋(인용 네트워크)에서는 지도 학습 모델과 대등하거나 더 나은 성능을 기록하였다.
2. **FA vs TA:** 특징 증강(FA) 기법, 특히 **Split**과 **Standardize**가 위상 증강(TA)과 유사한 성능을 낸다는 점을 확인하였다. TA는 계산 비용이 매우 높지만, FA는 상수 시간 복잡도로 수행 가능하므로 훨씬 효율적이다.
3. **효율성:** ClusterSelfGNN은 메모리 부족 문제로 인해 Full-batch TA를 적용할 수 없었던 대규모 그래프(Physics 등)에서도 안정적으로 작동하며, 성능 하락 또한 매우 적었다.
4. **비교 분석:** MvGrl 및 DGI와 같은 기존 SSL 모델들과 비교했을 때, SelfGNN이 대부분의 데이터셋에서 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### BatchNorm의 결정적 역할

절제 연구(Ablation Study)를 통해 BatchNorm이 제거되었을 때 모델의 성능이 급격히 떨어지고 불안정해짐을 확인하였다. 이는 BatchNorm이 단순히 정규화를 수행하는 것을 넘어, 대조 학습에서 필수적인 '부정 샘플'의 역할을 암시적으로 수행하고 있음을 시사한다. LayerNorm은 BatchNorm과 같은 효과를 내지 못했다.

### Projection Head의 불필요성

CV 분야의 대조 학습(예: SimCLR)에서는 인코더 뒤에 Projection head를 두는 것이 성능 향상에 중요하다고 알려져 있으나, 본 연구의 GRL(Graph Representation Learning) 실험에서는 Projection head가 유의미한 성능 향상을 가져오지 않았으며 오히려 분산을 높이는 결과를 보였다.

### 한계점 및 향후 과제

SelfGNN은 두 개의 네트워크를 메모리에 동시에 올려야 하므로 대규모 그래프에서 메모리 병목 현상이 발생한다. ClusterSelfGNN으로 이를 일부 완화했으나, 더 근본적이고 원칙적인 메모리 효율화 방안에 대한 연구가 향후 필요하다.

## 📌 TL;DR

본 논문은 명시적인 부정 샘플링 없이 **Batch Normalization을 통해 암시적인 대조 학습**을 수행하는 **SelfGNN**을 제안하였다. 특히 계산 비용이 큰 위상 증강 대신 효율적인 **특징 증강(Feature Augmentation)** 기법들을 도입하여, 계산 효율성을 높이면서도 최신 지도/자기지도 학습 GNN에 필적하는 성능을 달성하였다. 이 연구는 그래프 데이터의 레이블 부족 문제를 해결하고, 대규모 그래프에 적용 가능한 효율적인 자기지도 학습 프레임워크를 제공한다는 점에서 의의가 있다.
