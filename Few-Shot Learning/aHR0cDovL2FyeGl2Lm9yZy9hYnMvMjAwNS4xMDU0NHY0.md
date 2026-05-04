# Cross-Domain Few-Shot Learning with Meta Fine-Tuning

John Cai, Shen Sheng Mei (2020)

## 🧩 Problem to Solve

본 논문은 CVPR 2020 Challenge에서 제안된 **Cross-Domain Few-Shot Learning (CD-FSL)** 벤치마크 문제를 해결하고자 한다. 일반적인 Few-Shot Learning (FSL) 방법론들은 훈련 데이터셋과 테스트 데이터셋이 동일한 분포에서 생성되었다는 가정을 전제로 한다. 그러나 실제 환경에서는 훈련 도메인과 테스트 도메인 사이에 심각한 **Domain Shift**가 발생하며, 이는 특징 추출기의 견고한 전이를 방해하여 성능을 저하시키는 원인이 된다.

특히 자연 이미지 외에도 의료 이미지, 위성 이미지, 색상이 없는 이미지 등 매우 이질적인 도메인 간의 일반화 능력을 평가하는 것이 본 연구의 핵심 목표이다. 즉, 제한된 수의 샘플(Few-shot)만으로도 전혀 새로운 도메인의 클래스를 정확하게 분류할 수 있는 모델을 구축하는 것이 목적이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델이 단순히 특징을 추출하는 것을 넘어, **'나중에 미세 조정(Fine-tuning)되기 적합한 상태'**로 훈련되도록 만드는 것이다. 주요 기여 사항은 다음과 같다.

1. **Meta Fine-Tuning**: First-order MAML(Model-Agnostic Meta-Learning) 알고리즘을 에피소드 훈련 과정에 통합하여, 테스트 도메인의 Support-set에서 효율적으로 미세 조정될 수 있는 초기 가중치(initial weights)를 학습한다.
2. **GNN과의 결합**: Meta Fine-Tuning 알고리즘을 Graph Neural Network (GNN)에 적용하여, Support-set과 Query 샘플 간의 비유클리드 구조적 관계를 활용한 특징 비교를 수행한다.
3. **테스트 단계의 데이터 증강**: 훈련 단계뿐만 아니라, 테스트 시의 Fine-tuning 과정에서도 Support-set에 데이터 증강(Data Augmentation)을 적용하여 성능을 향상시킨다.
4. **앙상블 전략**: 제안한 Meta Fine-Tuning GNN 모델과 수정된 Baseline Fine-tuning 모델을 결합한 앙상블 모델을 통해 최종 예측 정확도를 높인다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들을 기반으로 한다.

- **Graph Neural Networks (GNN)**: 단순한 유클리드 공간을 넘어 데이터의 유연한 표현을 가능하게 한다. FSL 맥락에서는 Support-set을 밀집 연결된 무방향 그래프로 표현하고, 레이블 정보를 Query 샘플로 전파하는 Belief Propagation 문제로 재구성하여 해결한다.
- **Model-Agnostic Meta-Learning (MAML)**: 빠른 적응(fast adaptation)이 가능한 내부 표현을 학습하는 방법론이다. 본 논문은 계산 비용을 줄이기 위해 2차 미분항을 무시하는 First-order 근사 알고리즘(예: Reptile)을 채택한다.
- **Domain Adaptation**: 초기 특징 층(early feature layers)은 고정하고 후반부 층(later feature layers)만 미세 조정하여, 일반적인 고수준 특징은 유지하면서 도메인 특화 특징만 재학습하는 방식이 사용된다.
- **Ensemble Methods**: 서로 다른 아키텍처를 가진 모델들의 예측 점수를 평균 내어 분산을 줄이고 정확도를 높이는 일반적인 기법을 활용한다.

## 🛠️ Methodology

### 1. Graph Neural Network (GNN) 모듈

특징 추출기를 통해 얻은 $d$-차원 벡터를 선형 층을 통해 $k$-차원으로 투영한다. 이후 GNN은 $N_s$개의 Support 샘플과 1개의 Query 샘플을 정점으로 하는 그래프 입력을 받는다.

- **Graph Convolution (GC)**: $\text{GC}(\cdot)$ 연산을 통해 지역 신호를 선형 연산하며, 결과물 $X^{(k+1)} = \text{GC}(X^{(k)})$를 생성한다.
- **Edge Feature**: 각 정점 벡터 간의 절대 차이(absolute difference)를 MLP(Multi-Layer Perceptron)에 입력하여 에지 가중치를 학습한다.

### 2. Meta Fine-Tuning 절차

기존의 pre-trained 모델을 단순히 Fine-tuning하는 대신, MAML을 통해 Fine-tuning에 최적화된 초기 가중치를 찾는다.

**학습 알고리즘 (Algorithm 1):**

1. 특징 추출기 가중치 $\phi_f$와 메트릭 학습 모듈 가중치 $\phi_m$을 초기화한다.
2. 각 에피소드마다 다음을 반복한다:
    - 특징 추출기의 앞부분 $L-k$개 층을 동결(Freeze)한다.
    - **Inner Loop (Fine-tuning simulation)**: Support 샘플에 대해 선형 분류기를 사용하여 손실 함수 $\mathcal{L}_s$를 계산하고, SGD 또는 Adam을 통해 마지막 $k$개 층의 가중치를 업데이트하여 $\tilde{\phi}_f^{(k)}$를 얻는다.
    - **Outer Loop (Meta-update)**: 업데이트된 가중치 $\tilde{\phi}_f$를 사용하여 Query 샘플에 대한 손실 $\mathcal{L}(\tilde{\phi}_f, \phi_m)$을 계산한다.
    - 이 손실을 바탕으로 전체 모델 파라미터에 대한 그래디언트 $g_{f(L-k)}, g_{f(k)}, g_m$을 계산하고, 학습률 $\theta$를 사용하여 초기 파라미터를 업데이트한다.

본 논문에서는 ResNet10을 백본으로 사용하며, 마지막 ResNet 블록(최근 2개 층)을 업데이트 대상으로 설정하였다.

### 3. Data Augmentation 및 앙상블

- **Data Augmentation**: 테스트 단계의 Fine-tuning 시, Support 이미지에서 17장의 추가 이미지를 생성(Jitter, Random Crop, Horizontal Flip)한다. 이때 원본 이미지에 더 높은 가중치를 두어 모델이 노출되는 빈도를 높인다.
- **Ensemble**: 수정된 Baseline 모델과 Meta Fine-Tuning GNN 모델의 예측 점수를 Softmax로 정규화하여 합산한 후, $\text{argmax}$를 통해 최종 클래스를 결정한다.
- **메모리 최적화**: 50-shot의 경우 GNN의 공간 복잡도가 $O(n^2)$으로 급증하므로, 2개의 Support 샘플 특징 벡터를 평균 내어 25개의 노드로 축소하여 16GB V100 GPU 메모리에 맞춘다.

## 📊 Results

### 실험 설정

- **훈련 데이터**: miniImageNet
- **테스트 데이터**: CropDisease, EuroSAT, ISIC, ChestX (총 4개 도메인)
- **훈련 조건**: 기본 훈련 400 epoch $\rightarrow$ Meta Fine-tuning 200 epoch.

### 정량적 결과

제안된 모델은 기존 벤치마크의 최적 모델인 'Ft-Last1'과 비교하여 비약적인 성능 향상을 보였다.

- **평균 정확도**: 제안 모델 **73.78%** vs. 기존 벤치마크 **67.27%** (약 **6.51%p 향상**)
- **Shot 수에 따른 성능 향상**:
  - 5-shot: 8.48% 향상 (가장 큰 폭의 향상)
  - 20-shot: 4.68% 향상
  - 50-shot: 6.38% 향상

### 분석 결과

데이터 증강은 샘플 수가 매우 적은 5-shot에서 가장 큰 효과를 보였으며, Fine-tuning은 샘플 수가 많은 50-shot에서 더 큰 효과를 나타냈다.

## 🧠 Insights & Discussion

### 강점 및 발견

- **Meta Fine-Tuning의 효용성**: 모델이 '어떻게 적응해야 하는가'를 미리 학습함으로써, 단순한 전이 학습보다 빠른 적응이 가능함을 입증하였다.
- **도메인 유사성의 영향**: 분석 결과, Meta Fine-Tuning은 훈련 도메인(miniImageNet)과 유사한 도메인에서 더 높은 성능 향상을 보였다.

### 한계 및 비판적 해석

- **극단적 도메인 시프트에 대한 취약성**: ChestX와 같이 훈련 도메인과 매우 거리가 먼 도메인에서는 오히려 Simple Fine-tuning + Data Augmentation 조합이 더 효과적이었다. 이는 miniImageNet에서 '미세 조정하는 법'을 배운 것이, 너무 이질적인 도메인으로의 전이에는 오히려 제약이 되었을 가능성을 시사한다.
- **가정의 한계**: 본 연구는 훈련 데이터셋을 miniImageNet 하나로 제한하였으나, 실제로는 더 다양한 데이터셋을 사용하여 Meta Fine-Tuning을 수행한다면 도메인 불가지론적(domain-agnostic)인 적응 능력을 키울 수 있을 것으로 판단된다.

## 📌 TL;DR

본 논문은 **MAML 기반의 Meta Fine-Tuning**과 **GNN**, 그리고 **테스트 단계의 데이터 증강**을 결합하여 Cross-Domain Few-Shot Learning 성능을 높인 연구이다. 특히 모델이 미세 조정에 최적화된 초기 가중치를 학습하도록 유도함으로써, 기존 벤치마크 대비 평균 정확도를 **6.51%p 향상**시켰다. 이 연구는 단순한 특징 추출을 넘어 '적응 방법 자체를 학습'하는 메타 학습의 실용성을 보여주었으며, 향후 더 다양한 도메인을 포함한 메타 학습 모델 설계의 필요성을 제시한다.
