# FedFN: Feature Normalization for Alleviating Data Heterogeneity Problem in Federated Learning

Seongyoon Kim, Gihun Lee, Jaehoon Oh, Se-Young Yun (2023)

## 🧩 Problem to Solve

본 논문은 연합 학습(Federated Learning, FL) 환경에서 발생하는 데이터 이질성(Data Heterogeneity) 문제로 인한 성능 저하를 해결하고자 한다. 분산된 클라이언트들이 서로 다른 데이터 분포를 가지는 Non-IID 환경에서는 글로벌 모델의 성능이 크게 하락하는 경향이 있다.

저자들은 기존 연구들이 주로 분류기 가중치(Classifier weight)의 편향에 집중했던 것과 달리, 데이터 이질성이 심화될수록 분류기 가중치보다 특징 표현(Feature representation)의 품질이 더 심각하게 저하된다는 점에 주목한다. 특히, 로컬 모델에서 학습된 관찰 클래스(Observed classes)의 특징 노름(Feature norm)과 관찰되지 않은 클래스(Unobserved classes)의 특징 노름 사이의 간극이 넓어지며, 이것이 로컬 모델과 글로벌 모델 간의 특징 노름 불일치(Feature norm discrepancy)로 이어진다는 점을 문제의 핵심으로 정의한다. 따라서 본 연구의 목표는 특징 정규화(Feature Normalization)를 통해 이러한 노름 불일치를 제거하고, 데이터 이질성이 높은 환경에서도 강건한 특징 표현을 유지하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 로컬 학습 과정에서 특징 벡터를 정규화하여 특징 노름의 편향을 강제로 제거하는 것이다. 

중심적인 직관은 로컬 모델이 자신이 가진 일부 클래스에 대해서만 과도하게 높은 특징 노름을 갖게 되는 현상을 막음으로써, 글로벌 모델로 집계되었을 때 발생할 수 있는 표현의 왜곡을 방지하는 것이다. 이를 위해 특징 벡터의 노름을 항상 1로 고정하는 Feature Normalization(FN) 기법을 FedAVG 프레임워크에 통합한 FedFN을 제안한다.

## 📎 Related Works

기존의 FL 연구들은 크게 두 가지 방향으로 진행되었다.

1.  **분류기 중심 접근법**: Luo et al. [23] 등은 분류기 가중치가 데이터 이질성에 가장 민감하다고 주장하며, Restricted softmax loss나 고정된 직교 분류기(Fixed orthogonal classifiers)를 사용하여 분류기 편향을 완화하려 하였다.
2.  **특징 표현 중심 접근법**: Shi et al. [30] 등은 데이터 이질성이 특징 표현의 차원 붕괴(Dimensional collapse)를 야기할 수 있음을 지적하고, 특징 정렬(Feature alignment)을 통해 이를 해결하려 하였다.

본 논문은 위 연구들과 달리, 구체적으로 '특징 노름의 불일치'가 글로벌 모델의 성능을 저하시키는 주범임을 실험적으로 분석하고, 이를 정규화를 통해 직접적으로 해결한다는 점에서 차별점을 갖는다. 또한, 특징 벡터와 분류기 가중치를 모두 정규화하여 코사인 유사도를 이용하는 SphereFed [4]와 같은 기존 정규화 기반 접근법과 비교하여, 로그잇(Logit)의 제약 조건 완화가 중요함을 시사한다.

## 🛠️ Methodology

### 1. 4-요소 분석 (4-Factor Analysis)
저자들은 FedAVG의 성능 저하 원인을 분석하기 위해 다음 네 가지 지표를 도입한다.
- **Weight similarity**: 클래스 간 분류기 가중치의 코사인 유사도 (낮을수록 좋음)
- **Inter-class similarity**: 클래스별 특징 프로토타입 간의 코사인 유사도 (낮을수록 좋음)
- **Intra-class similarity**: 동일 클래스 내 특징 벡터와 프로토타입 간의 유사도 (높을수록 좋음)
- **Prototype-weight alignment**: 특징 프로토타입과 분류기 가중치 간의 정렬 정도 (높을수록 좋음)

분석 결과, 데이터 이질성이 증가할 때 가장 부정적인 영향을 받는 요소는 **Inter-class similarity**와 **Prototype-weight alignment**였으며, 이는 문제의 핵심이 특징 표현에 있음을 뒷받침한다.

### 2. FedFN 알고리즘
FedFN은 FedAVG의 로컬 학습 단계에서 특징 정규화(FN) 업데이트를 적용한다. 기본적으로 특징 벡터 $f(x; \theta^{ext})$를 그 노름으로 나누어 단위 벡터로 만든 뒤 분류기에 입력한다.

수정된 로그잇(Logit) 벡터 $\hat{z}$는 다음과 같이 계산된다.
$$\hat{z}(x; \theta) = \theta^{cls} \frac{f(x; \theta^{ext})}{\|f(x; \theta^{ext})\|_2}$$

이에 따라 분류기 가중치 $\theta^{cls}$에 대한 교차 엔트로피 손실 $L^{CE}$의 그래디언트는 다음과 같이 변형된다.
$$\nabla_{\theta^{cls}} L^{CE}(x; \theta) = \frac{\nabla_{\hat{z}(x; \theta)} L^{CE}(x; \theta) f(x; \theta^{ext})^\top}{\|f(x; \theta^{ext})\|_2}$$

이 과정에서 그래디언트가 특징 벡터의 노름으로 스케일링되기 때문에, FedFN은 FedAVG보다 더 큰 초기 학습률(Initial learning rate)을 사용하도록 튜닝되었다.

### 3. 특징 노름 정규화 (FedFR)와의 비교
단순히 손실 함수에 $L_2$ 정규화 항을 추가하는 방식($L_\mu = L^{CE} + \mu \|f(x; \theta^{ext})\|^2$)인 FedFR과 비교했을 때, 특징 벡터를 직접 정규화하는 FedFN이 훨씬 더 우수한 성능을 보임을 확인하였다.

## 📊 Results

### 실험 설정
- **데이터셋 및 모델**: CIFAR-10 (VGG11, ResNet18), CIFAR-100 (MobileNet)
- **데이터 이질성 구현**: Sharding 전략(클라이언트당 클래스 수 $s$ 조절) 및 LDA 전략($\alpha$ 조절)
- **평가 지표**: Test Accuracy

### 주요 결과
1.  **성능 향상**: 데이터 이질성이 극심한 설정($s=2$)에서 FedAVG의 정확도가 크게 떨어지는 반면, FedFN은 이를 유의미하게 끌어올렸다. (CIFAR-10, VGG11 기준 $s=2$에서 FedAVG 74.24% $\rightarrow$ FedFN 77.77%)
2.  **기존 알고리즘과의 호환성**: Scaffold, FedEXP 등 기존 FL 알고리즘에 FN 업데이트 모듈을 결합했을 때, 모든 설정에서 Baseline보다 높은 성능을 기록하였다.
3.  **사전 학습 모델(Pretrained Model) 적용**: ResNet18 사전 학습 모델을 사용할 때, FedAVG와 FedBABU는 오히려 데이터 이질성이 높을 때 성능이 하락하는 경향을 보였으나, FedFN은 사전 학습의 이점을 그대로 가져가며 성능이 지속적으로 향상되었다.
4.  **분석 결과**: FedFN은 특히 **Inter-class similarity**를 크게 개선하여 클래스 간 변별력을 높인 것이 성능 향상의 주된 요인임을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 FL의 데이터 이질성 문제를 '특징 노름의 불일치'라는 구체적인 관점에서 분석하고, 이를 정규화라는 단순하고 효과적인 방법으로 해결하였다. 특히 사전 학습된 파운데이션 모델(Foundation Models)에 적용 가능함을 보임으로써 실제 산업적 활용 가치를 높였다.

### 비판적 논의 및 한계
1.  **학습률 민감도**: FedFN은 특징 노름으로 그래디언트를 스케일링하므로 학습률에 매우 민감하다. 논문에서 그리드 서치(Grid search)를 통해 최적값을 찾았으나, 실제 다양한 환경에서 자동으로 최적의 학습률을 찾는 메커니즘에 대한 논의는 부족하다.
2.  **로그잇 제약 조건**: 저자들은 SphereFed가 로그잇 범위를 $[-1, 1]$로 제한하여 성능이 저하된다고 분석하며, 제약을 완화($\tau$ 도입)해야 한다고 주장한다. 이는 정규화를 사용하더라도 모델의 표현력(Expressive power)을 지나치게 제한해서는 안 된다는 중요한 통찰을 제공한다.
3.  **재현성 문제**: SphereFed의 기존 결과가 재현되지 않음을 명시적으로 언급하며, 제안 방법론의 우위성을 강조하였다.

## 📌 TL;DR

본 논문은 연합 학습에서 데이터 이질성이 심화될수록 로컬-글로벌 모델 간의 **특징 노름 불일치(Feature Norm Discrepancy)**가 발생하여 성능이 저하된다는 점을 발견하였다. 이를 해결하기 위해 특징 벡터를 정규화하여 노름 편향을 제거하는 **FedFN**을 제안하였으며, 이는 CIFAR-10/100 데이터셋과 사전 학습된 ResNet18 모델 실험을 통해 데이터 이질성 환경에서도 강건한 성능 향상을 입증하였다. 이 연구는 향후 FL 기반의 파운데이션 모델 튜닝 시 특징 표현의 안정성을 확보하는 데 중요한 기초가 될 것으로 보인다.