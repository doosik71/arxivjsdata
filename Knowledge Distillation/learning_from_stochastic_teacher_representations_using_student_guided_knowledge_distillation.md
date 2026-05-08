# Learning from Stochastic Teacher Representations Using Student-Guided Knowledge Distillation

Muhammad Haseeb Aslam et al. (2025)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델의 성능을 높이기 위해 널리 사용되는 앙상블 학습(Ensemble Learning)의 높은 계산 비용과 저장 공간 문제를 해결하고자 한다. 일반적으로 여러 개의 모델을 학습시켜 결합하는 앙상블 방식이나, 가중치 평균을 내는 Model Soups 방식은 높은 정확도를 보장하지만, 리소스가 제한된 웨어러블 기기나 지연 시간에 민감한 실시간 애플리케이션에 배포하기에는 매우 비효율적이다.

또한, 지식 증류(Knowledge Distillation, KD)를 통해 앙상블 모델의 지식을 단일 학생(Student) 모델로 이전하려는 시도가 있었으나, 이는 여전히 여러 개의 교사(Teacher) 모델을 학습시켜야 한다는 부담이 있다. 단일 모델 내에서 Dropout과 같은 확률적(Stochastic) 메커니즘을 사용하여 다양성을 확보하려는 시도는 가능하지만, 이렇게 생성된 확률적 표현(Representation)들 중에는 학습 작업과 맞지 않는 노이즈(Noisy representations)가 포함되어 있어 학생 모델의 학습을 방해한다는 문제가 존재한다. 따라서 본 논문의 목표는 **단일 모델만으로 효율적으로 다양한 교사 표현을 생성하고, 그중 작업에 유의미한 지식만을 선택적으로 증류하는 방법론을 제안하는 것**이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Distillation-time Dropout**을 통해 단일 교사 모델에서 다양한 확률적 표현을 생성하고, **Student-Guided Knowledge Distillation (SGKD)** 메커니즘을 통해 학생 모델이 스스로 유용한 지식을 필터링하게 하는 것이다.

주요 기여 사항은 다음과 같다:

1. **Stochastic Self-Distillation (SSD)**: 사전 학습된 단일 교사 모델에 증류 단계에서 Dropout을 적용하여, 추가적인 모델 학습 없이도 앙상블과 유사한 다양한 특징 표현을 생성한다.
2. **Student-Guided KD (SGKD)**: 학생 모델의 현재 상태를 기준으로 교사 모델의 표현들을 랭킹화하고 가중치를 부여함으로써, 노이즈를 제거하고 작업에 가장 관련성이 높은 표현만을 선택적으로 증류한다.
3. **효율성 증명**: 모델 크기를 증가시키지 않으면서도 SOTA(State-of-the-art) 수준의 성능을 달성하며, 전통적인 앙상블이나 가중치 평균 방식보다 학습 및 추론 비용을 획기적으로 낮추었다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 다룬다:

- **Knowledge Distillation (KD)**: 교사의 로짓(Logit)이나 특징 표현을 학생에게 전달하는 기본 개념이며, 특히 동일한 구조의 모델 간에 지식을 주고받는 Self-distillation 연구들이 언급된다.
- **Deep Ensembles & Model Soups**: 독립적으로 학습된 여러 모델을 결합하여 일반화 성능을 높이는 방식이다. Model Soups는 가중치 평균을 통해 추론 시 모델 크기를 유지하지만, 학습 단계에서 많은 비용이 발생한다는 한계가 있다.
- **Self-Distillation**: 동일 구조의 모델에서 학생이 교사를 능가하는 현상이 보고되었으며, 특히 Early stopping의 중요성이 강조되었다. Allen-Zhu와 Li (2020)의 연구가 SSD와 가장 유사하나, 이들은 데이터 증강이나 다양한 랜덤 시드 기반의 교사 앙상블을 필요로 한다.

SSD는 이러한 기존 방식들과 달리, 데이터 증강이나 여러 번의 모델 학습 없이 **특징 공간(Feature space)에서의 Dropout**과 **학생 가이드 기반의 주의 집중(Attention)** 메커니즘만으로 다양성을 확보한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

SSD의 전체 프로세스는 크게 교사 모델 학습, 학생 모델 초기화, 그리고 학생 가이드 기반의 확률적 증류 단계로 구성된다.

1. **Teacher Training**: 먼저 교사 모델 $T$를 학습시켜 최적의 파라미터 $\theta^T$를 얻는다.
2. **Student Initialization**: 학생 모델 $S$의 파라미터 $\theta^S$를 학습된 교사 모델의 가중치 $\theta^T$로 초기화한다. 이는 학생 모델이 교사 표현의 유효성을 판단할 수 있는 '권위(Authority)'를 갖게 하기 위함이다.
3. **Stochastic Representation Generation**: 입력 데이터 $x$에 대해 교사 모델에 Dropout을 적용한 채로 $n$번의 Forward pass를 수행하여, $n$개의 서로 다른 확률적 특징 벡터 $F^T(x) = [f^T_1(x), \dots, f^T_n(x)]$를 생성한다.

### Student-Guided Knowledge Distillation (SGKD)

생성된 $n$개의 교사 표현 중 노이즈를 제거하고 유의미한 표현을 추출하기 위해 다음 과정을 거친다.

**1) 유사도 계산 및 어텐션 가중치 산출**
학생 모델의 현재 표현 $f^S(x)$와 각 교사 표현 $f^T_i(x)$ 간의 내적(Dot product)을 통해 유사도 $\phi_i$를 계산하고, Softmax를 통해 가중치 $\alpha_i$를 구한다.
$$\alpha_i = \frac{\exp(\phi_i / h)}{\sum_{j=1}^{N} \exp(\phi_j / h)}$$
여기서 $h$는 가중치가 특정 표현에 과도하게 쏠리는 것을 방지하는 정규화 계수(Temperature scaling)이다.

**2) $\epsilon$-퍼센타일 기반 마스킹**
단순한 top-k 방식 대신, $\epsilon$-th 퍼센타일 임계값을 사용하여 하위 표현들을 제거한다.
$$\hat{\alpha}_i = \begin{cases} \alpha_i, & \text{if } \alpha_i \geq \epsilon \\ 0, & \text{otherwise} \end{cases}$$
이를 통해 각 샘플의 분포에 따라 동적으로 유의미한 표현의 개수를 조절한다.

**3) 가중 합산 표현 생성 및 증류**
최종적으로 가중치가 적용된 교사 표현 $\hat{f}^T(x)$를 생성한다.
$$\hat{f}^T(x) = \sum_{i=1}^{N} \hat{\alpha}_i \cdot f^T_i(x)$$

### 학습 목표 및 손실 함수

학생 모델은 다음의 총 손실 함수 $L_{total}$을 최소화하도록 학습된다.
$$L_{total} = L_{task} + \lambda L_{dist}$$
여기서 $L_{task}$는 원래의 분류/회귀 작업 손실이며, $L_{dist}$는 학생 표현과 필터링된 교사 표현 간의 평균 제곱 오차(MSE)이다.
$$L_{dist} = \frac{1}{d} \sum_{j=1}^{d} (f^S(x_j) - \hat{f}^T(x_j))^2$$

## 📊 Results

### 실험 설정

- **데이터셋**: Biovid Heat Pain(통증 인식), StressID(스트레스 인식), UCR Archive(생체 신호), HAR(행동 인식), CIFAR-10/100(이미지 분류) 등 매우 다양한 도메인에서 검증하였다.
- **비교 대상**: SOTA 방법론, 기본 모델(Baseline), 전통적 앙상블(Majority Vote, Average), Model Soup (Uniform, Greedy), SWA(Stochastic Weight Averaging).
- **평가 지표**: Accuracy, F1-Score.

### 주요 결과

1. **정확도 향상**:
   - Biovid 데이터셋에서 Baseline 대비 정확도가 2.5% 향상되어 86.90%를 달성하였다.
   - StressID 데이터셋의 이진 분류 작업에서 F1-score와 Accuracy 모두 SOTA 대비 약 3~4% 향상되었다.
   - UCR Archive의 12개 데이터셋 평균 정확도는 0.8508로, TS2Vec/SoftCLT 기반 방법론보다 우수한 성능을 보였다.
2. **효율성 및 모델 크기**:
   - HAR 데이터셋 실험 결과, SSD는 25개 모델을 사용한 앙상블(Accuracy 91.83%)과 거의 동일한 성능(91.82%)을 내면서도, 추론 시 모델 크기는 단일 모델 수준($0.9\text{M}$)으로 유지하였다.
   - Model Soup 방식은 추론 시 크기는 작지만 학습 시 FLOPs가 비약적으로 증가($0.87 \rightarrow 21.8\text{ G-FLOPs}$)하는 반면, SSD는 교사와 학생을 각각 한 번씩만 학습하면 되므로 매우 효율적이다.
3. **이미지 데이터 검증**: CIFAR-10(94.83%)과 CIFAR-100(81.73%)에서도 Baseline 대비 각각 1.5%, 2.2%의 성능 향상을 보여, 시계열 데이터뿐 아니라 일반적인 비전 작업에서도 유효함을 증명하였다.

## 🧠 Insights & Discussion

### 분석 및 강점

- **모델 구조와의 상관관계**: 본 논문은 모델 내에 Dropout 레이어가 많을수록 SSD의 효과가 크다는 점을 발견하였다. 이는 확률적 표현의 다양성이 더 풍부해져, 더 정보가 많은 $\hat{f}^T(x)$를 구성할 수 있기 때문이다.
- **동적 필터링의 유효성**: $\epsilon$-퍼센타일 임계값 방식이 고정된 top-k 방식보다 뛰어난 이유는 배치(Batch)마다 다른 특징 분포를 동적으로 반영할 수 있기 때문이다. 실험 결과 $\epsilon=90$에서 최적의 성능을 보였다.
- **초기화의 중요성**: 학생 모델을 랜덤 초기화했을 때 성능이 baseline보다 떨어지는 현상이 관찰되었다. 이는 학생 모델이 교사의 가중치로 초기화되어야만 교사 표현의 유효성을 판단할 수 있는 기준점(Anchor) 역할을 할 수 있기 때문이다.

### 한계 및 비판적 해석

- **Dropout 의존성**: 본 방법론은 아키텍처 내에 이미 Dropout이 포함되어 있어야 한다는 전제가 있다. Dropout이 없는 모델에 적용하려면 특징 공간에 직접 노이즈를 주입하는 등의 추가적인 변형이 필요할 것이다.
- **하이퍼파라미터 민감도**: $\epsilon, h, \lambda, n$(반복 횟수) 등 설정해야 할 하이퍼파라미터가 다수 존재하며, 이에 대한 최적화 과정이 필요하다.

## 📌 TL;DR

본 논문은 단일 모델에서 **Distillation-time Dropout**을 통해 다양한 교사 표현을 생성하고, 이를 **학생 모델의 현재 상태를 가이드로 삼아 필터링**하여 증류하는 **Stochastic Self-Distillation (SSD)** 방법을 제안한다. 이를 통해 모델 크기 증가 없이 앙상블 학습에 준하는 성능 향상을 달성하였으며, 특히 리소스가 제한된 웨어러블 기기 환경에서 매우 효율적인 대안이 될 수 있음을 입증하였다. 향후 연구에서는 Dropout 외의 다른 섭동(Perturbation) 방법을 통한 다양성 확보 가능성을 제시하고 있다.
