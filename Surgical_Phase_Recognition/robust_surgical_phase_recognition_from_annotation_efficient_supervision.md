# Robust Surgical Phase Recognition From Annotation Efficient Supervision

Or Rubin, and Shlomi Laufer (2024)

## 🧩 Problem to Solve

본 논문은 컴퓨터 보조 수술(computer-assisted surgery)의 핵심 과제인 수술 단계 인식(Surgical Phase Recognition)에서 발생하는 주석 비용 문제와 데이터 누락에 따른 모델의 취약성을 해결하고자 한다.

일반적으로 수술 단계 인식 모델은 모든 프레임에 대해 정답이 지정된 Fully-supervised 학습 방식을 사용한다. 그러나 이는 숙련된 외과 의사가 모든 프레임을 세밀하게 주석 처리해야 하므로 시간과 비용이 매우 많이 소모된다는 단점이 있다. 최근 이를 해결하기 위해 각 단계별로 단 하나의 프레임에만 주석을 다는 Timestamp supervision 방식이 제안되었으나, 수술 과정이 비결정적인 복잡한 수술의 경우 주석자가 특정 단계를 누락(missing phase annotations)할 가능성이 크며, 이는 모델 성능의 심각한 저하로 이어진다.

따라서 본 연구의 목표는 다음과 같다.

1. Timestamp supervision 환경에서 주석 누락에 강건한(robust) 수술 단계 인식 방법을 제안한다.
2. 주석 효율성을 극대화하기 위해, 비디오당 고정된 $K$개의 프레임만 주석 처리하는 SkipTag@K 방식을 수술 도메인에 도입하고 그 효용성을 검증한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **주석 효율적인 지도 학습(Annotation Efficient Supervision)** 환경에서도 모델이 안정적인 성능을 유지할 수 있도록 하는 강건한 학습 프레임워크를 구축하는 것이다.

주요 기여 사항은 다음과 같다.

- **누락된 라벨에 대한 강건성 확보**: Timestamp supervision에서 발생할 수 있는 라벨 누락 문제를 해결하기 위해, 단계의 순서를 유지하면서도 유연하게 누락된 토큰을 처리할 수 있는 학습 구조를 제안한다.
- **SkipTag@K 도입**: 수술 도메인에 SkipTag@K 주석 방식을 도입하여, 매우 적은 수의 샘플(예: 비디오당 2~3개 프레임)만으로도 경쟁력 있는 성능을 낼 수 있음을 증명한다.
- **효율적인 Pseudo-labeling 파이프라인**: Entropy 기반의 불확실성 측정과 Temperature scaling을 활용하여 전이 지점(transition moments)을 정확히 감지하고, 이를 제외한 신뢰도 높은 영역에서 Pseudo-label을 생성하여 모델을 고도화한다.

## 📎 Related Works

### 수술 단계 인식 (Surgical Phase Recognition)

기존 연구들은 주로 ResNet, Inception, ViT와 같은 Backbone 네트워크로 특징을 추출하고, LSTM, MS-TCN, Transformer와 같은 Temporal 모델로 시간적 의존성을 모델링하는 2단계 구조를 채택한다. 대부분은 Fully-supervised 방식에 의존하며, 일부 연구에서 Semi-supervised 학습이나 Active learning이 탐구되었다.

### 액션 세그멘테이션 (Action Segmentation)

수술 단계 인식은 액션 세그멘테이션의 특수한 사례로 볼 수 있다. 기존 연구에서는 Timestamp supervision, Set supervision, 그리고 CTC(Connectionist Temporal Classification) 손실 함수를 이용한 전사 기반 지도 학습 등이 연구되었다. 특히 본 논문은 누락된 액션을 처리하기 위한 EM 기반 접근 방식이나 SkipTag 설정 등의 기존 액션 세그멘테이션 연구를 수술 도메인으로 확장하여 적용한다.

### 클래스 불균형 및 불확실성 추정

수술 단계는 단계별 지속 시간이 매우 달라 클래스 불균형이 심하다. 이를 해결하기 위해 Focal loss나 Weighted focal loss가 사용되어 왔다. 또한, 모델의 예측 신뢰도를 측정하기 위해 Monte Carlo Dropout(MCD)이나 Temperature scaling과 같은 불확실성 추정 기법이 활용되어 왔다.

## 🛠️ Methodology

### 전체 파이프라인 (System Architecture)

시스템은 크게 두 단계의 아키텍처로 구성된다.

1. **특징 추출 (Feature Extraction)**: ImageNet으로 사전 학습된 ResNet-50을 기반으로 하며, 대상 수술 데이터셋에 대해 DINO(Self-supervised learning) 방법으로 파인튜닝하여 프레임별 특징을 추출한다.
2. **시간적 모델 (Temporal Model)**: 추출된 특징을 입력으로 하여 각 프레임의 단계를 예측하는 TCN(Temporal Convolutional Network) 기반 모델을 사용한다.

### 학습 목표 및 손실 함수 (Loss Functions)

모델은 다음과 같은 복합 손실 함수 $\mathcal{L}$을 통해 학습된다.
$$\mathcal{L} = \mathcal{L}_{cls} + \alpha_1 \mathcal{L}_{S} + \alpha_2 \mathcal{L}_{Entropy} + \alpha_3 \mathcal{L}_{conf} + \alpha_4 \mathcal{L}_{stc}$$

각 구성 요소의 역할은 다음과 같다.

- **Balanced Classification Loss ($\mathcal{L}_{cls}$)**: 클래스 불균형을 해결하기 위해 Weighted Focal Loss를 사용한다. 역 클래스 빈도 가중치 $w^c = \frac{N}{N_c}$를 적용하여 빈도가 낮은 단계의 학습 비중을 높인다.
- **Entropy Loss ($\mathcal{L}_{Entropy}$)**: 모델이 예측 결과에 대해 더 확신을 갖도록 유도하며, 예측 분포의 엔트로피 $H(q) = -\sum q_j \log(q_j)$를 최소화한다.
- **Confidence Loss ($\mathcal{L}_{conf}$)**: 정답 주석 주변에서 예측 신뢰도가 단조 증가/감소하도록 하여, 튀는 값(outlier)을 억제하고 시간적 일관성을 강제한다.
- **Smoothness Loss ($\mathcal{L}_{S}$)**: 인접한 프레임 간의 예측 변화를 페널티를 주어 매끄러운 세그멘테이션 결과를 생성한다. 로그 확률의 Mean Squared Error를 기반으로 계산된다.
- **Star Temporal Classification (STC) Loss ($\mathcal{L}_{stc}$)**: CTC loss의 변형으로, 'startoken'을 추가하여 누락된 라벨을 유연하게 처리하면서도 전체적인 단계의 순서를 학습하도록 한다.

### 추가 학습 단계 및 Pseudo-label 생성

기본 모델(Base model) 학습 후, 이를 이용해 Pseudo-label을 생성하여 모델을 한 번 더 정교화한다.

1. **전이 지점 감지**: Temperature scaling이 적용된 엔트로피 측정치 $U(l) = H(\text{softmax}(l/M_T))$를 사용하여 단계가 바뀌는 전이 지점을 찾는다.
2. **Pseudo-label 추출**: 전이 지점 주변 $W$ 프레임 윈도우를 제외한, 신뢰도가 높은 영역의 예측값을 Pseudo-label로 확정한다.
3. **최종 학습**: 생성된 Pseudo-label과 원래의 주석을 합쳐 Unweighted focal loss와 Smoothness loss를 사용하여 최종 모델을 학습시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80 (담낭 절제술, 7개 단계), MultiBypass140 (위 우회술, 12개 단계).
- **지표**: Accuracy(AC), Precision(PR), Recall(RE), Jaccard(JA), F1-score. Cholec80의 경우 10초의 여유를 두는 'relaxed' 지표를 사용한다.
- **비교 대상**: Ding et al. [7]의 Timestamp supervision 방법론.

### 주요 결과

1. **누락 라벨에 대한 강건성**:
   - 주석 누락률($p_m$)이 0.3까지 증가할 때, Ding et al.의 방법은 성능이 급격히 하락(Cholec80 기준 10~28% 하락)하는 반면, 제안 방법은 성능 저하가 매우 적고 안정적인 수치를 유지한다.
   - MultiBypass140 데이터셋에서도 모든 누락률 설정에서 Ding et al.보다 일관되게 높은 성능을 보였다.

2. **SkipTag@K 성능**:
   - 매우 적은 주석만으로도 높은 정확도를 달성했다.
   - **Cholec80**: 비디오당 단 2개의 샘플만 사용했을 때 Accuracy 83.6%를 달성했다.
   - **MultiBypass140**: 비디오당 단 3개의 샘플만 사용했을 때 Accuracy 85.1%를 달성했다.
   - 샘플 수를 줄여도(K값 감소) 성능 하락 폭이 매우 완만하여, 주석 비용을 획기적으로 줄일 수 있음을 입증했다.

3. **Ablation Study**:
   - DINO 기반의 특징 추출기가 기본 ResNet-50보다 성능과 강건성을 크게 향상시켰다.
   - $\mathcal{L}_{conf}$, Focal loss, STC loss, 그리고 추가 학습 단계가 순차적으로 성능 향상에 기여함을 확인했다.

## 🧠 Insights & Discussion

본 논문은 수술 단계 인식에서 주석 비용을 낮추면서도 실제 환경에서 발생할 수 있는 라벨 누락 문제에 대응할 수 있는 실용적인 프레임워크를 제시하였다. 특히 STC loss와 엔트로피 기반의 Pseudo-label 생성 전략이 모델의 강건성을 높이는 데 결정적인 역할을 하였다.

**한계점 및 향후 연구 방향:**

- **잘못된 라벨 처리**: 본 연구는 라벨의 '누락' 문제는 해결했으나, 잘못 지정된 '오답 라벨(incorrect label)' 문제에 대해서는 다루지 않았다.
- **샘플링 전략**: 현재 SkipTag@K에서 비디오를 균등 분할하여 샘플링하는 단순한 방식을 사용하고 있으나, 향후 클러스터링 기반의 정교한 샘플링 기법을 도입하면 더 나은 성능을 기대할 수 있다.
- **약한 신호(Weak signals) 활용**: 수술 도구의 사용, 장면의 변화, 수술실 내 오디오 신호 등 수술 과정에서 발생하는 부가적인 약한 신호를 결합한다면 주석 부담을 더욱 줄일 수 있을 것이다.

## 📌 TL;DR

본 연구는 수술 단계 인식에서 **주석 비용을 획기적으로 줄이면서도 라벨 누락에 강건한 학습 방법**을 제안한다. STC loss와 엔트로피 기반 Pseudo-labeling을 통해 누락된 주석의 영향을 최소화하였으며, **SkipTag@K** 방식을 통해 비디오당 단 몇 개의 프레임 주석만으로도 높은 정확도(Cholec80 AC 83.6% @K=2, MultiBypass140 AC 85.1% @K=3)를 달성하였다. 이 결과는 실제 의료 현장에서 주석 비용 문제를 해결하고 효율적인 수술 워크플로우 인식 시스템을 구축하는 데 중요한 기여를 할 것으로 보인다.
