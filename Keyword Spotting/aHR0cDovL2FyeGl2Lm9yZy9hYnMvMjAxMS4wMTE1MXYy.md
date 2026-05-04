# OPTIMIZE WHAT MATTERS: TRAINING DNN-HMM KEYWORD SPOTTING MODEL USING END METRIC

Ashish Shrivastava, Arnav Kundu, Chandra Dhir, Devang Naik, Oncel Tuzel (2021)

## 🧩 Problem to Solve

본 논문은 항상 켜져 있는(always-on) 기기에서 특정 호출어(wake word)를 감지하는 Keyword Spotting (KWS) 시스템의 성능 최적화 문제를 다룬다. 기존의 DNN-HMM 기반 KWS 모델은 두 단계로 나누어 학습된다. 먼저 DNN은 각 음성 프레임의 상태 확률을 예측하도록 Cross-Entropy loss를 통해 독립적으로 학습되고, 이후 HMM 디코더는 Dynamic Programming (DP)을 통해 이 확률들을 결합하여 최종 감지 점수(detection score)를 계산한다.

여기서 발생하는 핵심 문제는 학습 시의 손실 함수(Cross-Entropy)와 실제 평가 지표인 감지 점수 사이의 불일치, 즉 **loss-metric mismatch**이다. DNN은 모든 상태(state)를 동일하게 정확하게 예측하도록 학습되지만, 실제 감지 성능에는 특정 상태가 더 중요할 수 있으며 HMM의 전이 확률(transition probability) 등이 고려되지 않는다. 특히 자원이 제한된 소형 모델(low capacity models)에서는 이러한 불일치가 성능 저하의 주된 원인이 된다. 따라서 본 논문의 목표는 모델 아키텍처나 데이터 양을 늘리지 않고, 최종 감지 점수를 직접 최적화하는 학습 전략을 통해 KWS 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **HMM 디코더를 미분 가능(differentiable)하게 만들어, 최종 감지 점수를 기반으로 DNN의 파라미터를 직접 학습시키는 end-to-end 최적화**를 구현하는 것이다.

주요 기여 사항은 다음과 같다:
1. **End-metric 기반 최적화**: 상태 확률의 정확도가 아닌, 최종 감지 점수를 직접 최적화하는 학습 방식을 제안하였다.
2. **IOU 기반 샘플링**: 객체 탐지(Object Detection) 분야의 개념을 도입하여, 정답 윈도우와 Intersection-over-Union (IOU)이 높은 샘플은 Positive로, 낮은 샘플은 Negative로 정의하여 학습에 활용하였다.
3. **미분 가능한 HMM 디코더**: DP 알고리즘을 통해 계산되는 감지 점수를 DNN까지 역전파(back-propagate)하여 파라미터를 업데이트하는 메커니즘을 설계하였다.
4. **성능 향상**: 모델 구조의 변경 없이 독립 학습 방식 대비 False Rejection Rate (FRR)를 70% 이상 감소시켰다.

## 📎 Related Works

기존의 KWS 연구들은 주로 다음과 같은 접근 방식을 취해왔다:
- **HMM 및 DNN-HMM**: 전통적인 방식으로, 음향 모델이 관측 확률을 계산하고 HMM 디코더가 점수를 산출한다. 하지만 DNN을 독립적으로 학습시키기 때문에 앞서 언급한 loss-metric mismatch 문제가 발생한다.
- **Discriminative Training**: 생성 모델(Generative model)인 HMM을 위해 판별적 학습 방법들이 제안되었으나, 본 논문은 IOU 기준의 윈도우 샘플링과 HMM 디코딩 과정의 직접적인 미분을 통해 이를 차별화한다.
- **RNN 및 CNN 기반 모델**: End-to-end 학습이 가능하며 성능이 뛰어나지만, RNN은 저전력 스트리밍 환경에 부적합하고, CNN은 대규모 컨텍스트를 인코딩하기 위해 더 많은 연산량을 요구한다. 또한 일부 CNN 기반 모델조차 프레임 레벨의 Cross-Entropy를 최소화하므로 여전히 loss-metric mismatch 문제에서 자유롭지 못하다.

## 🛠️ Methodology

### 1. 시스템 구조 및 파이프라인
전체 시스템은 입력 음성 신호에서 MFCC 특징을 추출하고, 이를 DNN에 입력하여 상태 확률을 얻은 뒤, HMM 디코더를 통해 최종 감지 점수를 도출하는 구조이다.

- **DNN**: 몇 개의 Fully-connected layer와 Softmax layer로 구성된다. 입력으로 $\delta=9$인 컨텍스트 윈도우를 포함한 MFCC 특징을 받으며, 출력으로 $C=20$개 상태(포네임 18개, 정적/배경 소음 각 1개)에 대한 확률 분포를 내놓는다.
- **HMM 디코더**: DNN의 출력을 받아 Dynamic Programming을 통해 가장 가능성 높은 경로(most likely path)를 찾고 감지 점수를 계산한다.

### 2. IOU 기반 샘플링
학습 데이터에서 긍정(Positive) 및 부정(Negative) 샘플을 정의하기 위해 IOU를 사용한다. 정답 윈도우 $[g_1, g_2]$와 샘플링된 윈도우 $[w_1, w_2]$ 사이의 IOU는 다음과 같이 정의된다:

$$IOU = \frac{\max(0, \min(g_2, w_2) - \max(g_1, w_1))}{\max(g_2, w_2) - \min(g_1, w_1)}$$

- **Positive samples**: $IOU \ge 0.95$인 윈도우.
- **Negative samples**: $IOU \le 0.5$인 윈도우.
- **Hard negatives**: 정답 윈도우를 두 부분으로 나누어 순서를 바꾼 샘플을 추가하여, 모델이 단순히 상태의 존재 여부가 아니라 정확한 순서(ordering)를 학습하도록 강제한다.

### 3. 미분 가능한 HMM 디코딩 및 손실 함수
HMM 디코더의 각 타임 스텝 $t$에서 상태 $i$에 도달할 최대 확률 $v_i(t)$는 다음과 같이 계산된다:

$$v_i(t) = \max \{v_i(t-1) \cdot b_{i,i}, v_{i-1}(t-1) \cdot b_{i-1,i}\} \cdot f_\theta(x_t)[i]$$

여기서 $b_{i,j}$는 HMM 전이 확률이며, $f_\theta(x_t)[i]$는 DNN이 예측한 상태 $i$의 확률이다. 최종 감지 점수 $d$는 마지막 상태 $C$에 도달한 확률을 윈도우 길이 $T$로 나눈 값이다: $d = v_C(T)/T$.

이 점수 $d$를 사용하여 **Hinge Loss**를 정의하고 DNN의 파라미터 $\theta$를 최적화한다:

$$L_{e2e} = \min_\theta \sum_{j \in X_p} \max(0, 1 - d_j) + \sum_{j \in X_n} \max(0, 1 + d_j)$$

여기서 $X_p$는 Positive 윈도우 집합, $X_n$은 Negative 윈도우 집합이다. 이 손실 함수는 구간별로 매끄러운(piece-wise smooth) 형태이므로 경사 하강법을 통해 DNN 파라미터까지 역전파가 가능하다.

## 📊 Results

### 1. 실험 설정
- **데이터**: 약 500k 개의 호출어 포함 발화 데이터(학습), 약 2000시간의 Dense speech 데이터(테스트).
- **비교 대상**: 독립적으로 학습된 DNN-HMM, Stacked 1D CNN (S1DCNN) 기반 모델.
- **지표**: DET 곡선 (FRR vs FA/hr), IOU 기반의 Localization 정확도.

### 2. 정량적 결과
- **FRR 감소**: FA(False Accept)가 15/hr인 지점에서, 제안 방법은 독립 학습 방식 대비 FRR을 획기적으로 낮추었다. (표 1 기준, 독립 학습 93.95% $\to$ 제안 방법 1.13%).
- **Localization 성능**: S1DCNN 모델과 비교했을 때, 제안 방법의 평균 IOU가 27.7% 더 높았으며, 위치 오차(localization error) 또한 $0.03$초로 S1DCNN($0.13$초)보다 훨씬 정교했다.

### 3. 정성적 분석 (Confusion Matrix)
그림 2의 혼동 행렬 분석 결과, 독립 학습 DNN은 상태 분류 정확도(68.1%)는 높았으나 이것이 최종 감지 성능으로 이어지지 않았다. 반면, End-to-end 학습 DNN은 상태 분류 정확도(54.7%)는 낮아졌지만(특히 인접 상태 간의 혼동 증가), 최종 감지 점수를 높이는 데 결정적인 핵심 상태들에 집중함으로써 더 높은 감지 성능을 보였다.

## 🧠 Insights & Discussion

본 논문의 가장 중요한 통찰은 **"모든 상태를 정확하게 예측하는 것이 반드시 최선의 감지 성능을 보장하는 것은 아니다"**라는 점이다. 상태 분류 정확도라는 대리 지표(proxy metric)를 최적화하는 대신, 실제 시스템의 최종 출력값인 감지 점수를 최적화했을 때 더 우수한 결과가 나옴을 증명하였다.

**강점**:
- 모델 아키텍처나 추론 파이프라인의 변경 없이 학습 알고리즘만으로 성능을 극대화하였다.
- IOU 기반 샘플링과 Hard negative 생성을 통해 실제 환경에서 발생할 수 있는 오작동(부분 일치 등)을 효과적으로 억제하였다.
- CNN 기반 end-to-end 모델보다 연산 효율성이 높고 해석 가능성이 좋으며 Localization 성능이 뛰어나다.

**한계 및 논의**:
- 본 연구는 DNN-HMM 구조에 집중하고 있으며, 최신 Transformer 기반의 KWS 모델들과의 직접적인 비교는 제시되지 않았다.
- HMM의 전이 확률 $b_{i,j}$를 고정된 통계치로 사용하였는데, 이 파라미터까지 함께 학습시켰을 때의 이득에 대해서는 명시되지 않았다.

## 📌 TL;DR

본 논문은 DNN-HMM 기반 호출어 감지 모델에서 학습 손실(Cross-Entropy)과 평가 지표(Detection Score) 사이의 불일치 문제를 해결하기 위해, **HMM 디코더를 미분 가능하게 설계하여 최종 점수를 직접 최적화하는 End-to-End 학습 방법**을 제안한다. 특히 IOU 기반의 윈도우 샘플링 전략을 도입하여, 모델 구조 변경 없이도 FRR을 70% 이상 감소시키고 정교한 위치 탐지 능력을 확보하였다. 이는 저전력 임베디드 환경의 KWS 시스템 성능을 높이는 매우 실용적인 접근 방식으로, 향후 실시간 음성 인터페이스의 신뢰성을 높이는 데 기여할 가능성이 크다.