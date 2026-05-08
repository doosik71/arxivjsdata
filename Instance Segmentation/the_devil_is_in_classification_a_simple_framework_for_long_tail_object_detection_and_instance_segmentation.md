# The Devil is in Classification: A Simple Framework for Long-tail Instance Segmentation

Tao Wang et al. (2020)

## 🧩 Problem to Solve

현대적인 객체 탐지(Object Detection) 및 인스턴스 분할(Instance Segmentation) 모델들은 주로 COCO와 같이 클래스별 샘플 수가 비교적 균형 잡힌 데이터셋에서 우수한 성능을 보인다. 그러나 실제 환경의 데이터셋은 소수의 클래스가 많은 샘플을 가지고 대다수의 클래스는 매우 적은 샘플을 가지는 Long-tail 분포를 띠는 경우가 많다.

이 논문은 Long-tail 분포의 데이터셋에서 최신 2단계(Two-stage) 인스턴스 분할 모델인 Mask R-CNN의 성능이 급격히 저하되는 원인을 분석하고, 이를 해결하는 것을 목표로 한다. 특히 저자들은 LVIS 데이터셋을 통해 성능 저하의 핵심 원인을 체계적으로 규명하고, 이를 효과적으로 완화할 수 있는 간단한 프레임워크를 제안하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 직관은 인스턴스 분할 모델의 성능 저하가 RPN(Region Proposal Network)이나 박스/마스크 예측 헤드가 아닌, **분류 헤드(Classification Head)의 편향(Bias)**에서 비롯된다는 점이다.

이를 해결하기 위해 제안된 핵심 아이디어는 다음과 같다.

1. **Decoupled Learning**: 특징 추출과 분류 학습을 분리하여, 일반적인 학습이 끝난 후 분류 헤드만을 다시 학습시키는 캘리브레이션(Calibration) 과정을 도입한다.
2. **Bi-level Class Balanced Sampling**: 이미지 수준과 인스턴스 수준의 샘플링을 결합하여 분류 헤드에 균형 잡힌 샘플을 제공함으로써 편향을 제거한다.
3. **Dual Head Inference**: 캘리브레이션된 헤드(Tail 클래스에 강점)와 원래의 헤드(Head 클래스에 강점)를 동시에 사용하여, 클래스 빈도에 따라 선택적으로 예측값을 취하는 추론 방식을 제안한다.

## 📎 Related Works

### 기존 연구 및 한계

- **Object Detection & Segmentation**: Faster R-CNN과 Mask R-CNN과 같은 2단계 프레임워크가 주류를 이루고 있으며, 최근에는 HTC(Hybrid Task Cascade)와 같은 다단계(Multi-stage) 모델이 SOTA 성능을 달성했다. 하지만 이러한 모델들은 주로 균형 잡힌 데이터셋에서 평가되었다.
- **Long-tailed Recognition**: 이미지 분류 분야에서는 샘플링(Sampling)과 손실 함수 재가중치 부여(Loss Re-weighting) 방식이 주로 연구되었다. 예를 들어, 소수 클래스를 오버샘플링하거나, Inverse class frequency를 이용해 손실 값을 조정하는 방식 등이 있다.

### 기존 접근 방식과의 차별점

기존의 Long-tail 분류 기법들을 인스턴스 분할에 적용했을 때, Tail 클래스의 성능은 일부 향상되지만 Head 클래스의 성능을 크게 희생시키는 Trade-off 문제가 발생한다. 본 논문의 SimCal 프레임워크는 학습 과정을 분리(Decoupled)하고 추론 단계에서 Dual Head 구조를 사용함으로써, Head 클래스의 성능을 유지하면서 Tail 클래스의 성능을 끌어올린다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 분석: 성능 저하의 원인

저자들은 Mask R-CNN의 RPN이 생성한 Proposal에 Ground Truth(GT) 라벨을 강제로 부여했을 때, Tail 클래스의 AP가 비약적으로 상승함을 확인했다. 이는 Box 및 Mask 예측 헤드는 Long-tail 분포에 덜 민감하며, **분류 헤드가 Tail 클래스의 Proposal을 정확하게 분류하지 못하는 것이 성능 저하의 주범**임을 시사한다.

### SimCal 프레임워크

SimCal은 표준 학습 이후 분류 헤드만을 재학습시키는 캘리브레이션 과정을 거친다.

#### 1. Bi-level Sampling 기반의 캘리브레이션 학습

표준 학습이 완료된 모델에서 Backbone, RPN, Box/Mask 헤드는 모두 동결(Freeze)하고 분류 헤드만을 학습시킨다. 이때 다음과 같은 Bi-level 샘플링을 통해 학습 데이터를 구성한다.

- **이미지 수준 샘플링**: 모든 클래스에서 동일한 확률로 $n$개의 클래스를 선택하고, 해당 클래스를 포함하는 이미지를 무작위로 샘플링한다.
- **인스턴스 수준 샘플링**: 선택된 이미지 내에서 샘플링된 클래스에 해당하는 Proposal과 배경(Background) 샘플만을 수집하여 학습에 사용한다.

학습 시 사용되는 손실 함수는 다음과 같다.
$$\mathcal{L} = \frac{1}{\sum_{i=0}^{N} n_i} \sum_{i=0}^{N} \sum_{j=1}^{n_i} \mathcal{L}_{cls}(p_{ij}, p^*_{ij})$$
여기서 $N$은 배치당 샘플링된 클래스 수, $n_i$는 클래스 $i$의 샘플 수, $\mathcal{L}_{cls}$는 Cross Entropy Loss를 의미한다.

#### 2. Dual Head Inference

캘리브레이션된 헤드는 Tail 클래스 성능은 좋지만 Head 클래스 성능이 떨어진다. 이를 보완하기 위해 원래의 헤드($p_{orig}$)와 캘리브레이션된 헤드($p_{cal}$)를 결합한다.

특정 클래스 $z$에 대한 최종 예측값 $p[z]$는 다음과 같이 결정된다.
$$p[z] = \begin{cases} p_{cal}[z] & \text{if } N_z \le T \\ p_{orig}[z] & \text{otherwise} \end{cases}$$
여기서 $N_z$는 클래스 $z$의 학습 인스턴스 수이며, $T$는 Head 클래스와 Tail 클래스를 나누는 임계값(Threshold)이다.

## 📊 Results

### 실험 설정

- **데이터셋**: LVIS(v0.5) 및 COCO에서 샘플링한 COCO-LT 데이터셋을 사용한다.
- **평가 지표**: AP(Average Precision)를 사용하며, 특히 인스턴스 빈도에 따라 4개의 구간(Bin)으로 나누어 $\text{AP}_1, \text{AP}_2, \text{AP}_3, \text{AP}_4$를 측정하여 세부 성능을 분석한다. ($\text{AP}_1$이 가장 적은 샘플을 가진 Tail 클래스 영역이다.)

### 주요 결과

1. **기존 기법과의 비교**: Loss Re-weighting, Focal Loss, Image-level Repeat Sampling 등을 적용했을 때 Tail 클래스 성능은 소폭 상승하나 Head 클래스 성능이 하락한다. 반면, SimCal은 $\text{AP}_1$ 등 Tail 영역에서 월등한 성능 향상을 보였다.
2. **SimCal의 효과**: Mask R-CNN에 적용했을 때, 캘리브레이션만 진행하면 Head 클래스 성능이 하락하지만, Dual Head Inference를 적용하면 Head 클래스 성능을 유지하면서 Tail 클래스 성능을 극대화할 수 있음을 확인했다.
3. **SOTA 모델 적용**: 다단계 모델인 HTC(Hybrid Task Cascade)에 SimCal을 적용한 결과, LVIS 데이터셋에서 기존 최상위 단일 모델보다 AP가 6.3 절대 수치만큼 향상되어 SOTA 성능을 달성했다.
4. **일반화 성능**: COCO-LT 데이터셋에서도 동일한 성능 향상 경향이 관찰되어, 제안 방법이 일반적인 Long-tail 상황에서 유효함을 입증했다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 Long-tail 인스턴스 분할 문제의 핵심이 '분류'에 있음을 실험적으로 증명하고, 이를 해결하기 위해 모델 전체를 재학습시키는 대신 분류 헤드만을 분리하여 학습시키는 매우 효율적인 방식을 제안했다. 특히 추론 단계에서 간단한 조건부 선택(Dual Head)만으로 Head/Tail 클래스 간의 성능 트레이드-오프를 해결한 점이 인상적이다.

### 한계 및 논의사항

- **임계값 $T$의 설정**: Dual Head Inference에서 $T$라는 하이퍼파라미터가 필요하다. 실험적으로 $T \in [90, 500]$ 범위 내에서 성능이 안정적임을 보였으나, 최적의 $T$를 찾는 기준에 대한 이론적 근거는 부족하다.
- **헤드 구조의 영향**: 분석 과정에서 Tail 클래스에는 Class-agnostic 헤드가, Head 클래스에는 Class-wise 헤드가 더 유리하다는 관찰 결과가 있었다. 이는 향후 클래스 빈도에 따라 헤드 구조 자체를 다르게 가져가는 방향으로 연구가 확장될 가능성을 제시한다.

## 📌 TL;DR

본 논문은 Long-tail 분포의 데이터셋에서 인스턴스 분할 모델의 성능 저하 원인이 분류 헤드의 편향(Classification Bias)에 있음을 밝혀냈다. 이를 해결하기 위해 분류 헤드만을 독립적으로 재학습시키는 **SimCal** 프레임워크를 제안하였으며, **Bi-level 샘플링**을 통해 균형 잡힌 학습을 수행하고 **Dual Head Inference**를 통해 Head와 Tail 클래스 모두에서 높은 성능을 유지하도록 설계했다. 이 방법은 Mask R-CNN과 HTC 모델 모두에서 탁월한 성능 향상을 보였으며, 특히 LVIS 챌린지에서 우수한 성적을 거두며 실용성을 입증했다.
