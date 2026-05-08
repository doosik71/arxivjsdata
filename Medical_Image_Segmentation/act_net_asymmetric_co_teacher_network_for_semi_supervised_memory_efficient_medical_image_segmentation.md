# ACT-NET: ASYMMETRIC CO-TEACHER NETWORK FOR SEMI-SUPERVISED MEMORY-EFFICIENT MEDICAL IMAGE SEGMENTATION

Ziyuan Zhao, Andong Zhu, Zeng Zeng, Bharadwaj Veeravalli, Cuntai Guan (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 발생하는 두 가지 핵심적인 현실적 제약을 해결하고자 한다.

첫째는 **데이터 라벨링의 비용 문제**이다. 딥러닝 모델이 높은 성능을 내기 위해서는 대량의 정교한 라벨링 데이터가 필요하지만, 의료 현장에서 전문의가 직접 라벨링을 수행하는 것은 매우 시간 소모적이고 노동 집약적이어서 실제 임상 적용에 어려움이 있다.

둘째는 **모델의 계산 복잡도 및 메모리 문제**이다. 높은 정확도를 가진 모델들은 일반적으로 파라미터 수가 많고 크기가 커서, 실시간 모바일 헬스(mHealth) 애플리케이션과 같이 자원이 제한된 환경이나 지연 시간(latency) 요구 사항이 엄격한 실제 시나리오에 배포하기 어렵다.

따라서 본 논문의 목표는 **적은 양의 라벨링 데이터만으로도 학습이 가능하며, 동시에 메모리 효율적인(경량화된) 고성능 분할 모델을 구축하는 것**이다.

## ✨ Key Contributions

본 연구의 중심 아이디어는 **비대칭 코-티처(Asymmetric Co-teacher)** 프레임워크를 통해 반지도 학습(Semi-supervised Learning)과 지식 증류(Knowledge Distillation, KD)를 결합하는 것이다.

핵심 설계 아이디어는 서로 다른 구조를 가진 거대 모델(Teacher)과 소형 모델(Student) 사이의 **이종 지식 증류(Heterogeneous KD)**와, 동일한 구조를 가진 소형 모델(Student)과 그 EMA(Exponential Moving Average) 버전인 코-티처(Co-teacher) 사이의 **동종 지식 증류(Homogeneous KD)**를 동시에 수행하는 것이다. 이를 통해 라벨 부족 문제를 해결함과 동시에, 거대 모델의 성능을 소형 모델로 전이시켜 메모리 효율적인 모델을 얻을 수 있다.

## 📎 Related Works

논문에서는 라벨 효율적인 학습을 위한 기존 접근 방식을 다음과 같이 설명한다.

1. **Self-ensembling 기반 반지도 학습**: Temporal Ensembling과 Mean Teacher(MT) 프레임워크가 대표적이다. 특히 Mean Teacher는 학생 모델의 가중치를 EMA 방식으로 업데이트하여 교사 모델을 만들고, 두 모델 간의 출력 일관성(Consistency)을 강제함으로써 라벨이 없는 데이터를 효과적으로 활용한다. 하지만 이러한 방식은 주로 동일한 구조의 네트워크 간 지식 전이에 집중한다.
2. **지식 증류(Knowledge Distillation, KD)**: 거대 교사 모델의 지식을 소형 학생 모델로 전이하여 모델을 압축하는 기술이다. 응답 기반, 특징 기반, 관계 기반 증류 등 다양한 방식이 제안되었으나, 대부분의 KD 방법은 대규모의 라벨링된 데이터셋을 필요로 한다는 한계가 있어 라벨이 부족한 의료 영상 분야에 적용하기 어렵다.

본 논문은 위 두 가지 접근 방식이 모두 '교사-학생 학습' 구조를 사용한다는 점에 착안하여, 이들을 통합한 비대칭 구조를 제안함으로써 라벨 부족과 모델 복잡도 문제를 동시에 해결하고자 한다.

## 🛠️ Methodology

ACT-Net은 Teacher 모델, Student 모델, 그리고 Co-teacher 모델로 구성된 파이프라인을 가진다.

### 1. Heterogeneous Knowledge Distillation (Hete-KD)

서로 다른 아키텍처를 가진 Teacher($f_t$)와 Student($f_s$) 간의 지식 전이를 수행한다. 동일한 입력 $x_i$에 대해 두 모델의 소프트 예측값(soft predictions)을 다음과 같이 생성한다.

$$P^s_i = \sigma(f_s(x_i; \theta_s) / \tau), \quad P^t_i = \sigma(f_t(x_i; \theta_t) / \tau)$$

여기서 $\sigma$는 softmax 함수, $\tau$는 확률 분포를 부드럽게 만드는 온도(temperature) 파라미터($\tau=20$으로 설정)이다. 학생 모델이 교사 모델의 거동을 모방하도록 하기 위해 Mean Squared Error(MSE) 손실 함수를 사용한 일관성 손실 $L^{kd}_{con}$을 정의한다.

$$L^{kd}_{con} = \sum_{i=1}^{N} \| P^s_i - P^t_i \|^2$$

### 2. Homogeneous Knowledge Distillation (Homo-KD)

라벨 부족 문제를 해결하기 위해 Student 모델과 동일한 구조를 가진 Co-teacher 모델($f_c$)을 구축한다. Co-teacher의 가중치 $\theta_c$는 Student의 가중치 $\theta_s$를 EMA 방식으로 업데이트하여 생성한다.

$$\theta^t_c = \alpha \theta^{t-1}_c + (1 - \alpha) \theta^t_s$$

여기서 $\alpha$는 EMA 감쇠율이다. Student와 Co-teacher에 서로 다른 섭동(perturbation, $\xi, \xi'$)을 준 입력을 넣고, 두 출력의 일관성을 MSE 손실 함수 $L^{co}_{con}$으로 강제한다.

$$L^{co}_{con} = L^{co}_{con}(f_s(x; \theta_s, \xi), f_c(x; \theta_c, \xi'))$$

### 3. Asymmetric Co-teaching Strategy 및 전체 손실 함수

Student 모델은 라벨링된 데이터 $x_s$에 대해 지도 학습 손실 $L_{seg}$를 사용한다. $L_{seg}$는 Dice loss와 Cross-entropy loss의 합으로 구성된다.

$$L_{seg} = L_{dice}(f_s(x_s; \theta_s), y_s) + L_{ce}(f_s(x_s; \theta_s), y_s)$$

최종적으로 Student 모델이 최적화해야 할 전체 손실 함수 $L_{stu}$는 다음과 같이 정의된다.

$$L_{stu} = L_{seg} + \lambda^{kd}_{con} L^{kd}_{con} + \lambda^{co}_{con} L^{co}_{con}$$

여기서 $\lambda^{kd}_{con}$과 $\lambda^{co}_{con}$은 각 손실 항의 비중을 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: ACDC 데이터셋 (심장 하부 구조 분할: LV, RV, MYO).
- **데이터 분할**: 총 100케이스 중 70(학습), 10(검증), 20(테스트). 학습 데이터 중 단 **10%(7케이스)**만 라벨링된 데이터로 사용하였다.
- **모델 구조**: U-Net을 백본으로 사용하였으며, 거대 모델은 U-Net [6, 64], 소형 모델은 U-Net [4, 16]을 사용하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC).

### 정량적 결과

- **모델 압축 효율**: Table 2에 따르면, 거대 모델 대비 소형 모델의 파라미터 수는 약 **250배** 적다.
- **성능 비교**:
  - 10%의 라벨만 사용했을 때, 일반적인 지도 학습(FS)이나 Mean Teacher(MT) 방식보다 ACT-Net이 더 높은 성능을 보였다.
  - 특히 소형 모델(U-Net [4, 16])을 사용했음에도 불구하고, ACT-Net은 거대 모델의 성능에 근접하는 결과를 얻었으며, 특정 설정에서는 10% 라벨의 MT 거대 모델보다 더 나은 성능을 기록하였다.
- **정성적 결과**: 시각화 결과, ACT-Net이 다른 방법들보다 분할 오류가 적고 더 정확한 경계를 찾아내는 것을 확인하였다.

### 절제 연구 (Ablation Analysis)

- Hete-KD와 Homo-KD 모두 개별적으로는 성능 향상에 기여한다.
- 흥미로운 점은 두 방법을 순차적으로 적용(Sequential combination)하는 것보다, ACT-Net처럼 **동일한 학습 단계에서 동시에 통합하여 학습시키는 것**이 더 좋은 결과를 냈다는 점이다. 저자들은 순차적 적용 시 '부정적 전이(negative transfer)'가 발생할 수 있다고 분석하였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분야에서 매우 치명적인 '라벨 부족'과 '모델 크기'라는 두 마리 토끼를 동시에 잡으려 했다는 점에서 강점이 있다. 특히, 단순히 모델을 압축하는 KD에 그치지 않고, 반지도 학습의 Consistency Regularization을 결합하여 라벨이 적은 상황에서도 KD가 효과적으로 작동할 수 있는 환경을 구축하였다.

비판적 해석 측면에서 보면, 본 연구는 ACDC라는 단일 데이터셋에서만 검증되었다는 한계가 있다. 다양한 장기나 다른 모달리티(CT, MRI 등)의 데이터셋에서도 동일한 파라미터 효율성과 성능 유지 능력이 나타나는지에 대한 추가 검증이 필요하다. 또한, EMA 업데이트 파라미터 $\alpha$나 $\tau$와 같은 하이퍼파라미터에 대한 민감도 분석이 부족하여, 다른 태스크에 적용할 때 최적의 값을 찾는 과정이 까다로울 수 있다.

## 📌 TL;DR

ACT-Net은 의료 영상 분할을 위해 **거대 모델로부터의 이종 지식 증류(Hete-KD)**와 **EMA 기반 코-티처를 통한 동종 지식 증류(Homo-KD)**를 동시에 수행하는 비대칭 프레임워크이다. 이 연구는 단 10%의 라벨 데이터만으로도 학습이 가능하며, 파라미터 수를 약 **250배** 줄이면서도 성능 손실이 거의 없는 경량화 모델을 생성할 수 있음을 입증하였다. 이는 자원이 제한된 임상 현장에서 고성능 의료 AI 모델을 배포하는 데 중요한 역할을 할 가능성이 높다.
