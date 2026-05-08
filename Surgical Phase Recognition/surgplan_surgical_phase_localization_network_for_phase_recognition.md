# SURGPLAN: SURGICAL PHASE LOCALIZATION NETWORK FOR PHASE RECOGNITION

Xingjian Luo, You Pang, Zhen Chen, Jinlin Wu, Zongmin Zhang, Zhen Lei, Hongbin Liu (2023)

## 🧩 Problem to Solve

본 논문은 스마트 수술실의 수술 이해를 위한 필수 요소인 수술 단계 인식(Surgical Phase Recognition)에서 발생하는 두 가지 핵심 문제를 해결하고자 한다.

첫째, 기존의 많은 방법론이 단순한 2D 네트워크를 사용함으로써 각 프레임의 변별력 있는 시각적 특징과 모션 정보(Motion Information)를 충분히 캡처하지 못한다는 점이다. 3D CNN을 사용하는 방법도 존재하지만, 이는 계산 비용이 너무 높고 효율성이 떨어진다는 한계가 있다.

둘째, 프레임 단위(Frame-by-frame) 인식 패러다임으로 인해 발생하는 'Phase Shaking' 문제이다. 이는 한 단계(Phase) 내에서 예측값이 불안정하게 변하여 불연속적인 결과가 나타나는 현상을 의미하며, 이는 수술 단계 예측의 성능을 저하시킬 뿐만 아니라 실제 의료진의 효율적인 수술 모니터링을 방해하는 심각한 요소가 된다.

따라서 본 논문의 목표는 Temporal Detection 원리를 도입하여 더욱 정확하고 안정적인 수술 단계 인식을 가능하게 하는 SurgPLAN 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 단계 인식을 단순한 분류(Classification) 문제가 아닌, 시간적 영역 제안(Temporal Region Proposals)을 통한 검출(Detection) 문제로 재정의하는 것이다.

이를 위해 두 가지 핵심 모듈을 설계하였다. 첫째, 서로 다른 프레임 샘플링 속도를 가진 두 개의 브랜치를 통해 다중 스케일의 공간 및 시간 특징을 추출하는 Pyramid SlowFast (PSF) 아키텍처를 제안하였다. 둘째, 시간적 영역 제안을 기반으로 단계를 예측하여 예측의 일관성과 안정성을 보장하는 Temporal Phase Localization (TPL) 모듈을 제안하였다.

## 📎 Related Works

기존의 수술 단계 인식 연구들은 주로 2D CNN으로 특징을 추출한 후, Temporal Convolutional Networks (TCN), LSTM, 또는 Transformer와 같은 구조를 사용하여 시간적 정보를 활용해 왔다. 그러나 이러한 프레임 단위 예측 방식은 전역적인 정보(Global field of information)가 부족하여 앞서 언급한 Phase Shaking 문제에 취약하다는 한계가 있다.

본 연구는 자연어 비디오에서 동작의 경계를 검출하고 제안을 생성하는 Temporal Action Localization (TAL) 접근 방식에서 영감을 얻었으며, 이를 수술 비디오의 특성에 맞게 최적화하여 기존의 프레임 기반 접근 방식과 차별화를 두었다.

## 🛠️ Methodology

### 전체 파이프라인

SurgPLAN의 전체 구조는 입력 비디오에서 특징을 추출하는 PSF 모듈과, 추출된 특징을 바탕으로 시간적 영역 제안을 생성하는 TPL 모듈로 구성된다. 최종적으로 최적의 제안(Proposal)을 선택하여 수술 단계의 시작점과 끝점, 그리고 해당 단계를 결정한다.

### Pyramid SlowFast (PSF) Architecture

PSF는 입력 비디오 $V \in \mathbb{R}^{T \times (H \times W) \times N}$로부터 공간 및 시간 정보를 추출한다.

1. **SlowFast 특징 추출**: 낮은 프레임 속도로 공간적 정보를 캡처하는 Slow Path와 높은 프레임 속도로 모션 정보를 캡처하는 Fast Path를 병렬로 운영한다.
    $$v_{slow} = \phi_{slow}(V)$$
    $$v_{fast} = \phi_{fast}(V)$$
    $$v_{fuse} = D([v_{slow}, v_{fast}])$$
    여기서 $\phi_{slow}$와 $\phi_{fast}$는 각 브랜치 네트워크이며, $D$는 결합된 특징을 특징 공간으로 변환하는 다운샘플링 백본이다.

2. **Pyramid Temporal Features**: 추출된 $v_{fuse}$를 서로 다른 풀링 윈도우 크기 $S_{window} \in \{1, 2, 4\}$를 사용하여 Max-pooling 함으로써 3가지 다른 다운샘플링 비율의 특징 시퀀스를 생성한다. 이는 TPL 모듈이 다양한 크기의 시간적 세그먼트에 적응할 수 있도록 돕는다.

### Temporal Phase Localization (TPL) Module

TPL 모듈은 특정 스케일의 특징 $f$를 입력받아 세 개의 분류 네트워크($\psi_{phase}, \psi_{start}, \psi_{end}$)와 한 개의 회귀 네트워크를 통해 영역 제안을 생성한다.

1. **특징 예측**: 각 시간 위치 $i$의 특징 $f_i$에 대해 단계 확률 $p_{phase}^i$, 시작점 확률 $p_{start}^i$, 끝점 확률 $p_{end}^i$를 독립적으로 예측한다.
2. **영역 제안 생성**: 중심점 $f_i$가 주어졌을 때, 회귀 네트워크가 출력하는 조건부 확률 분포 $P(B|f_i)$와 분류 네트워크의 확률값을 더해 시작점 $\hat{t}_{start}$와 끝점 $\hat{t}_{end}$를 결정한다.
    $$\hat{t}_{start}(f_i) = \text{Argmax}(P(B|f_i)_{[i-\frac{B}{2}:i-1]} + p_{[i-\frac{B}{2}:i-1]}^{start})$$
    $$\hat{t}_{end}(f_i) = \text{Argmax}(P(B|f_i)_{[i+1:i+\frac{B}{2}]} + p_{[i+1:i+\frac{B}{2}]}^{end})$$
3. **제안 선택**: 생성된 수많은 제안 중 중복되거나 점수가 낮은 것을 제거하기 위해 Temporal Non-Maximum Suppression (NMS)을 적용한다. 추론 단계에서는 Soft-NMS를 사용하여 일반화 성능을 높인다.

### 손실 함수 (Loss Functions)

- **Focal Loss**: 수술 비디오 내 배경 특징이 매우 많기 때문에, 배경보다 전경(Foreground)에 집중하기 위해 사용한다.
    $$L_{focal} = (1-p_{phase}^i)^\gamma \log(p_{phase}^i)$$
- **Temporal IoU Loss**: 1차원 시간 시나리오에 맞게 IoU Loss를 변형하여 사용함으로써, 프레임 단위 예측에서 벗어나 전역적인 관점에서 수술 단계를 예측하게 한다.
    $$L_{IoU} = 1 - \frac{\min(\hat{s}_{start}, s_{start}) + \min(\hat{s}_{end}, s_{end})}{\max(\hat{s}_{start}, s_{start}) + \max(\hat{s}_{end}, s_{end})}$$
    여기서 $s$는 Ground Truth 거리, $\hat{s}$는 예측된 거리이다.

## 📊 Results

### 실험 설정

- **데이터셋**: CATARACTS 데이터셋(백내장 수술 비디오)을 사용하였으며, 25개 비디오는 학습용, 25개는 테스트용으로 나누었다. 총 19개의 단계 카테고리가 포함되어 있다.
- **평가 지표**: 검출 성능 측정을 위한 mAP(mean Average Precision)와 분류 성능 측정을 위한 Accuracy, F1 score, Precision, Recall을 사용하였다.

### 주요 결과

- **정량적 결과**: Table 1에 따르면 SurgPLAN은 Accuracy 83.10%, F1 score 71.87%를 기록하며 기존 SOTA 방법론들을 크게 상회하였다. 특히 SV-RCNet 대비 F1 score에서 13.14%의 향상을 보였으며, SlowFast 대비 Accuracy는 15.59% 향상되었다.
- **정성적 결과**: Color-coded ribbon 시각화 결과(Fig. 4), SurgPLAN은 기존 방법들(TeCNO, SlowFast, SV-RCNet)보다 Ground Truth에 가장 근접한 결과를 보였으며, 특히 Phase Shaking 문제가 현저히 해결되었음을 확인하였다.

### Ablation Study

- **백본 학습 영향**: 학습된 백본을 사용했을 때 mAP가 크게 상승하여, TAL 작업 이전에 숙련된 백본을 구축하는 것이 중요함을 입증하였다.
- **특징 집계 방식**: 세그먼트 내 모든 특징의 평균(Average)을 사용하는 것보다 중심점(Center) 특징을 사용하는 것이 성능이 더 높았다. 이는 경계 부분의 불필요한 특징이 중심점의 핵심 특징을 희석시킬 수 있음을 시사한다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 수술 단계 인식을 '분류'가 아닌 '검출'의 관점으로 접근하여, 수술 비디오 특유의 긴 세그먼트와 불연속성 문제를 효과적으로 해결했다는 점이다. 특히 Pyramid SlowFast 구조를 통해 다양한 시간적 스케일의 특징을 포착함으로써 인식의 정확도를 높였다.

다만, 논문에서 명시적으로 다루지 않은 부분은 PSF의 사전 학습(Pre-training)에 소요되는 계산 자원과 시간으로, 실제 적용 시 이 단계가 병목이 될 가능성이 있다. 또한, 백내장 수술 외에 다른 종류의 수술 비디오에서도 동일한 안정성 향상 효과가 나타날지에 대한 범용성 검증은 향후 과제로 남는다.

비판적으로 해석하자면, 제안된 방법론은 Temporal Action Localization의 구조를 수술 도메인에 맞게 적절히 변형한 형태이며, 완전히 새로운 아키텍처의 제안보다는 기존의 강력한 검출 프레임워크를 수술 단계 인식 문제에 성공적으로 이식하여 Phase Shaking이라는 고질적인 문제를 해결한 실용적인 연구라고 평가할 수 있다.

## 📌 TL;DR

SurgPLAN은 수술 단계 인식에서 발생하는 예측 불안정성(Phase Shaking)을 해결하기 위해, 프레임 단위 분류 대신 **Temporal Detection** 패러다임을 도입한 네트워크이다. **Pyramid SlowFast (PSF)**를 통해 다중 스케일의 시공간 특징을 추출하고, **Temporal Phase Localization (TPL)** 모듈로 안정적인 단계 세그먼트를 검출한다. 실험 결과, CATARACTS 데이터셋에서 기존 SOTA 모델 대비 월등한 정확도와 안정성을 보였으며, 이는 향후 실시간 수술 모니터링 시스템의 신뢰성을 높이는 데 중요한 기여를 할 것으로 보인다.
