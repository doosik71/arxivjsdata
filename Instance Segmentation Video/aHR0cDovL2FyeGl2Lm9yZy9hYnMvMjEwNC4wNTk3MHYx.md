# Crossover Learning for Fast Online Video Instance Segmentation

Shusheng Yang, Yuxin Fang, Xinggang Wang, Yu Li, Chen Fang, Ying Shan, Bin Feng, Wenyu Liu (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 비디오 인스턴스 분할(Video Instance Segmentation, VIS) 작업에서 **추론 속도(Latency)와 정확도(Accuracy) 사이의 효율적인 트레이드-오프(Trade-off)를 달성**하는 것이다.

VIS는 비디오 시퀀스 내의 각 인스턴스에 대해 픽셀 단위의 라벨링을 수행하는 작업으로, 단순한 이미지 분할을 넘어 시간적 맥락(Temporal Context)을 모델링해야 한다. 하지만 기존의 시간적 정보 활용 방식들은 다음과 같은 한계가 있다.
1. **추론 비용 증가**: 픽셀 또는 인스턴스 수준의 특징 집계(Feature Aggregation)를 위해 추가적인 네트워크 파라미터나 연산량이 필요한 경우가 많다.
2. **학습 불안정성**: 메트릭 학습(Metric Learning) 기반의 인스턴스 연관(Association) 방식은 샘플링된 프레임 쌍에 의존하므로 학습이 불안정하고 수렴 속도가 느리다.

따라서 본 논문의 목표는 추가적인 추론 비용 없이 시간적 정보를 효과적으로 학습하고, 안정적인 인스턴스 연관을 가능하게 하는 빠른 온라인 VIS 모델인 **CrossVIS**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Crossover Learning**과 **Global Balanced Instance Embedding**이라는 두 가지 설계 전략으로 요약된다.

1. **Crossover Learning**: 현재 프레임의 인스턴스 특징(Appearance 정보)을 사용하여 다른 프레임에서 동일한 인스턴스를 픽셀 단위로 지역화(Localize)하도록 학습하는 방식이다. 이는 추론 시에는 추가 연산이 없으며, 학습 과정에서 인스턴스 분할 손실 함수와 통합되어 인스턴스 표현력을 강화하고 배경 노이즈를 억제한다.
2. **Global Balanced Instance Embedding**: 기존의 쌍별(Pair-wise) 지역 임베딩 방식 대신, 전체 학습 데이터셋의 모든 ID를 대표하는 학습 가능한 전역 프록시(Global Proxy) 가중치를 도입한다. 여기에 Focal Loss를 적용하여 양성-음성 샘플 간의 불균형 문제를 해결함으로써 더 안정적이고 정확한 인스턴스 연관을 구현한다.
3. **CrossVIS 모델**: 위 두 가지 기법을 결합하여 YouTube-VIS-2019, OVIS, YouTube-VIS-2021 벤치마크에서 온라인 VIS 방법론 중 최첨단(SOTA) 성능과 우수한 속도를 동시에 달성하였다.

## 📎 Related Works

논문에서는 기존의 시간적 정보 모델링 방식을 네 가지 범주로 분류하여 설명한다.
1. **Pixel-level feature aggregation**: Non-local network나 3D convolution을 사용하여 현재 프레임의 픽셀 특징을 강화하는 방식 (예: STM-VOS, STEm-Seg).
2. **Instance-level feature aggregation**: Deformable convolution이나 Spectral clustering을 통해 프레임 간 인스턴스 특징을 융합하는 방식 (예: MaskProp, SELSA).
3. **Metric learning based association**: 별도의 연관 헤드를 통해 인스턴스를 연결하는 방식 (예: MaskTrack R-CNN, SipMask-VIS).
4. **Post-processing**: Dynamic programming이나 learnable linking을 통해 결과를 정제하는 방식 (예: Seq-NMS, ObjLink).

**CrossVIS와의 차별점**: 
Crossover Learning은 기존의 특징 집계 방식과 달리 추가적인 네트워크 블록(Alignment, Fusion)이 필요 없으므로 추론 비용이 전혀 발생하지 않는다. 또한, 메트릭 학습 기반 방식과 달리 별도의 손실 함수가 아닌 인스턴스 분할 손실과 통합되어 동작하며, 후처리 방식과 달리 역전파(Back-propagation)를 통한 end-to-end 최적화가 가능하다.

## 🛠️ Methodology

### 1. 기본 마스크 생성 (Still-image Mask Generation)
CrossVIS는 CondInst를 기반으로 하며, 동적 조건부 컨볼루션(Dynamic Conditional Convolutions)을 사용한다. 위치 $(x,y)$에 있는 인스턴스의 마스크 $M_{x,y}$는 다음과 같이 생성된다.

$$\tilde{F}_{x,y} = \text{Concat}(F_{\text{mask}}; O_{x,y}) \quad (1)$$
$$M_{x,y} = \text{MaskHead}(\tilde{F}_{x,y}; \theta_{x,y}) \quad (2)$$

여기서 $\tilde{F}_{x,y}$는 마스크 특징 맵과 상대 좌표 $O_{x,y}$의 결합이며, $\theta_{x,y}$는 컨트롤러 헤드에서 생성된 인스턴스별 동적 필터(Dynamic Filter)이다.

### 2. Crossover Learning
Crossover Learning의 핵심은 한 프레임의 외형 정보($\theta$)를 다른 프레임의 위치 정보($O$)와 결합하여 마스크를 생성하는 것이다.

- **정방향 Crossover**: 시간 $t$의 필터 $\theta_{x,y}(t)$를 사용하여 시간 $t+\delta$의 특징 맵 $\tilde{F}'_{x',y'}(t+\delta)$로부터 마스크를 생성한다.
$$M^\times_{x',y'}(t+\delta) = \text{MaskHead}(\tilde{F}'_{x',y'}(t+\delta); \theta_{x,y}(t)) \quad (6)$$
- **역방향 Crossover**: 시간 $t+\delta$의 필터 $\theta'_{x',y'}(t+\delta)$를 사용하여 시간 $t$의 특징 맵 $\tilde{F}_{x,y}(t)$로부터 마스크를 생성한다.
$$M^\times_{x,y}(t) = \text{MaskHead}(\tilde{F}_{x,y}(t); \theta'_{x',y'}(t+\delta)) \quad (7)$$

학습 시에는 예측된 마스크 $M$과 정답 마스크 $M^*$ 사이의 Dice Loss를 사용하여 최적화한다.
$$L_{\text{dice}}(M, M^*) = 1 - \frac{2 \sum HW_i M_i M^*_i}{\sum HW_i (M_i)^2 + \sum HW_i (M^*_i)^2} \quad (8)$$

### 3. Global Balanced Instance Embedding
인스턴스 연관을 위해 전역 프록시 가중치 $\{w_M\}$을 도입한다. 인스턴스 $I_i$가 클래스 $n$에 속할 확률 $p_i(n)$은 다음과 같이 정의된다.

$$p_i(n) = \frac{\exp(e_i^\top w_n)}{\sum_{j=1}^M \exp(e_i^\top w_j)} \quad (11)$$

여기서 $e_i$는 인스턴스 임베딩이다. 클래스 불균형 문제를 해결하기 위해 Focal Loss를 도입하여 학습한다.
$$L_{\text{id}} = L_{\text{Focal}} = -\alpha_t (1-p_i(n))^\gamma \log(p_i(n)) \quad (14)$$

### 4. 전체 학습 절차 및 손실 함수
모든 태스크를 end-to-end로 공동 학습하며, 전체 손실 함수는 다음과 같다.
$$L = L_{\text{det}} + L_{\text{seg}} + L_{\text{cross}} + L_{\text{id}} \quad (15)$$
여기서 $L_{\text{cross}}$는 식 (6)과 (7)을 통해 생성된 Crossover 마스크들에 대한 Dice Loss의 합이다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: YouTube-VIS-2019, OVIS, YouTube-VIS-2021.
- **지표**: Average Precision (AP) 및 Average Recall (AR).
- **구현**: ResNet-50, ResNet-101, DLA-34 백본 사용.

### 2. 주요 정량적 결과 (YouTube-VIS-2019)
- **정확도 및 속도**: ResNet-50 기반의 CrossVIS는 단일 스케일 학습 시 **34.8 AP**, 멀티 스케일 학습 시 **36.3 AP**를 달성하였으며, 속도는 **39.8 FPS**로 매우 빠르다.
- **효율성**: DLA-34 백본을 사용한 CrossVIS-Lite는 **36.2 AP**와 **48.5 FPS**라는 매우 뛰어난 속도-정확도 트레이드-오프를 보여준다.
- **타 모델 비교**: 온라인 방식인 MaskTrack R-CNN 및 SipMask-VIS보다 높은 AP를 기록하였으며, 오프라인 방식인 VisTR보다도 우수한 성능을 보였다.

### 3. 기타 벤치마크 결과
- **OVIS**: 매우 어려운 데이터셋임에도 불구하고, CrossVIS는 **18.1 AP**(멀티 스케일)를 기록하며 비교 대상 모델들 중 가장 높은 성능을 보였다.
- **YouTube-VIS-2021**: MaskTrack R-CNN 및 SipMask-VIS 대비 큰 폭의 성능 향상을 이루었다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
- **비용 없는 성능 향상**: Crossover Learning은 오직 학습 단계에서만 작동하며, 추론 단계에서는 추가 연산이 전혀 없다. 이는 실시간성이 중요한 온라인 VIS 작업에서 매우 강력한 이점이다.
- **시간적 일관성 확보**: 실험 결과, 시간 간격 $\delta$가 클수록 Crossover Learning의 효과가 더 크게 나타났다. 이는 외형 변화가 심하거나 배경이 바뀌는 상황에서도 동일 인스턴스를 식별할 수 있는 강건한 표현력을 학습했음을 의미한다.
- **안정적인 연관**: Global Balanced Embedding은 기존 Pair-wise 방식보다 AP의 표준편차($\sigma_{AP}$)를 현저히 낮추어, 샘플링에 의존하지 않는 안정적인 수렴 상태를 제공한다.

### 2. 한계 및 비판적 해석
- **데이터셋 의존성**: OVIS 데이터셋에서 모든 모델의 성능이 급격히 하락하는 현상이 관찰되었다. 이는 오클루전(Occlusion)이 심한 환경에서 여전히 VIS 작업이 매우 어렵다는 점을 시사하며, 향후 더 강력한 오클루전 처리 메커니즘이 필요함을 보여준다.
- **가정**: 본 논문은 추론 시 $\delta=1$ (프레임 단위 처리)을 가정하는 온라인 방식을 취하고 있어, 오프라인 방식이 가질 수 있는 미래 프레임 정보 활용의 이점은 완전히 누리지 못한다.

## 📌 TL;DR

본 논문은 추론 비용 증가 없이 비디오 인스턴스 분할 성능을 높이는 **CrossVIS** 모델을 제안한다. 핵심은 학습 시에만 작동하여 시간적 강건성을 키우는 **Crossover Learning**과, 전역 프록시와 Focal Loss를 통해 인스턴스 연관의 안정성을 높인 **Global Balanced Embedding**이다. 결과적으로 ResNet-50 기반으로 약 40 FPS의 빠른 속도와 SOTA 수준의 정확도를 동시에 달성하여, 실제 서비스 적용 가능성이 매우 높은 효율적인 온라인 VIS 베이스라인을 제시하였다.