# SiamMOT: Siamese Multi-Object Tracking

Bing Shuai, Andrew Berneshawi, Xinyu Li, Davide Modolo, Joseph Tighe (2021)

## 🧩 Problem to Solve

본 논문은 온라인 다중 객체 추적(Online Multi-Object Tracking, MOT)의 성능 향상을 목표로 한다. 특히, 프레임 간 객체의 움직임을 예측하는 **Motion Modelling**이 온라인 MOT 성능에 미치는 영향에 주목한다.

기존의 온라인 추적기들은 주로 인접한 프레임 간의 지역적 연결(local linking)에 집중하며, 최근의 SOTA 모델들은 딥러닝을 통해 객체의 변위(displacement)를 예측함으로써 성능을 높여왔다. 그러나 많은 기존 방식들이 포인트 기반(point-based) 특징을 사용하거나 단순한 기하학적 특징에 의존하여, 빠른 움직임이나 복잡한 배경이 존재하는 환경에서 추적 성능이 저하되는 문제가 있다. 따라서 본 연구의 목표는 영역 기반(region-based) Siamese 네트워크를 도입하여 더욱 강건한 motion model을 구축하고, 이를 통해 온라인 MOT의 정확도와 효율성을 동시에 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **영역 기반 Siamese 네트워크를 MOT 프레임워크에 통합**하여 인스턴스 수준의 움직임을 정교하게 모델링하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **SiamMOT 아키텍처 제안**: Faster-RCNN 검출기와 Siamese 추적기를 결합하여, 객체 검출과 인스턴스 간의 시간적 연관성을 동시에 처리하는 네트워크를 설계하였다.
2. **두 가지 Motion Model 제시**: 움직임을 암시적으로 학습하는 **Implicit Motion Model (IMM)**과 템플릿 매칭을 통해 명시적으로 학습하는 **Explicit Motion Model (EMM)**을 제안하여, motion modelling의 방식이 추적 성능에 미치는 영향을 분석하였다.
3. **명시적 매칭의 우수성 입증**: EMM이 IMM 및 기존의 포인트 기반 방식(예: CenterTrack)보다 복잡한 시나리오(빠른 움직임, 심한 포즈 변화 등)에서 더 강력한 성능을 보임을 실험적으로 증명하였다.
4. **SOTA 성능 달성**: MOT17, TAO-person, HiEve 등 주요 벤치마크 데이터셋에서 기존의 최신 모델들을 상회하는 성능을 기록하였으며, 실시간 추적이 가능한 수준(720P 영상 기준 17 FPS)의 효율성을 보여주었다.

## 📎 Related Works

### 1. Single Object Tracking (SOT)에서의 Siamese Tracker

SOT는 첫 프레임에서 지정된 타겟을 이후 프레임에서 찾는 작업이다. Siamese 기반 추적기들은 두 프레임의 이미지 쌍을 입력으로 받아 매칭 함수를 통해 타겟의 위치를 예측한다. 본 논문은 이러한 SOT의 설계를 MOT의 인스턴스 연관성 문제로 확장하여 적용하였다.

### 2. Tracking-by-Detection 기반 MOT

대부분의 MOT 연구는 각 프레임에서 객체를 먼저 검출하고, 이후 시각적 유사성과 시공간적 일관성을 이용해 이들을 연결하는 방식을 취한다.

- **오프라인 추적기**: 전체 영상에 대해 거대한 그래프 최적화 문제를 풀지만, 계산 비용이 매우 커 실시간 적용이 어렵다.
- **온라인 추적기**: 미래 프레임의 정보 없이 즉각적으로 연관성을 계산한다. SORT가 대표적이며, 이후 Tracktor나 CenterTrack과 같이 검출기와 motion model을 통합 학습하는 방향으로 발전하였다.

### 3. 기존 접근 방식과의 차별점

SiamMOT는 CenterTrack과 같이 포인트 기반 특징을 사용하는 대신 **영역 기반(region-based) 특징**을 사용한다. 이는 인스턴스 인식 및 위치 추정에서 더 정교한 정보를 제공하며, 특히 EMM을 통해 템플릿 매칭 방식을 명시적으로 구현함으로써 빠른 움직임에 더 강건하게 대응할 수 있다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

SiamMOT는 **Faster-RCNN** 검출기를 기반으로 하며, 그 위에 영역 기반 Siamese 추적기를 추가한 구조이다. 전체 파이프라인은 다음과 같다.

1. 시간 $t$에서의 검출 결과 $R^t$와 시간 $t+\delta$에서의 프레임 $I_{t+\delta}$를 입력으로 받는다.
2. Siamese 추적기는 $R^t$의 각 인스턴스가 $I_{t+\delta}$의 어느 위치로 이동했는지 예측하여 $\tilde{R}^{t+\delta}$를 생성한다.
3. 동시에 검출기는 $I_{t+\delta}$에서 새로운 검출 결과 $R^{t+\delta}$를 생성한다.
4. SORT 알고리즘과 유사한 공간적 매칭 프로세스를 통해 $\tilde{R}^{t+\delta}$와 $R^{t+\delta}$를 결합하여 궤적(trajectory)을 업데이트한다.

### 주요 구성 요소 및 방법론

#### 1. Siamese Tracker의 기본 동작

시간 $t$의 인스턴스 $i$에 대해, 시간 $t+\delta$의 탐색 영역 $S^{t+\delta}_i$ (이전 위치 $R^t_i$를 $r$배 확장한 영역)에서 해당 인스턴스를 찾는다.
$$(v^{t+\delta}_i, \tilde{R}^{t+\delta}_i) = T(f^t_{R_i}, f^{t+\delta}_{S_i}; \Theta)$$
여기서 $f^t_{R_i}$는 타겟 영역의 특징 맵, $f^{t+\delta}_{S_i}$는 탐색 영역의 특징 맵이며, $T$는 학습 가능한 Siamese 추적기이다. $v^{t+\delta}_i$는 가시성 신뢰도(visibility confidence)를 의미한다.

#### 2. Implicit Motion Model (IMM)

IMM은 MLP(Multi-Layer Perceptron)를 사용하여 두 프레임 간의 움직임을 암시적으로 추정한다.

- **절차**: 타겟 특징과 탐색 특징을 연결(concatenate)하여 MLP에 입력하고, 가시성 신뢰도 $v_i$와 상대적 위치/크기 변화량 $m_i$를 예측한다.
- **방정식**: $m_i$는 다음과 같이 정의된다.
$$m_i = [\frac{x^{t+\delta}_i - x^t_i}{w^t_i}, \frac{y^{t+\delta}_i - y^t_i}{h^t_i}, \log \frac{w^{t+\delta}_i}{w^t_i}, \log \frac{h^{t+\delta}_i}{h^t_i}]$$
- **손실 함수**: 가시성 예측을 위한 Focal Loss와 위치 회귀를 위한 Smooth $\text{L}_1$ Loss의 합으로 구성된다.
$$\mathcal{L} = \ell_{\text{focal}}(v_i, v^*_i) + \mathbb{1}[v^*_i] \ell_{\text{reg}}(m_i, m^*_i)$$

#### 3. Explicit Motion Model (EMM)

EMM은 채널별 교차 상관(channel-wise cross-correlation) 연산을 통해 픽셀 수준의 응답 맵을 생성하는 명시적 방식이다.

- **절차**: 타겟 특징 $f^t_{R_i}$와 탐색 특징 $f^{t+\delta}_{S_i}$를 상관 연산($*$)하여 응답 맵 $r_i$를 생성한다. 이후 FCN($\psi$)을 통해 가시성 신뢰도 맵 $v_i$와 위치 오프셋 맵 $p_i$를 예측한다.
- **패널티 맵 ($\eta_i$)**: 갑작스러운 과도한 움직임을 방지하기 위해 코사인 윈도우 함수($C$)와 가우시안 함수($S$)를 결합한 패널티 맵을 적용하여 최종 위치를 결정한다.
$$\eta_i(x, y) = \lambda C + (1-\lambda) S(R(p(x, y)), R^t_i)$$
- **최종 위치 결정**: $\tilde{R}^{t+\delta}_i$는 $v_i \odot \eta_i$ 값이 최대가 되는 지점 $(x^*, y^*)$에서 결정된다.
- **손실 함수**: 가시성 맵에 대한 Focal Loss와 IOU Loss를 사용하며, 이때 Centerness $w(x, y)$를 가중치로 사용하여 중심부 예측의 정확도를 높인다.

### 학습 및 추론 절차

- **학습**: 검출기(RPN, Detection head)와 추적기(SiamMOT)를 엔드투엔드(end-to-end)로 함께 학습시킨다. 전체 손실 함수는 $\mathcal{L} = \ell_{\text{rpn}} + \ell_{\text{detect}} + \ell_{\text{motion}}$이다.
- **추론**:
    1. 검출 결과와 추적 결과에 대해 각각 NMS를 수행한다.
    2. IOU $\ge 0.5$인 경우 중복된 검출을 제거한다.
    3. 신뢰도 임계값 $\alpha, \beta$를 기준으로 궤적의 생성, 유지, 소멸을 결정한다.
    4. **단기 폐쇄(Short Occlusion) 처리**: 객체가 일시적으로 보이지 않더라도 $\tau$ 프레임 동안은 메모리에 유지하며 지속적으로 탐색하여 재연결을 시도한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MOT17 (혼잡한 장면), TAO-person (대규모, 다양한 장면), CRP (빠른 카메라 움직임)
- **지표**: MOTA, IDF1, TrackAP (TAO-person)
- **기준선 (Baselines)**: Tracktor, Tracktor + Flow, CenterTrack, DeepMOT 등

### 주요 결과

1. **Motion Model의 중요성**: 실험 결과 Tracktor &lt; Flow < IMM &lt; EMM 순으로 성능이 향상됨을 확인하였다. 특히 빠른 움직임이 특징인 CRP 데이터셋에서 SiamMOT는 Tracktor + Flow 대비 MOTA가 35, IDF1이 25 포인트 상승하는 비약적인 성능 향상을 보였다.
2. **정량적 성능 (MOT17)**: EMM을 사용한 SiamMOT는 65.9 MOTA / 63.3 IDF1를 달성하여, CenterTrack(61.5 MOTA)을 포함한 기존 SOTA 모델들을 압도하였다.
3. **정량적 성능 (TAO-person)**: Tracktor++(36.7 TAP) 대비 41.1 TAP를 기록하며 성능을 크게 향상시켰으며, Re-ID 모델과 결합했을 때 더욱 높은 성능을 보였다.
4. **HiEve 챌린지**: DLA-169 백본을 사용한 모델이 이전 챌린지 우승자들의 성능을 모두 경신하였다.
5. **효율성**: 720P 영상 기준 단일 GPU에서 17 FPS로 동작하여 실시간 추적 가능성을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **명시적 매칭의 효용성**: EMM이 IMM보다 우수한 이유는 채널 독립적 상관 연산을 통해 동일 인스턴스 간의 매칭 함수를 명시적으로 학습하고, 픽셀 수준의 세밀한 감독(supervision)을 통해 방해 요소(distractors)에 의한 오매칭을 줄였기 때문이다.
- **영역 기반 특징의 우위**: 포인트 기반 방식(CenterTrack)보다 영역 기반 특징이 인스턴스 인식 및 지역화에 더 유리하며, 이것이 추적 성능 향상으로 직결됨을 보여주었다.
- **결합 학습(Joint Training)의 효과**: 검출기와 추적기를 함께 학습시켰을 때, 단독 학습 시보다 MOTA가 상승(61.5 $\to$ 63.3)하는 것을 통해 두 모듈 간의 상호 보완적 학습 효과를 확인하였다.

### 한계 및 향후 과제

- **장기 폐쇄 문제**: 현재의 motion model은 단기 폐쇄(약 1초 내외)는 처리할 수 있으나, 객체가 오랫동안 사라졌다가 다시 나타나는 장기 폐쇄 상황에서는 한계가 있다. 이는 향후 더 발전된 motion modelling이나 Re-ID 기법과의 통합이 필요함을 시사한다.
- **데이터셋 편향**: 주로 사람 추적에 집중되어 있으나, 제안된 프레임워크는 범용적이므로 향후 다중 클래스(multi-class) MOT로 확장 가능할 것으로 보인다.

## 📌 TL;DR

SiamMOT는 Faster-RCNN 검출기에 영역 기반 Siamese 추적기를 통합하여 온라인 MOT의 성능을 극대화한 연구이다. 특히 **명시적 모션 모델(EMM)**을 통해 템플릿 매칭 기반의 정교한 움직임 예측을 구현함으로써, 빠른 움직임이나 복잡한 환경에서도 매우 강건한 추적 성능을 보여주었다. 이 연구는 온라인 MOT에서 인스턴스 수준의 정밀한 motion modelling이 결정적인 역할을 한다는 것을 입증하였으며, 향후 실시간 다중 객체 추적 시스템의 성능 기준을 높이는 데 기여할 것으로 평가된다.
