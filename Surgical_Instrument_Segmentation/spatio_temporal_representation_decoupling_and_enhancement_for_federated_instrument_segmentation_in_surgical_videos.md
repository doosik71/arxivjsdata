# Spatio-Temporal Representation Decoupling and Enhancement for Federated Instrument Segmentation in Surgical Videos

Zheng Fang, Xiaoming Qi, Chun-Mei Feng, Jialun Pei, Weixin Si, and Yueming Jin (2025)

## 🧩 Problem to Solve

본 논문은 수술 비디오 내 수술 도구 분할(Surgical Instrument Segmentation)을 위한 연합 학습(Federated Learning, FL) 프레임워크 구축을 목표로 한다. 수술 도구 분할은 로봇 보조 최소 침습 수술(RAS)에서 매우 중요하지만, 여러 의료 기관의 데이터를 중앙 집중화하는 것은 환자의 개인정보 보호 문제로 인해 현실적으로 불가능하다. 따라서 데이터 공유 없이 협력 학습이 가능한 FL이 유망한 대안으로 제시된다.

그러나 기존의 일반적인 FL 및 개인화 연합 학습(Personalized Federated Learning, PFL) 방식은 수술 도메인의 다음과 같은 고유한 특성을 고려하지 못하고 있다.

1. **배경의 다양성과 도구의 유사성:** 수술 종류나 병원마다 해부학적 배경(Background tissue)은 매우 다양하지만, 사용되는 수술 도구의 외형과 움직임 패턴은 매우 유사하다.
2. **합성 데이터의 존재와 도메인 갭:** 수술 시뮬레이터를 통해 대규모 합성 데이터를 쉽게 생성할 수 있으나, 합성 데이터와 실제 수술 영상 사이의 상당한 도메인 갭(Domain Gap)이 존재하여 이를 효과적으로 활용하는 방법이 필요하다.

결과적으로 본 논문은 수술 도메인의 지식을 활용하여 사이트별 배경 차이를 극복하고, 합성 데이터를 통해 글로벌 모델의 일반화 성능을 높이는 PFL 스킴인 **FedST**를 제안한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 도구의 **공통된 시공간적 표현(Spatio-Temporal Representation)**과 각 사이트의 **고유한 배경 표현**을 분리(Decoupling)하고 강화하는 것이다.

- **RSC (Representation Separation and Cooperation) 메커니즘:** 지역 사이트 학습 시, 배경을 인코딩하는 Query Embedding 레이어는 개인화(Private)하여 유지하고, 도구의 공통 특성과 시공간적 움직임을 포착하는 나머지 파라미터는 글로벌하게 공유하여 협력 학습을 수행한다.
- **Textual-guided Channel Selection:** 사전 학습된 CLIP 모델을 사용하여 각 사이트의 수술 타입에 맞는 텍스트 가이드를 제공함으로써, 사이트별 특성에 맞는 특징 채널을 선택적으로 강화한다.
- **SERQ (Synthesis-based Explicit Representation Quantification) 전략:** 서버 측 학습에서 합성 데이터를 활용하여 도구의 명시적인 표현(Explicit Representation)을 정량화한다. 이를 통해 도메인 갭을 줄이고, 각 로컬 모델의 수렴을 동기화하여 글로벌 모델의 일반화 성능을 향상시킨다.

## 📎 Related Works

### 관련 연구 및 한계

- **수술 도구 분할:** 기존 연구들은 Holistically-nested networks, Graph-based networks, SAM(Segment Anything Model) 파인튜닝 등을 통해 성능을 높였으나, 대부분 단일 데이터셋 기반의 로컬 학습에 치중하여 여러 사이트의 협력 학습 이점을 활용하지 못했다.
- **연합 학습(FL) 및 개인화 연합 학습(PFL):** 의료 영상 분할(폴립, 시신경 유두 등)에 PFL이 적용되어 왔으며, 주로 예측 헤드(Prediction head)나 Batch Normalization 레이어를 개인화하는 방식을 사용했다. 하지만 수술 비디오와 같은 시공간적(Spatio-temporal) 데이터의 특성을 반영한 PFL 연구는 부족한 실정이다.

### 차별점

FedST는 단순히 모델의 일부 레이어를 분리하는 것을 넘어, 수술 도구의 시공간적 일관성과 배경의 다양성이라는 도메인 지식을 아키텍처 설계(Query Embedding 분리 및 Temporal Modeling)와 학습 전략(합성 데이터 기반 SERQ)에 직접 반영했다는 점에서 기존 PFL 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인

FedST는 지역 사이트 단계의 **RSC**와 서버 단계의 **SERQ**로 구성된다. 학습은 합성 데이터로 사전 학습된 모델에서 시작하여, $T$ 라운드 동안 로컬 업데이트와 글로벌 집계(Aggregation)를 반복한다.

### 2. Representation Separation and Cooperation (RSC)

RSC는 도구의 공통 표현과 배경의 고유 표현을 분리한다.

- **시공간적 표현 모델링:** 현재 프레임 $x_{ts}$와 이전 프레임들 $\{x_{ts-m}, \dots, x_{ts-1}\}$을 입력으로 받아, 멀티 스케일 접근법과 Multi-head Cross-Attention을 통해 도구의 움직임 패턴을 캡처한다.
- **파라미터 분리:** 모델 파라미터 $v$를 개인화 파라미터 $\rho$(Self-attention의 Query Embedding 레이어)와 공유 파라미터 $\gamma$(그 외 나머지)로 분리한다.
  - $\rho$는 각 사이트의 고유한 배경을 학습하며 로컬에서만 업데이트된다.
  - $\gamma$는 도구의 공통 표현을 학습하며 서버를 통해 평균화되어 공유된다.
- **Textual-guided Channel Selection:** CLIP 모델에 "This is the Local site X model and the surgery type is XXX"와 같은 텍스트 프롬프트를 입력하여 사이트별 지표 $\xi^k$를 생성한다. 이를 통해 특징 맵 $F^*_{ts}$에서 해당 사이트에 중요한 채널을 선택적으로 강화한다:
  $$F'_{ts} = F^*_{ts} + F^*_{ts} \otimes \hat{\xi}^k$$
  여기서 $\hat{\xi}^k$는 CLIP 지표와 특징 맵의 Global Average Pooled 값을 결합하여 생성된 composite indicator이다.

### 3. Synthesis-based Explicit Representation Quantification (SERQ)

서버에서 합성 데이터를 사용하여 글로벌 모델을 강화하는 전략이다.

- **명시적 정량화 (Explicit Quantification):** 특징 표현 $F$를 도메인 표현 $D$(배경 및 스타일)와 도구 공통 표현 $I$의 합으로 정의한다 ($F = D + I$).
- **도메인 디스크립터 학습:** 서버는 글로벌 모델의 특징 $F_g$의 이동 평균(Moving Average) $\hat{F}_g$를 계산하여, 합성 데이터의 특징 $F^{sy}$에서 도구 표현 $I^{sy}$를 분리해낸다.
  - 사전 학습 모델 $v_{pre}$는 다음 손실 함수를 통해 업데이트된다:
    $$L_{pre} = L_{Seg} + \lambda_2 \cdot \frac{1}{N_e} \| F^{sy} - D^{sy} - \hat{F}_g \|_2^2$$
- **로컬 수렴 동기화 (Synchronizing Local Convergence):** 추출된 도구 표현 $I^{sy} = F^{sy} - D^{sy}$를 사용하여 글로벌 모델 $v_g$를 업데이트함으로써, 다양한 사이트의 모델들이 일관된 도구 표현으로 수렴하도록 유도한다:
    $$L_g = L_{Seg} + \lambda_3 \frac{1}{N_e} \| F^g - (F^{sy} - D^{sy}) \|_2^2$$

## 📊 Results

### 실험 설정

- **데이터셋:** 5개의 사이트(Site A~E)로 구성된 벤치마크를 구축하였다. Site A~D는 연합 학습에 참여하고, Site E(Lobectomy)는 모델의 일반화 성능을 측정하기 위한 Out-of-federation 테스트 셋으로 사용되었다. 또한 서버 측에서는 합성 데이터셋인 Sisvse를 활용하였다.
- **평가 지표:** Dice, IoU, HD95, ASSD를 사용하였다.
- **비교 대상:** Local Train, Upper Bound(중앙 집중화), FedAvg, FedRep, FedDP, FedLD, FedAS.

### 주요 결과

- **내부 검증 (Site A~D):** FedST는 모든 지표에서 SOTA FL 방법들을 능가하였다. 특히 데이터가 부족한 Site D에서 중앙 집중화 방식(Upper Bound)보다 높은 성능을 보이기도 하였다.
- **외부 검증 (Site E):** 가장 주목할만한 결과로, 연합 학습에 참여하지 않은 Site E에서 기존 방법들은 거의 작동하지 않았으나, FedST는 매우 높은 일반화 성능을 보였다 (IoU 기준 45.29% 향상).
- **이진 분할 일반화 (GraSP 데이터셋):** 전혀 다른 데이터셋인 GraSP에서도 FedST는 90.64%의 Dice score를 기록하며 압도적인 일반화 능력을 입증하였다.
- **효율성:** RSC를 통한 파라미터 부분 공유 덕분에 통신 비용이 FedAvg(19.71MB)보다 낮은 12.69MB로 감소하였으며, 23.04 FPS로 실시간 추론이 가능함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문의 가장 큰 성과는 **수술 도구의 시공간적 일관성**이라는 도메인 특성을 PFL 프레임워크에 성공적으로 녹여냈다는 점이다. 특히 Query Embedding을 개인화하여 배경 표현을 분리하고, temporal modeling을 통해 도구의 움직임을 캡처한 것이 Out-of-federation 사이트에서의 비약적인 성능 향상으로 이어졌다. 이는 수술 환경이 바뀌어도 도구의 형태와 움직임은 일정하다는 가설이 유효함을 증명한다.

또한 SERQ 모듈을 통해 합성 데이터의 도메인 갭 문제를 해결하고, 이를 글로벌 모델의 가이드로 활용함으로써 로컬 모델들의 수렴을 안정화시킨 점이 기술적으로 매우 영리한 접근이다.

### 한계 및 향후 과제

- **시각적 극한 상황:** 반사(Specular reflection), 모션 블러, 조직에 의한 가려짐(Occlusion)이 심한 프레임에서는 성능이 저하되는 문제가 관찰되었다. 저자는 이를 해결하기 위해 저수준 강화 모듈이나 불확실성 인식 학습 전략이 필요함을 언급한다.
- **작업 확장성:** 현재는 분할(Segmentation) 작업에 한정되어 있으며, 수술 워크플로우 인식이나 깊이 추정(Depth estimation)과 같은 타 작업으로의 확장이 필요하다.

## 📌 TL;DR

본 논문은 수술 비디오의 배경 다양성과 도구 유사성을 이용한 개인화 연합 학습 프레임워크 **FedST**를 제안한다. **RSC**를 통해 배경 표현(Query Embedding)과 도구 표현(Spatio-temporal features)을 분리하고, **SERQ**를 통해 합성 데이터의 도구 표현을 글로벌 모델에 주입하여 일반화 성능을 극대화하였다. 실험 결과, 학습에 참여하지 않은 외부 사이트에서도 압도적인 분할 성능을 보였으며, 이는 수술 도메인의 시공간적 특성을 활용한 표현 분리가 실제 임상 환경의 데이터 이질성 문제를 해결하는 핵심 열쇠임을 시사한다.
