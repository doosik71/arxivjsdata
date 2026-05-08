# Unsupervised Multiple Person Tracking using AutoEncoder-Based Lifted Multicuts

Kalun Ho, Janis Keuper, Margret Keuper (2020)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전의 고전적 과제인 다중 객체 추적(Multiple Object Tracking, MOT), 특히 다중 보행자 추적 문제를 다룬다. 현재의 MOT 접근 방식은 대부분 '추적 기반 탐지(Tracking-by-Detection)' 패러다임을 따르며, 탐지된 객체들을 시공간적으로 연결하여 궤적을 생성한다.

이 과정에서 핵심은 서로 다른 프레임에서 탐지된 객체가 동일 인물인지 판별하는 재식별(Re-identification, ReID) 작업이다. 기존의 성공적인 방법들은 Siamese Network와 같은 딥러닝 모델을 사용하여 외형 특징(Appearance features)을 학습시키지만, 이는 대량의 정답 라벨(Annotation)이 포함된 지도 학습(Supervised learning) 데이터를 필요로 한다. 하지만 실제 환경에서 매번 특정 시나리오에 맞는 대규모 라벨링 데이터를 구축하는 것은 비용이 매우 많이 들며, 이는 실용적인 적용에 큰 제약이 된다.

따라서 본 논문의 목표는 인간의 개입(Human annotation) 없이, 오직 영상 데이터 자체에서 추출한 시공간적 단서와 자기 지도 학습(Self-supervision)만을 이용하여 강건한 외형 특징을 학습하고, 이를 통해 보행자를 추적하는 완전히 비지도(Unsupervised) 방식의 MOT 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"단순한 시공간적 연결성(Spatio-temporal cues)을 통해 얻은 가짜 라벨(Pseudo-labels)을 활용하여, 외형의 변화에 강건한 잠재 공간(Latent space)을 학습시키는 것"**이다.

구체적인 설계 직관은 다음과 같다. 영상의 인접한 프레임 사이에서는 Bounding Box의 겹침 정도(IoU)와 같은 단순한 시공간적 정보만으로도 동일 인물일 확률이 매우 높다. 이러한 단서를 이용해 일시적인 궤적(Tracklets)을 먼저 생성하고, 동일한 궤적에 속한 탐지 결과들이 잠재 공간 상에서 서로 가깝게 위치하도록 Convolutional AutoEncoder를 학습시킨다. 이렇게 학습된 잠재 공간은 인물의 포즈 변화나 관점 변화와 같은 외형 변동성을 내포하게 되며, 결과적으로 시공간적 정보만으로는 연결할 수 없는 장기적인 폐쇄(Long-range occlusion) 상황에서도 외형 기반의 재식별을 가능하게 한다.

## 📎 Related Works

기존의 MOT 연구들은 탐지된 결과들을 그래프의 노드로 설정하고 이를 연결하는 조합 최적화 문제로 접근해 왔다. Integer Linear Programming, MAP estimation, CRF 등 다양한 최적화 기법이 사용되었으며, 계산 비용을 줄이기 위해 탐지 결과들을 먼저 Tracklet으로 그룹화하는 전처리가 흔히 사용되었다.

최근에는 딥러닝 기반의 외형 특징 추출기가 도입되었으며, 특히 Tang 등이 제안한 Multicut 및 Lifted Multicut 방식은 그래프 파티셔닝 문제를 통해 최적의 궤적을 찾는 효율적인 구조를 제시하였다. 그러나 이러한 최신 기법들은 대부분 ReID 네트워크를 학습시키기 위해 대규모의 지도 학습 데이터를 필요로 한다는 한계가 있다. 본 논문은 이러한 지도 학습의 의존성을 제거하고, 시공간적 단서와 AutoEncoder를 결합하여 비지도 방식으로 외형 특징을 학습한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

본 논문에서 제안하는 파이프라인은 크게 세 단계로 구성된다: 1) 시공간적 그룹화를 통한 초기 라벨 생성, 2) AutoEncoder를 이용한 외형 특징 학습, 3) Lifted Multicut을 이용한 최종 추적.

### 1. Multicut Formulation 및 초기 Tracklet 생성

본 연구는 MOT 문제를 그래프 파티셔닝 문제인 최소 비용 멀티컷(Minimum Cost Multicut) 문제로 정의한다.

- **그래프 정의**: 무방향 그래프 $G = (V, E)$에서 노드 $v \in V$는 객체 탐지 결과(Bounding Box)를 나타내고, 엣지 $e \in E$는 시공간적 연결성을 나타낸다. 각 엣지에는 실수 값의 비용 $c_e$가 할당된다.
- **목적 함수**: 엣지 라벨 $y_e \in \{0, 1\}$를 결정하여 다음의 비용을 최소화한다.
  $$\min_{y \in \{0, 1\}} \sum_{e \in E} c_e y_e$$
  이때, 사이클 불평등식(Cycle inequalities) 제약 조건이 추가되어 그래프가 일관되게 분할되도록 보장한다.
- **초기 그룹화**: 지도 학습 없이 초기 궤적을 만들기 위해, DeepMatching을 통해 인접 프레임 간의 일치 쌍을 찾고, 이들의 $IoU^{DM}$이 $0.7$보다 클 경우 동일 인물로 간주하여 전-그룹화(Pre-grouping)를 수행한다.

### 2. Deep Convolutional AutoEncoder

단순한 시공간적 그룹화 정보를 이용하여 외형 특징을 학습하는 Convolutional AutoEncoder(CAE)를 구축한다.

- **구조**: Encoder $f_\theta(\cdot)$와 Decoder $g_\phi(\cdot)$로 구성된다. Encoder는 5개의 Convolution 및 Max-pooling 레이어를 통해 입력을 32차원의 잠재 벡터(Latent vector) $z_i$로 압축하고, Decoder는 이를 다시 원래 이미지로 복원한다.
- **손실 함수**: 단순한 복원 손실(Reconstruction loss)에 시공간적 클러스터 정보를 반영한 클러스터링 손실(Clustering loss)을 추가한다.
  $$\min_{\theta, \phi} \sum_{i=1}^{N} \lambda L(g(f(x_i)), x_i) + (1-\lambda) L(f(x_i), \tilde{c}_i)$$
  여기서 $L$은 최소 제곱 오차(Least-squared loss)이며, $\tilde{c}_i$는 해당 탐지 결과 $x_i$가 속한 시공간적 클러스터의 잠재 특징 평균(Centroid)이다. $\lambda$는 두 손실 사이의 균형을 조절하는 파라미터이다.

### 3. AutoEncoder 기반 Affinity Measure 및 최종 추적

학습된 CAE를 사용하여 두 탐지 결과 $x_i, x_j$ 사이의 유사도를 유클리드 거리로 측정한다.
$$d_{i,j} = \|f(x_i) - f(x_j)\|$$

- **Lifted Multicut**: 장기적인 연결(Long-range connection)을 처리하기 위해 Lifted Multicut 프레임워크를 사용한다. 이는 기존 엣지 $E$ 외에 추가적인 Lifted 엣지 $F$를 도입하여, 시공간적으로 멀리 떨어진 노드들 사이에도 직접적인 연결 비용을 부여할 수 있게 한다.
- **절단 확률(Cut Probability) 추정**: $IoU^{DM}$과 잠재 공간 거리 $d_{AE}$를 특징 벡터로 사용하여 로지스틱 회귀(Logistic regression)를 통해 엣지의 절단 확률 $p_e$를 계산한다.
  $$p_e = \frac{1}{1 + \exp(-\langle \beta, f(e) \rangle)}$$
  최종적으로 이 확률을 Logit 함수로 변환하여 Multicut의 비용 $c_e$로 사용함으로써 최적의 궤적을 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MOT17 벤치마크 (14개 시퀀스, SDP/DPM/FRCNN 탐지기 사용).
- **평가 지표**: MOTA (Tracking Accuracy), MOTP (Precision), IDs (Identity Switches), MT (Mostly Tracked), ML (Mostly Lost) 등을 사용한다.
- **비교 대상**: LSST17, Tracktor17, JBNOT, FAMNet, eTC17 등 최신 지도 학습 기반 모델들과 비교하였다.

### 주요 결과

- **Ablation Study**:
  - 단순히 $IoU^{DM}$만 사용했을 때보다 $d_{AE}$ (AutoEncoder 거리)를 추가했을 때 성능이 향상되었다.
  - 특히, 클러스터링 손실을 추가하여 학습한 $d_{AE+C}$가 일반 $d_{AE}$보다 훨씬 높은 MOTA를 기록했다.
  - Lifted Multicut을 통해 장기 연결을 허용했을 때 MOTA가 $49.9\%$까지 상승하며 최적의 성능을 보였다.
- **최종 성능**: 테스트 데이터셋에서 본 모델은 **MOTA 48.1%**를 달성하였다.
- **비교 분석**: 지도 학습 기반의 SOTA 모델들(MOTA $51 \sim 54\%$)보다는 약간 낮지만, 어떤 형태의 정답 라벨이나 사전 학습된 모델 없이 달성한 결과라는 점에서 매우 경쟁력 있는 수치이다.

## 🧠 Insights & Discussion

### 강점

본 연구는 MOT 분야에서 가장 큰 병목 중 하나인 '데이터 라벨링 의존성'을 완전히 제거하였다. 시공간적 단서 $\rightarrow$ 비지도 특징 학습 $\rightarrow$ 그래프 최적화로 이어지는 파이프라인을 통해, 추가 비용 없이 영상 데이터만으로 강건한 ReID 특징을 얻을 수 있음을 입증하였다.

### 한계 및 해석

- **성능 격차**: 지도 학습 모델과의 성능 차이는 당연한 결과로 해석된다. 정답 라벨은 정교한 정체성을 제공하지만, 본 모델이 사용하는 시공간적 단서는 '가짜 라벨'이므로 어느 정도의 노이즈가 포함될 수밖에 없다.
- **탐지기 의존성**: 실험 결과 SDP 탐지기에서는 높은 성능을 보였으나, 노이즈가 많은 DPM 탐지기에서는 성능이 낮아졌다. 이는 비지도 학습 기반의 특징 추출기가 입력 데이터(탐지 결과)의 품질에 민감하게 반응함을 시사한다.
- **특정 시나리오 취약성**: MOT17-08 시퀀스에서 모든 탐지기에 대해 매우 낮은 성능(MOTA $30\%$ 미만)을 보였는데, 이는 해당 환경의 특수성(예: 극심한 혼잡도 또는 조명 변화)이 비지도 학습으로 극복하기 어려운 수준이었음을 의미한다.

## 📌 TL;DR

본 논문은 **인간의 라벨링 없이** 보행자를 추적하는 비지도 학습 기반의 MOT 프레임워크를 제안한다. **단순 시공간 정보(IoU)로 생성한 임시 궤적을 가이드 삼아 Convolutional AutoEncoder를 학습**시켜 외형 특징을 추출하고, 이를 **Lifted Multicut 그래프 최적화**에 적용하여 장거리 재식별 문제를 해결하였다. MOT17 벤치마크에서 지도 학습 모델에 근접한 경쟁력 있는 성능(MOTA 48.1%)을 달성함으로써, 실용적인 비지도 추적 시스템의 가능성을 제시하였다.
