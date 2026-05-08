# WeakSurg: Weakly supervised surgical instrument segmentation using temporal equivariance and semantic continuity

Qiyuan Wang, Yanzhe Liu, Shang Zhao, Rong liu and S. Kevin Zhou (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 로봇 수술 비디오에서 수술 도구 분할(Surgical Instrument Segmentation, SIS)을 수행할 때 발생하는 막대한 수동 어노테이션 비용의 문제이다. 일반적으로 SIS는 픽셀 단위의 정밀한 마스크(pixel-wise mask)가 필요한 Fully Supervised Learning 방식으로 학습되지만, 수술 영상의 특성상 이러한 데이터를 대량으로 확보하는 것은 매우 어렵고 비용이 많이 든다.

수술 비디오 스트림에는 도구의 존재 여부(instrument presence)에 대한 라벨이 이미 기록되어 있는 경우가 많아 이를 활용할 잠재력이 크다. 하지만 이미지 레벨의 존재 여부 라벨만 사용하는 Weakly Supervised Segmentation은 제약 조건이 매우 부족하여(under-constrained), 도구의 정확한 경계를 찾아내는 것이 매우 어렵다는 도전 과제가 있다. 따라서 본 연구의 목표는 오직 도구의 존재 여부 라벨만을 활용하여, 수술 비디오의 시간적 특성(temporal properties)을 이용해 정밀한 도구 분할을 수행하는 WeakSurg 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 수술 비디오가 가진 시간적 연속성과 의존성을 활용하여, 부족한 픽셀 단위 감독 신호를 보완하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Weakly Supervised Architecture 제안**: 오직 도구의 존재 여부 라벨만을 사용하여 수술 도구를 분할하는 프레임워크를 제시하여 수동 어노테이션 비용을 획기적으로 줄였다.
2. **시간적 제약 조건 도입**: 픽셀 단위의 일관성을 유지하기 위한 Temporal Equivariance Constraint(TER)와 지역-전역 간의 의미적 일관성을 강화하는 Class-aware Semantic Continuity Constraint(CSCR)를 도입하였다.
3. **시간적 강화 의사 마스크(Temporal-enhanced Pseudo Masks) 생성**: 단일 프레임의 CAM(Class Activation Map) 대신 인접 프레임들의 정보를 통합한 의사 마스크를 생성하여 배경 노이즈를 억제하고 전경 정보를 강조하였다.

## 📎 Related Works

기존의 수술 도구 분할 연구는 크게 다음과 같은 흐름으로 발전해 왔다.

- **Fully Supervised Methods**: U-Net 기반의 딥러닝 모델들이 주를 이루나, 정밀한 픽셀 단위 라벨이 필수적이라는 한계가 있다.
- **Unsupervised/Semi-supervised Methods**: 색상, 객체성(objectness) 등의 수작업 큐를 이용하거나, 시뮬레이션 데이터를 통한 도메인 적응(Domain Adaptation), 또는 일부 라벨링된 데이터와 라벨링되지 않은 데이터 간의 일관성을 이용하는 방식이 제안되었다.
- **Weakly Supervised Methods**: 스크리블(Scribble) 수준의 라벨이나 바운딩 박스(Bounding Box)를 사용하는 방식이 연구되었으며, 최근에는 오직 도구의 존재 여부 라벨만을 이용한 위치 추정이나 탐지 연구가 진행되었다.

그러나 기존의 Weakly Supervised 방식들은 수술 도메인에서 픽셀 및 지역 수준의 관계를 충분히 활용하지 못해 분할 성능이 낮다는 한계가 있었으며, 본 논문은 이를 시간적 특성 활용을 통해 해결하고자 한다.

## 🛠️ Methodology

WeakSurg는 기본적으로 이미지 기반의 2단계 WSSS(Weakly Supervised Semantic Segmentation) 방법인 MCT(Multi-Class Token Transformer)를 기반으로 하며, 여기에 수술 비디오의 시간적 특성을 반영한 제약 조건들을 추가하였다.

### 1. Multi-Class Token Transformer (MCT)

MCT는 단일 클래스 토큰 대신 여러 개의 클래스 토큰을 사용하여, 각 클래스 토큰과 패치 토큰 간의 Attention을 통해 클래스별 위치를 특정한다. 입력으로 단일 프레임이 아닌 시간적 쌍(temporal pair) $(I_t, I_T)$를 입력받아 출력 클래스 토큰 $(z^t_c, z^T_c)$와 패치 토큰 $(z^t_p, z^T_p)$을 생성한다. 학습 시에는 다음과 같은 Multi-label soft margin loss를 사용한다.

$$\psi(y, \hat{y}) = -\frac{1}{C} \sum_{i=1}^{C} \{\hat{y}_i \log \sigma(y_i) + (1-\hat{y}_i) \log(1-\sigma(y_i))\}$$
$$L_{CLS} = \psi(y^t_c, \hat{y}^t) + \psi(y^t_p, \hat{y}^t) + \psi(y^T_c, \hat{y}^T) + \psi(y^T_p, \hat{y}^T)$$

### 2. Temporal Equivariance Constraint (TER)

TER는 인접한 프레임 간의 픽셀 단위 시간적 일관성을 강화한다. 기준 프레임 $t$의 특징을 대상 프레임 $T$로 변환했을 때, 현재 관찰되는 $T$ 프레임의 특징과 일치해야 한다는 원리이다.

- **Class-Aware Projection**: 최적 운송 클러스터링(Optimal-transport clustering) 알고리즘을 통해 패치 토큰들을 의미적 프로토타입에 따라 클러스터링하여 투영 결과 $(S_t, S_T)$를 얻는다.
- **Temporal Propagator**: 국소 공간 유사도 $Q$를 계산하여 $S_t$를 $\hat{S}_T$로 전파(warp)시킨다. 유사도 $Q$는 다음과 같이 계산된다.
$$Q = [\Theta(z^t_{p,i}, z^T_{p,j})]_{hw \times hw}$$
- **Loss Function**: 전파된 결과 $\hat{S}_T$와 실제 $S_T$ 사이의 교차 엔트로피를 통해 제약을 가한다.
$$L_{TER} = -\frac{1}{hw} \sum_{i,j} g(\hat{S}_T) \log(S_T)$$

### 3. Class-aware Semantic Continuity Constraint (CSCR)

수술 영상에서는 도구의 일부 지역이 배경과 유사하여 인식 일관성이 떨어지는 경우가 많다. CSCR은 전역 뷰(Global view)와 지역 뷰(Local view) 간의 의미적 유사성을 강화하여 비변별적 지역(non-discriminative regions)의 활성화를 돕는다.

- **Local View Generation**: CAM을 기반으로 불확실성이 높은 지역을 샘플링하여 $L$개의 지역 크롭(Local crops)을 생성한다.
- **Temporal Class Tokens Contrast**: 전역 기준 프레임의 클래스 토큰 $x_{g,t}$와 지역 대상 프레임의 클래스 토큰 $x_{l,T}$ 사이의 차이를 최소화하는 멀티 라벨 대조 학습(Multi-label contrastive learning)을 수행한다.
$$L_{CSCR} = \sum_{i=1}^{B_l} \sum_{j=1}^{B_g} I_{ij} \log \frac{\exp(x_{l,T}^i \cdot x_{g,t}^j / \tau)}{\sum_{j'=1}^{B_g} I_{ij'} \exp(x_{l,T}^i \cdot x_{g,t}^{j'} / \tau)}$$

최종 학습 손실 함수는 다음과 같다.
$$L_{overall} = L_{CLS} + L_{TER} + L_{CSCR}$$

### 4. Temporal-enhanced Pseudo Masks Generation

학습 후, 단일 프레임 CAM 대신 연속된 프레임들의 정보를 통합한 Temporal-enhanced CAM($M$)을 생성하여 배경 노이즈를 억제한다.
$$M = \frac{1}{2\delta_t+1} (M + \sum_{t'=(T-\delta_t)}^{T+\delta_t} \Phi((z^t_p, z^T_p), M_{t'}))$$
이후 이 CAM을 임계값 처리하여 거친(coarse) 의사 마스크를 만들고, 이를 Bounding Box 프롬프트로 변환하여 SAM(Segment Anything Model)에 입력함으로써 정밀한 최종 의사 마스크를 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80(담낭 절제술), RLLS(로봇 간 절제술) 두 가지 데이터셋을 사용하였다. 존재 여부 라벨은 OCR 등을 통해 획득하였으며, 평가를 위해 숙련된 임상의가 확인한 픽셀 단위 마스크를 별도로 제작하였다.
- **평가 지표**: Semantic Segmentation의 경우 $IoU_C, IoU_S, IoU_{mc}$를 사용하였고, Instance Segmentation의 경우 $AP_{50}, AP_{75}, AP$를 사용하였다.
- **비교 대상**: WLSTM, ToCo, MCT, MCST, MCT+ 등 최신 WSSS 및 수술 도구 분할 방법론과 비교하였다.

### 주요 결과

- **Semantic Segmentation**: WeakSurg는 모든 데이터셋에서 SOTA 성능을 달성하였다. 특히 Cholec80에서 MCT 대비 $IoU_C$는 7%, $IoU_{mc}$는 24% 향상되었다.
- **Instance Segmentation**: 인스턴스 분할에서도 우수한 성능을 보였으며, MCST 대비 $AP_{50}$에서 7% 이상의 향상을 기록하였다.
- **Ablation Study**:
  - TER 도입 시 $IoU_C$가 5.2%, $AP_{50}$이 20.4% 크게 상승하여 시간적 일관성 제약의 중요성이 입증되었다.
  - CSCR은 특히 $IoU_{mc}$ 지표를 크게 개선하였다.
  - TMG(시간적 강화 의사 마스크)는 최종 분할 성능을 추가로 끌어올려 최대 $IoU_C$ 82.08%를 달성하였다.

## 🧠 Insights & Discussion

본 논문은 수술 비디오의 시간적 특성을 학습 단계(TER, CSCR)와 마스크 생성 단계(TMG) 모두에 적용함으로써, 이미지 레벨의 약한 감독 신호만으로도 높은 수준의 분할 성능을 낼 수 있음을 보여주었다. 특히, 단순한 프레임 간 복사가 아니라 semantic prototype 기반의 클러스터링과 대조 학습을 통해 도구의 비변별적 지역까지 효과적으로 활성화시킨 점이 돋보인다.

**한계 및 논의사항:**

- **SAM 의존성**: 최종 정밀 마스크 생성을 위해 SAM을 사용하는데, 이는 SAM이 매우 강력한 모델이기 때문이기도 하지만, 실시간 추론 환경에서는 SAM의 연산 비용이 부담이 될 수 있다.
- **의사 마스크의 품질**: TMG가 배경을 억제하지만, 여전히 의사 마스크(Pseudo mask)에 기반하여 최종 모델을 학습시키므로, 초기 CAM의 품질이 낮을 경우 전파되는 오류(error propagation)가 발생할 가능성이 있다.

## 📌 TL;DR

본 연구는 수술 비디오에서 **오직 도구의 존재 여부 라벨만으로 도구를 분할**하는 **WeakSurg** 프레임워크를 제안한다. 시간적 등가성 제약(TER)과 클래스 인지 의미적 연속성 제약(CSCR)을 통해 모델의 학습을 안정화하고, 인접 프레임 정보를 통합한 시간적 강화 의사 마스크를 통해 SAM으로 정밀한 경계를 추출한다. 이 방법은 Cholec80 및 RLLS 데이터셋에서 기존 SOTA 모델들을 뛰어넘는 성능을 보였으며, 수술 영상 분석에서 어노테이션 비용을 획기적으로 줄일 수 있는 가능성을 제시하였다.
