# WeakSurg: Weakly supervised surgical instrument segmentation using temporal equivariance and semantic continuity

Qiyuan Wang, Yanzhe Liu, Shang Zhao, Rong liu and S. Kevin Zhou (2024)

## 🧩 Problem to Solve

본 논문은 로봇 수술 비디오에서 수술 도구를 분할(Surgical Instrument Segmentation, SIS)하는 문제를 다룬다. 일반적으로 정교한 분할 모델을 학습시키기 위해서는 픽셀 단위의 정밀한 어노테이션(pixel-wise annotation)이 필요하지만, 이는 임상적으로 막대한 비용과 시간이 소요되는 작업이다.

반면, 수술 비디오 스트림과 함께 기록되는 도구의 존재 여부(instrument presence labels)는 OCR 등의 방법을 통해 저비용으로 획득할 수 있다. 그러나 이미지 수준의 레이블(image-level labels)만 사용하는 약지도 학습(Weakly Supervised Learning) 방식은 제약 조건이 매우 부족하여(highly under-constrained), 픽셀 단위의 정확한 경계를 찾아내는 것이 매우 어렵다는 한계가 있다.

따라서 본 연구의 목표는 수술 비디오가 가진 고유한 **시간적 특성(Temporal properties)**을 활용하여, 오직 도구의 존재 여부 레이블만을 사용하여 정밀한 수술 도구 분할을 수행하는 **WeakSurg** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 정지 영상 기반의 약지도 분할 패러다임을 확장하여, 비디오의 시간적 연속성을 통해 학습의 제약 조건을 강화하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Temporal Equivariance Constraint (TER):** 인접한 프레임 간의 픽셀 수준 시간적 일관성을 강화하여 특징 표현의 안정성을 높인다.
2. **Class-aware Semantic Continuity Constraint (CSCR):** 전역 뷰와 지역 뷰 사이의 클래스 기반 의미적 연속성을 강제함으로써, 기존 방식에서 놓치기 쉬운 비변별적(non-discriminative) 영역까지 활성화한다.
3. **Temporal-enhanced Pseudo Masks Generation:** 단일 프레임이 아닌 연속된 프레임 클립을 활용하여 배경 노이즈를 억제하고 전경 정보를 강조한 의사 마스크(pseudo masks)를 생성하며, 이를 SAM(Segment Anything Model)과 결합하여 정밀도를 높인다.

## 📎 Related Works

기존의 수술 도구 분할 연구는 크게 세 가지 방향으로 진행되었다.

- **완전 지도 학습(Fully Supervised):** 높은 정확도를 보이지만 픽셀 단위 어노테이션 비용이 매우 높다.
- **비지도/준지도 학습(Unsupervised/Semi-supervised):** 핸드크래프트 큐(color, location)를 사용하거나, 시뮬레이션 데이터의 도메인 적응(Domain Adaptation)을 시도하고, 일부 레이블이 있는 프레임의 예측치를 없는 프레임으로 전파하는 방식을 사용한다.
- **약지도 학습(Weakly Supervised):** 스크리블(scribble)이나 바운딩 박스(box) 수준의 레이블을 사용하여 비용을 줄이려 했다. 최근에는 도구의 존재 여부(presence labels)만 사용하는 연구가 등장했으나, 여전히 픽셀 및 지역 수준의 관계를 학습하는 데 한계가 있었다.

본 논문은 기존의 약지도 분할 모델이 정지 영상 위주로 설계되었다는 점에 주목하고, 비디오 데이터의 시간적 의존성을 명시적인 제약 조건으로 도입함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조

WeakSurg는 기본적으로 MCT(Multi-Class Token Transformer)라는 2단계 약지도 의미론적 분할(WSSS) 방법을 기반으로 한다. 1단계에서는 존재 여부 레이블을 통해 클래스별 로컬라이제이션 맵(CAM)을 생성하고, 2단계에서는 이 CAM으로 만든 의사 마스크를 이용해 최종 분할 모델을 학습시킨다. 본 논문은 이 과정에 시간적 제약 조건(TER, CSCR)을 추가하였다.

### 2. Multi-Class Token Transformer (MCT)

MCT는 단일 클래스 토큰 대신 여러 개의 클래스 토큰을 사용하여 각 클래스별 특성을 학습한다.

- **입력:** 시간적 쌍(temporal pair) $(I_t, I_T)$가 입력된다.
- **출력:** 클래스 토큰 $(z_t^c, z_T^c)$와 패치 토큰 $(z_t^p, z_T^p)$이 생성된다.
- **손실 함수:** 클래스 예측값과 실제 이미지 수준 레이블 $\hat{y}$ 사이의 Multi-label soft margin loss를 사용한다.

$$L_{CLS} = \psi(y_t^c, \hat{y}_t) + \psi(y_t^p, \hat{y}_t) + \psi(y_T^c, \hat{y}_T) + \psi(y_T^p, \hat{y}_T)$$
$$\psi(y, \hat{y}) = -\frac{1}{C} \sum_{i=1}^{C} \{ \hat{y}_i \log \sigma(y_i) + (1-\hat{y}_i) \log(1-\sigma(y_i)) \}$$

### 3. Temporal Equivariance Constraint (TER)

TER는 시간 $t$의 특징을 시간 $T$로 변환했을 때, 실제 시간 $T$에서 관측된 특징과 일치해야 한다는 제약 조건이다.

- **Class-Aware Projection:** 패치 토큰들을 클래스 토큰(semantic prototypes)을 기준으로 클러스터링하여 의미적 프로토타입 $S_t, S_T$를 생성한다.
- **Temporal Propagator:** 국소 공간 유사도 $Q$를 계산하여 $S_t$를 $\hat{S}_T$로 워핑(warping)한다.
  $$Q = [\Theta(z_{t,i}^p, z_{T,j}^p)]_{hw \times hw}$$
  $$\hat{S}_T(j) = \exp(N(Q_{ij})) S_t(i), \quad i \in \{1, \dots, k\}$$
- **TER Loss:** 변환된 특징 $\hat{S}_T$와 실제 특징 $S_T$ 사이의 교차 엔트로피를 최소화한다.
  $$L_{TER} = -\sum_{i,j} g(\hat{S}_T) \log(S_T)$$

### 4. Class-aware Semantic Continuity Constraint (CSCR)

특정 영역이 프레임마다 다르게 인식되는 불일치 문제를 해결하기 위해, 전역 뷰와 지역 뷰 사이의 의미적 연속성을 강제한다.

- **Local View Generation:** 생성된 CAM을 기반으로 불확실성이 높은 지역을 크롭(crop)하여 로컬 뷰를 생성한다.
- **Contrastive Learning:** 전역 참조 뷰의 클래스 토큰 $x_{g,t}$와 로컬 타겟 뷰의 클래스 토큰 $x_{l,T}$ 사이의 거리(cosine similarity)를 이용한 대조 학습을 수행한다.
  $$L_{CSCR} = \sum_{i=1}^{B_l} \sum_{j=1}^{B_g} I_{ij} \log \frac{\exp(x_{l,T}^i \cdot x_{g,t}^j / \tau)}{\sum_{j'=1}^{B_g} I_{ij'} \exp(x_{l,T}^i \cdot x_{g,t}^{j'} / \tau)}$$

최종 학습 손실 함수는 다음과 같다.
$$L_{overall} = L_{CLS} + L_{TER} + L_{CSCR}$$

### 5. Temporal-enhanced Pseudo Masks Generation

학습 후, 단일 프레임 CAM 대신 인접 프레임들의 CAM을 통합하여 노이즈를 제거한 시간 강화 CAM $M$을 생성한다.
$$M = \frac{1}{2\delta_t+1} \left( M + \sum_{t'=(T-\delta_t)}^{T+\delta_t} \Phi((z_{t'}^p, z_T^p), M_{t'}) \right)$$
이후 임계값 처리로 거친 마스크를 만들고, 이를 기반으로 바운딩 박스 프롬프트를 생성하여 **SAM(Segment Anything Model)**에 입력함으로써 정밀한 의사 마스크를 얻는다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Cholec80(담낭 절제술), RLLS(간 절제술) 비디오 데이터셋을 사용하였다.
- **지표:** 의미론적 분할(SS)을 위해 $\text{IoU}_C, \text{IoU}_S, \text{IoU}_{mc}$를 사용하였으며, 인스턴스 분할(IS)을 위해 $\text{AP}_{50}, \text{AP}_{75}, \text{AP}$를 사용하였다.
- **구현:** 1단계에서는 존재 여부 레이블로 학습하고, 2단계에서는 생성된 의사 마스크를 이용해 Mask2Former를 학습시켜 평가하였다.

### 2. 정량적 결과

Table I에 따르면, WeakSurg는 모든 데이터셋에서 기존 SOTA 방식들을 상회하는 성능을 보였다.

- **Cholec80:** MCT 대비 $\text{IoU}_C$에서 7%, $\text{IoU}_S$에서 8%, $\text{IoU}_{mc}$에서 24%라는 큰 폭의 향상을 이루었다. 인스턴스 분할에서도 MCST 대비 $\text{AP}_{50}$이 7%p 상승하였다.
- **RLLS:** 의미론적 분할 $\text{IoU}_C$에서 약 2.9% 향상되었으며, 특히 인스턴스 분할 $\text{AP}_{75}$에서 15.4%p라는 괄목할 만한 성능 향상을 보였다.

### 3. 절제 연구 (Ablation Study)

Table II 분석 결과, 각 구성 요소의 기여도는 다음과 같다.

- **TER:** 가장 큰 성능 향상을 가져왔으며, 특히 인스턴스 분할 $\text{AP}_{50}$을 크게 끌어올렸다.
- **CSCR:** $\text{IoU}_{mc}$ 지표에서 특히 유의미한 향상을 보이며, 비변별적 영역의 활성화를 도왔음을 입증하였다.
- **TMG:** 최종 분할 성능을 한 단계 더 높였으며, 배경 영역의 노이즈를 효과적으로 억제하였다.

## 🧠 Insights & Discussion

본 연구는 수술 비디오라는 특수한 도메인에서 **시간적 연속성**이 약지도 학습의 부족한 제약 조건을 보완할 수 있는 강력한 도구가 될 수 있음을 증명하였다.

특히, 단순한 프레임 간 복사가 아니라 **Temporal Equivariance**와 **Semantic Continuity**라는 두 가지 관점에서 제약을 가한 점이 인상적이다. 픽셀 수준의 일관성(TER)과 지역-전역 수준의 의미적 일관성(CSCR)을 동시에 고려함으로써, 약지도 학습의 고질적인 문제인 '변별적인 부분만 활성화되는 현상'을 극복하고 도구의 전체 형태를 더 잘 복원할 수 있게 되었다.

또한, 최신 기반 모델인 SAM을 의사 마스크 정교화 단계에 도입하여, 약지도 학습으로 생성된 거친 CAM을 고품질의 마스크로 변환한 전략이 최종 성능 향상에 크게 기여한 것으로 보인다. 다만, SAM을 통한 정교화 과정이 추론 속도나 실시간 적용 가능성에 어떤 영향을 미치는지에 대한 분석은 본문에 명시되지 않았다.

## 📌 TL;DR

본 논문은 오직 수술 도구의 **존재 여부 레이블(presence labels)**만을 사용하여 도구를 분할하는 약지도 학습 프레임워크 **WeakSurg**를 제안한다. 비디오의 시간적 특성을 활용한 **TER(시간적 동등성 제약)**와 **CSCR(클래스 인식 의미적 연속성 제약)**을 도입하여 학습의 제약을 강화하였고, 시간적으로 강화된 의사 마스크를 SAM과 결합하여 정밀도를 극대화하였다. 실험 결과, Cholec80 및 RLLS 데이터셋에서 기존 SOTA 모델들을 뛰어넘는 성능을 달성하였으며, 이는 수술 비디오 분석에서 시간적 정보의 중요성을 시사한다.
