# Visual Tracking by TridentAlign and Context Embedding

Janghoon Choi, Junseok Kwon, and Kyoung Mu Lee (2020)

## 🧩 Problem to Solve

본 논문은 Siamese 네트워크 기반의 비주얼 트래킹(Visual Tracking)에서 발생하는 두 가지 핵심적인 문제를 해결하고자 한다. 첫째는 대상 객체의 극심한 크기 변화(Scale Variation)와 변형(Deformation)이며, 둘째는 타겟과 유사한 외형을 가진 방해 객체(Distractor Objects)로 인한 오탐지 문제이다.

기존의 많은 Siamese 트래커들은 모션의 부드러움(Motion Smoothness)과 점진적인 크기 변화를 가정하고 설계되어 단기 트래킹 시나리오에 최적화되어 있다. 이를 위해 타겟 주변의 좁은 탐색 영역(Search Region)만을 사용하는데, 이는 타겟을 놓쳤을 때 복구가 어렵고 오차가 누적되어 표류(Drift) 현상이 발생하는 원인이 된다. 반면, 최근 제안된 전체 프레임 탐색(Full-frame Search) 방식은 타겟 재검출은 가능하게 하지만, 전역적 문맥 정보(Global Context Information)의 부재로 인해 배경 내 유사한 방해 객체를 타겟으로 오인하는 취약점이 있다. 또한, 고정된 공간 차원의 타겟 특징 표현은 광범위한 크기 변화를 효과적으로 수용하지 못한다는 한계가 있다.

따라서 본 논문의 목표는 실시간 속도를 유지하면서도, 크기 변화에 유연하게 대응하고 전역 문맥 정보를 활용하여 방해 객체를 효과적으로 구별할 수 있는 트래킹 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문은 **TACT (TridentAlign and Context Embedding Tracker)**라는 새로운 프레임워크를 제안하며, 핵심 아이디어는 다음과 같다.

1. **TridentAlign 모듈**: 타겟 객체의 특징 표현을 여러 공간 차원으로 풀링(Pooling)하여 피처 피라미드(Feature Pyramid)를 구축한다. 이를 통해 RPN(Region Proposal Network) 단계에서 타겟의 다양한 크기 변화와 변형에 대한 적응력을 높인다.
2. **Context Embedding 모듈**: 프레임 내의 전역 문맥 정보를 추출하여 국부 특징 표현(Local Feature Representation)에 임베딩한다. 이는 타겟과 유사한 방해 객체들 사이에서 타겟을 더욱 정확하게 식별하여 오탐지를 줄이는 역할을 한다.
3. **효율적인 구조 설계**: 앵커 프리(Anchor-free) 방식인 FCOS 검출기 헤드를 RPN에 도입하여 파라미터 수를 줄이고 유연성을 높였으며, 전체 시스템이 실시간 속도로 동작하도록 설계하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 한계를 지적하며 차별점을 제시한다.

- **CNN 기반 및 Siamese 트래커**: 기존 Siamese 네트워크(SiamFC, SiamRPN++ 등)는 단순하고 빠르지만, 주로 단기 트래킹에 치중되어 있으며 전역적 문맥 모델링이 부족하다.
- **문맥 인식 트래커(Context-aware Trackers)**: 일부 연구가 문맥 정보를 활용하려 했으나, 대개 타겟 주변의 고정된 공간 영역으로 제한되어 있어 장면 전체의 방해 객체를 고려하는 전역 문맥 정보가 부족하다.
- **장기 트래커(Long-term Trackers)**: GlobalTrack과 같은 전체 프레임 탐색 기반 트래커가 존재하지만, 타겟 특징을 $1 \times 1$ 커널로 풀링하여 공간 정보를 소실한다는 점과 전역 문맥 정보 없이 국부 특징만을 사용하기에 방해 객체에 취약하다는 한계가 있다.

## 🛠️ Methodology

TACT 프레임워크는 크게 **Region Proposal Stage**와 **Classification Stage**의 두 단계로 구성된다.

### 1. Region Proposal with Scale Adaptive TridentAlign
이 단계에서는 TridentAlign 모듈을 통해 타겟의 크기 변화에 강인한 후보 영역(RoI)을 생성한다.

- **특징 피라미드 생성**: 쿼리 이미지의 특징 맵 $z$에 대해 서로 다른 공간 차원 $s_i \in \{3, 5, 9\}$로 ROIAlign을 수행하여 특징 피라미드 $Z = \{z_1, z_2, \dots, z_K\}$를 생성한다.
- **깊이별 상호 상관(Depth-wise Cross-correlation)**: 검색 이미지의 특징 맵 $x$와 피라미드의 각 $z_i$ 간의 상호 상관을 계산한다.
  $$\hat{x}_i = x \sim z_i$$
- **특징 정제**: 생성된 다중 스케일 상관 맵 $[\hat{x}_1, \hat{x}_2, \dots, \hat{x}_K]$를 Self-attention 블록에 통과시켜 특정 위치와 타겟 스케일에 집중하도록 정제한다.
- **FCOS 기반 검출**: 정제된 맵을 통해 이진 분류 레이블 $p_{i,j}$와 바운딩 박스 회귀 값 $t_{i,j}$를 예측한다. 훈련 시 사용하는 손실 함수 $L_{rpn}$은 다음과 같다.
  $$L_{rpn}(\{p_{i,j}\}, \{t_{i,j}\}) = \frac{1}{N_{pos}} \sum_{i,j} L_{cls}(p_{i,j}, c^*_{i,j}) + \frac{\lambda}{N_{pos}} \sum_{i,j} \mathbb{1}\{c^*_{i,j} > 0\} L_{reg}(t_{i,j}, t^*_{i,j})$$
  여기서 $L_{cls}$는 Focal Loss를, $L_{reg}$는 Linear IoU Loss를 사용한다.

### 2. Classification with Context-Embedded Features
RPN에서 제안된 후보 RoI들을 전역 문맥 정보를 활용하여 최종적으로 분류하고 정제한다.

- **전역 문맥 생성**: 후보 특징 집합 $X = \{x_1, x_2, \dots, x_N\}$에 대해 요소별 최대값(Max-pooling)과 평균값(Average-pooling)을 계산하여 결합한 $x_{cxt} \in \mathbb{R}^{s \times s \times 2c}$를 생성한다.
- **문맥 임베딩 과정**: 
    1. **Context Generator $g_1(\cdot)$**: $x_{cxt}$로부터 전역 문맥 정보를 생성한다.
    2. **Context Embedder $g_2(\cdot)$**: 후보 특징 $x_i$와 생성된 문맥 정보를 결합하여 문맥 임베딩 특징 $\tilde{x}_i$를 생성한다.
    - 본 논문에서는 여러 변형(Simple concat, Simple add, CBAM, FILM)을 실험했으며, **FILM(Feature-wise Linear Modulation)** 방식이 가장 우수함을 확인하였다. FILM은 다음과 같은 아핀 변환(Affine Transformation)을 통해 특징을 변조한다.
      $$\text{Output} = \gamma \otimes x_i + \beta$$
- **최종 분류**: 문맥 임베딩된 후보 특징 $\tilde{x}_i$와 타겟 특징 $\tilde{z}_0$를 요소별 곱셈($\tilde{x}_i \otimes \tilde{z}_0$)하여 비교한 후, 이진 분류와 박스 정제를 수행한다. 손실 함수 $L_{det}$는 $L_{rpn}$과 동일한 구조의 분류 및 회귀 손실을 사용한다.

## 📊 Results

### 실험 설정
- **데이터셋**: LaSOT, OxUvA (장기 트래킹), TrackingNet, GOT-10k (단기/대규모 트래킹)
- **백본 네트워크**: ResNet-18, ResNet-50
- **지표**: AUC, Precision, MaxGM (OxUvA), Success Rate (SR), Average Overlap (AO)

### 주요 결과
- **정량적 성능**: LaSOT 데이터셋에서 TACT-50은 AUC 0.575를 기록하며 GlobalTrack, ATOM, SiamRPN++ 등의 최신 트래커보다 우수한 성능을 보였다. 특히 OxUvA의 MaxGM 및 TPR 지표에서 다른 장기 트래커들을 큰 차이로 앞질렀다.
- **실시간 속도**: ResNet-18 백본 사용 시 57 fps, ResNet-50 사용 시 42 fps로 동작하며, 배치 사이즈를 늘릴 경우 최대 101 fps까지 가속 가능하다. 이는 GlobalTrack 대비 약 9배 빠른 속도이다.
- **속성별 분석**: LaSOT의 챌린지 속성 분석 결과, TridentAlign 덕분에 **크기 변화(Scale Variation)**와 **변형(Deformation)**에 매우 강인한 모습을 보였으며, Context Embedding 덕분에 **배경 혼잡(Background Clutter)** 상황에서도 높은 성능을 유지하였다.
- **절제 실험(Ablation Study)**: 
    - TridentAlign과 Context Embedding을 각각 추가할 때마다 성능이 지속적으로 향상됨을 확인하였다.
    - 문맥 임베딩 방식 중에서는 FILM 기반 모듈이 가장 높은 성능 이득을 주었으며, 이는 원본 특징 공간의 변별력을 해치지 않으면서 적절한 조건 정보를 제공하기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 Siamese 네트워크의 고질적인 문제인 '스케일 적응력'과 '방해 객체 식별력'을 각각 TridentAlign과 Context Embedding이라는 효율적인 모듈로 해결하였다. 

특히 주목할 점은 **전체 프레임 탐색(Full-frame Search)**을 수행함에도 불구하고 실시간 속도를 달성했다는 점이다. 이는 무거운 백본 네트워크에 의존하기보다, 특징 피라미드와 전역 문맥 임베딩이라는 구조적 개선을 통해 효율성을 확보했기에 가능했다. 

비판적 관점에서 보면, 본 연구는 전역 문맥을 위해 후보 RoI들의 Max/Avg 풀링 값을 사용하는데, 이는 후보 RoI의 개수($N=64$)나 품질에 따라 전역 문맥의 대표성이 결정될 수 있다는 가정을 내포하고 있다. 하지만 실험 결과는 이러한 단순한 접근 방식이 실제 성능 향상에 충분히 기여함을 입증하고 있다.

## 📌 TL;DR

본 논문은 **TridentAlign(피처 피라미드 구축)**을 통해 대상의 크기 변화와 변형에 대응하고, **Context Embedding(전역 문맥 정보 활용)**을 통해 유사한 방해 객체를 구별하는 실시간 비주얼 트래킹 알고리즘 **TACT**를 제안한다. 이 연구는 장기 및 단기 트래킹 벤치마크에서 SOTA급 성능을 달성하면서도 매우 빠른 추론 속도를 보여, 실시간성이 중요한 실제 환경의 장기 트래킹 시스템 구축에 중요한 기여를 할 것으로 보인다.