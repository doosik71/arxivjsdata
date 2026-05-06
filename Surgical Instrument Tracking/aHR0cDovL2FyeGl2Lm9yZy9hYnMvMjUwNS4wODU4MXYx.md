# ReSurgSAM2: Referring Segment Anything in Surgical Video via Credible Long-term Tracking

Haofeng Liu, Mingqi Gao, Xuxiao Luo, Ziyue Wang, Guanyi Qin, Junde Wu, and Yueming Jin (2025)

## 🧩 Problem to Solve

수술 장면 분할(Surgical scene segmentation)은 컴퓨터 보조 수술에서 수술의 질을 높이고 환자의 예후를 개선하는 데 매우 중요한 기술이다. 특히, 텍스트 표현을 통해 특정 대상 객체를 식별하고 분할하는 Referring Surgical Segmentation은 외과의에게 인터랙티브한 경험을 제공할 수 있다는 점에서 큰 장점을 가진다.

그러나 기존의 방법론들은 다음과 같은 세 가지 핵심적인 문제점을 가지고 있다. 첫째, 효율성이 낮고 단기적인 추적(Short-term tracking)에 의존하여 복잡한 실제 수술 시나리오에 적용하기 어렵다. 둘째, 일반적인 Referring Video Object Segmentation (RVOS) 모델들은 10초 미만의 짧은 영상에 최적화되어 있어, 수 시간 동안 진행되며 역동적인 장면 변화가 발생하는 수술 영상의 장기 추적(Long-term tracking)에 취약하다. 셋째, 최근 주목받는 SAM2(Segment Anything Model 2)는 시각적 프롬프트(Bounding box, Points)에 의존하므로 수술 중 외과의에게 라벨링 부담을 주며, 대상 객체가 영상 초반에 나타나지 않을 경우 초기화가 불가능하다는 한계가 있다.

따라서 본 논문의 목표는 텍스트 프롬프트를 통해 수술 영상 내 특정 객체를 정밀하게 탐지하고, 이를 기반으로 장기적으로 안정적인 추적을 수행하는 효율적인 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM2를 기반으로 하되, 텍스트 기반의 탐지와 신뢰할 수 있는 추적을 분리한 **2단계(Two-stage) 프레임워크**를 설계한 것이다.

중심적인 설계 아이디어는 다음과 같다.

1. **CSTMamba (Cross-Modal Spatial-Temporal Mamba)**: Mamba의 선형 복잡도 특성을 활용하여 텍스트와 영상의 공간-시간적 의존성을 효율적으로 캡처함으로써 정밀한 초기 탐지를 수행한다.
2. **CIFS (Credible Initial Frame Selection)**: 탐지 단계에서 생성된 마스크의 신뢰도를 평가하여, 오차 누적을 방지할 수 있는 최적의 초기 프레임을 선택하여 추적 단계로 넘긴다.
3. **DLM (Diversity-Driven Long-term Memory)**: SAM2의 단순한 그리디(Greedy) 메모리 선택 방식을 개선하여, 신뢰도가 높으면서도 시각적 다양성이 확보된 프레임을 메모리 뱅크에 저장함으로써 장기적인 추적 성능을 극대화한다.

## 📎 Related Works

기존의 수술 도구 분할 연구들은 주로 비디오 데이터에만 의존하여 모든 도구를 집합적으로 분할하는 semantic mask를 생성하는 방식이었다. 최근 RSVIS와 같은 연구가 텍스트 기반의 referring segmentation을 시도하였으나, 이는 연속된 3개의 프레임 정보만을 사용하는 단기 정보 의존성 때문에 장기 추적이 어렵다는 한계가 있었다.

일반 도메인의 RVOS 방법론 중 Online 방식은 실시간 처리가 가능하지만 수술 영상과 같은 장편 영상에서의 견고함이 부족하며, Offline 방식은 미래 프레임 정보를 필요로 하므로 실시간 수술 중 적용(Intraoperative application)이 불가능하다. SAM2는 강력한 추적 능력을 보여주었지만, 시각적 프롬프트에 대한 의존성과 단순한 메모리 업데이트 전략으로 인해 수술 도메인의 특수한 요구사항을 충족하지 못했다. ReSurgSAM2는 이러한 한계들을 극복하기 위해 텍스트-비전 통합 탐지와 다양성 기반의 메모리 메커니즘을 도입하여 차별화를 꾀했다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

ReSurgSAM2는 크게 **탐지 단계(Detection Stage)**와 **추적 단계(Tracking Stage)**의 두 단계로 구성된다. 텍스트 표현 $e$와 영상 프레임 $f_t$가 입력되면, 먼저 CSTMamba를 통해 텍스트에 해당하는 대상 객체를 탐지하고 CIFS를 통해 가장 신뢰할 수 있는 프레임을 선정한다. 이후 해당 프레임을 초기값으로 하여 DLM이 적용된 SAM2 추적 단계로 전환된다.

### 1단계: CSTMamba 및 CIFS (탐지 및 초기 프레임 선정)

**CSTMamba**는 텍스트-비전 간의 상호작용을 위해 sensory memory bank $S$를 사용하며, 최근 2개의 프레임 특징을 저장한다. CSTMamba 블록은 다음과 같은 구성 요소를 포함한다.

- **STMamba & 2D DWConv**: Mamba의 선택적 스캐닝을 통해 글로벌 특징을 잡고, $7 \times 7$ Depth-wise Convolution을 통해 세밀한 지역 특징을 캡처한다.
- **Inverted Bottleneck**: MLP 블록의 차원을 4배 확장하여 공간-시간적 상호작용 표현력을 높인다.
- **Bidirectional Cross-modal Attention**: Text-to-Vision (T2V) 및 Vision-to-Text (V2T) 어텐션을 통해 두 모달리티 간의 정보를 융합한다.

**CIFS (Credible Initial Frame Selection)**는 오차 누적을 막기 위해 엄격한 기준을 적용한다. 슬라이딩 윈도우 $W$ 내에서 다음과 같은 조건을 모두 만족하는 프레임들을 후보로 선정한다.
$$W = \{ f_j | j \in [t-N_w+1, t] \land iou_j > \delta_{iou} \land \text{sigmoid}(o_j) > \delta_o \}$$
여기서 $iou_j$는 예측된 IoU 점수, $o_j$는 occlusion 점수이며, $\delta_{iou}$와 $\delta_o$는 각각의 임계값이다. 윈도우 크기 $N_w$만큼 조건을 만족하는 프레임이 쌓이면, 그중 IoU 점수가 가장 높은 프레임을 최종 초기 참조 프레임으로 선택한다.

### 2단계: DLM (Diversity-Driven Long-term Memory)

SAM2의 기본 메모리 방식은 단순히 가장 최근의 프레임들을 선택하는 그리디 전략을 사용하므로 관점 오버피팅(Viewpoint overfitting)과 중복성 문제가 발생한다. **DLM**은 이를 해결하기 위해 후보 풀 $P$를 운용한다.
먼저 IoU 점수가 임계값 $\gamma_{iou}$보다 높은 고신뢰도 프레임만을 후보 풀에 추가한다.
$$P = P \cup \{ f_t | iou_t > \gamma_{iou} \}$$
후보 풀이 가득 차면, 현재 롱텀 메모리 뱅크 $L$의 최신 프레임 $l_k$와 코사인 유사도가 가장 낮은(즉, 가장 다양한 정보를 가진) 프레임 $p^*$를 선택하여 업데이트한다.
$$p^* = \arg \min_{p_i \in P} \frac{M(p_i) \cdot M(l_k)}{\|M(p_i)\| \|M(l_k)\|}$$
여기서 $M(\cdot)$은 메모리 인코더이다. 이렇게 선택된 다양하고 신뢰할 수 있는 메모리는 SAM2의 숏텀 메모리와 결합되어 장기 추적의 일관성을 보장한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Ref-EndoVis17 (수술 도구) 및 Ref-EndoVis18 (수술 도구 및 조직 - kidney parenchyma, covered kidney, small intestine 포함).
- **평가 지표**: 영역 정확도를 측정하는 $J$ score, 경계 정확도를 측정하는 $F$ score, 그리고 이들의 평균인 $J\&F$를 사용한다. 또한 효율성 측정을 위해 FPS(Frames Per Second)를 측정하였다.
- **비교 대상**: Offline 방식(ReferFormer, MUTR)과 Online 방식(RSVIS, OnlineRefer, RefSAM)을 비교 대상으로 설정하였다.

### 정량적 및 정성적 결과

실험 결과, ReSurgSAM2는 모든 데이터셋에서 기존 SOTA 방법론들을 크게 상회하는 성능을 보였다. 특히 $J\&F$ 지표에서 Ref-EndoVis17의 경우 최대 14.17, Ref-EndoVis18 tool의 경우 7.76의 향상을 기록하였다. 효율성 면에서도 **61.2 FPS**라는 매우 빠른 속도로 작동하여 실시간 수술 지원 가능성을 입증하였다.

정성적 분석(Fig. 2) 결과, RSVIS와 RefSAM은 객체의 빠른 움직임이나 장면 변화가 있을 때 추적 안정성이 떨어지거나 분할이 불완전한 모습을 보였으나, ReSurgSAM2는 견고한 초기화와 다양성 기반 메모리를 통해 일관된 추적 성능을 유지하였다.

### Ablation Study

각 구성 요소의 기여도는 다음과 같이 확인되었다.

- **2단계 프레임워크**: 단순 탐지 대비 $J\&F$ 2.64 상승.
- **CSTMamba 도입**: 정밀한 탐지를 통해 $J\&F$ 4.77 상승.
- **CIFS 적용**: 신뢰할 수 있는 참조 프레임 선택으로 $J\&F$ 6.14 상승.
- **DLM 도입**: 장기 메모리 모델링을 통해 $J\&F$ 3.03 상승.

## 🧠 Insights & Discussion

본 논문은 SAM2라는 강력한 기반 모델을 수술 도메인의 특성에 맞게 최적화하여, 특히 '텍스트 기반 인터랙션'과 '장기 추적'이라는 두 마리 토끼를 잡았다는 점에서 강점이 있다. 특히 Mamba 구조를 도입하여 트랜스포머의 2차 복잡도 문제를 해결하고 실시간성(61.2 FPS)을 확보한 점은 실제 임상 적용 가능성을 높이는 중요한 요소이다.

다만, 본 연구는 SAM2의 가중치를 초기값으로 사용하며 SAM2의 기본 아키텍처에 의존하고 있다. 따라서 SAM2 자체의 한계나 잠재적 오류가 전이될 가능성이 있으며, 텍스트 인코더로 사용된 CLIP이 수술 도메인의 특수 용어(Specialized medical terminology)를 얼마나 완벽하게 이해하고 있는지에 대한 심층적인 분석은 명시되지 않았다. 또한, CIFS에서 사용하는 임계값($\delta_{iou}, \delta_o$)들이 데이터셋마다 최적값이 다를 수 있다는 점이 향후 일반화 성능의 변수가 될 수 있다.

## 📌 TL;DR

ReSurgSAM2는 텍스트 프롬프트를 통해 수술 영상 내 객체를 분할하고 추적하는 2단계 프레임워크이다. CSTMamba를 통한 정밀 탐지 $\rightarrow$ CIFS를 통한 최적 프레임 선정 $\rightarrow$ DLM을 통한 다양성 기반 장기 추적 과정을 거치며, 기존 모델 대비 월등한 정확도와 실시간성(61.2 FPS)을 달성하였다. 이 연구는 향후 수술 중 실시간 네비게이션 및 교육용 AR 시스템의 핵심 기술로 활용될 가능성이 매우 높다.
