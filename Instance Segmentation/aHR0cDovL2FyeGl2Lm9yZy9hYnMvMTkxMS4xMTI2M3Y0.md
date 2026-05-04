# Deeply Shape-guided Cascade for Instance Segmentation

Hao Ding, Siyuan Qiao, Alan Yuille, Wei Shen (2021)

## 🧩 Problem to Solve

인스턴스 분할(Instance Segmentation)의 핵심은 다단계 구조(cascade architecture)에서 바운딩 박스 검출(bounding box detection)과 마스크 분할(mask segmentation) 사이의 관계를 얼마나 잘 활용하느냐에 있다. 기존의 최신 인스턴스 분할 캐스케이드 모델들은 주로 바운딩 박스 검출이 먼저 정교해지면 마스크 분할의 성능이 향상되는 '단방향 관계(unidirectional relationship)'에만 의존해 왔다.

본 논문은 이러한 단방향성을 넘어, 정교하게 예측된 마스크 분할 결과가 다시 바운딩 박스 검출의 정확도를 높이는 '역방향'의 이점을 어떻게 활용할 것인가라는 문제를 다룬다. 즉, 두 작업 사이의 양방향 관계(bi-directional relationship)를 구축하여 인스턴스 분할의 전체적인 정밀도를 높이는 것이 본 연구의 목표이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 마스크 예측에서 추출된 **형태 가이드(shape guidance)**를 다음 단계의 바운딩 박스 검출에 반복적으로 적용하는 **Deeply Shape-guided Cascade (DSC)** 아키텍처를 제안하는 것이다. 이는 마스크-박스 간의 '양방향 긍정 피드백 루프(positive feedback loop)'를 형성하여, 더 정확한 박스가 더 정확한 마스크를 만들고, 다시 그 마스크가 더 정밀한 박스를 유도하는 구조이다.

이를 위해 저자들은 다음과 같은 세 가지 핵심 구성 요소를 도입하였다.

1. **초기 형태 가이드 (Initial shape guidance):** 클래스 불가지론적(class-agnostic) 마스크를 생성할 수 있는 마스크 감독 기반의 Region Proposal Network (mRPN)를 도입하였다.
2. **명시적 형태 가이드 (Explicit shape guidance):** 이전 단계의 마스크 예측을 사용하여 현재 단계의 특징 추출 시 단순한 사각형 RoI가 아닌, 인스턴스의 실제 형태에 정렬된 영역에 집중하는 Shape-guided RoIAlign을 제안하였다.
3. **암시적 형태 가이드 (Implicit shape guidance):** 이전 단계의 중간 마스크 특징(intermediate mask features)을 현재 단계의 바운딩 박스 헤드에 전달하는 특징 융합(feature fusion) 연산을 도입하였다.

## 📎 Related Works

인스턴스 분할 연구는 크게 두 가지 방향으로 나뉜다.

- **분할 기반 방법 (Segmentation-based):** 먼저 시맨틱 분할을 수행하여 픽셀 단위의 카테고리 맵을 얻은 후, 각 인스턴스를 식별하는 "segment then identify" 방식이다.
- **검출 기반 방법 (Detection-based):** 후보 바운딩 박스를 먼저 생성하고 그 내부에서 마스크를 추출하는 "detect then segment" 방식으로, Mask R-CNN이 대표적이다.

최근에는 정밀도를 높이기 위해 다단계 정제 과정을 거치는 캐스케이드 구조(Cascade Mask R-CNN, Hybrid Task Cascade(HTC) 등)가 등장하였다. 하지만 HTC를 포함한 기존의 캐스케이드 모델들은 마스크 헤드가 업데이트된 바운딩 박스 정보로부터 혜택을 받는 구조에 치중되어 있었다. DSC는 이와 직교하는 방향, 즉 마스크 예측 결과가 객체 검출 브랜치(분류 및 회귀)에 도움을 줄 수 있다는 점을 활용하여 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

DSC는 RPN을 통해 인스턴스 제안(proposal)을 생성하고, 이후 일련의 박스 헤드와 마스크 헤드를 통해 바운딩 박스와 마스크를 반복적으로 정제한다. 전체 흐름은 다음과 같다.

- **mRPN $\rightarrow$ [마스크 헤드 $\rightarrow$ Shape-guided RoIAlign $\rightarrow$ 박스 헤드] $\times N$ 단계**

### 2. 주요 구성 요소

#### (1) Mask-supervised RPN (mRPN)

기존 RPN에 **Mask Proposal Network (MPN)**를 추가한 구조이다. mRPN은 바운딩 박스 정보뿐만 아니라 마스크 감독($M_g$)을 통해 클래스 불가지론적 마스크 확률 맵($M_0$)과 중간 마스크 특징 맵($F_0^-$)을 생성한다. 이는 캐스케이드의 가장 초기 단계부터 형태 가이드를 제공하는 역할을 한다.

#### (2) Shape-guided RoIAlign (SgRoIAlign)

단순한 사각형 영역에서 특징을 추출하는 기존 RoIAlign과 달리, 이전 단계에서 예측된 마스크 확률 맵 $M$을 가이드로 사용하여 특징을 추출한다.
각 빈(bin) 내의 샘플링 포인트 $(a_{h,w}^i, b_{h,w}^i)$에 대해, 특징 값 $f$와 마스크 확률 값 $m$을 곱하여 가중 평균을 구한다. 방정식은 다음과 같다.

$$f_{B,M}(h, w) = \frac{\sum_{i=1}^N f(a_{h,w}^i, b_{h,w}^i) \times (1 + m(c_{h,w}^i, d_{h,w}^i))}{N}$$

여기서 $(1 + m)$ 항을 사용하는 이유는 예측된 형태 영역에 집중하면서도, 객체 인식에 도움이 되는 주변 맥락(context) 특징을 완전히 배제하지 않기 위함이다.

#### (3) Feature Fusion 및 Adaptive Feature Alignment (AFA)

이전 단계의 중간 마스크 특징 $F_{t-1}^-$를 $1 \times 1$ 컨볼루션 층을 거쳐 현재 단계의 박스 헤드 특징에 요소별 합산(element-wise summation) 방식으로 융합한다.
이때, 단계별로 바운딩 박스가 정제되면서 발생하는 특징 불일치(misalignment) 문제를 해결하기 위해 **Adaptive Feature Alignment (AFA)** 전략을 사용한다.

- **RoI enlargement:** 특징 추출 시 RoI 영역을 일정 비율($r$)로 확장하여 추출한다.
- **RoI clipping:** 다음 단계의 확장 영역이 현재 단계의 확장 영역 내에 포함되도록 RoI를 클리핑하여, 재계산 없이 이전 단계의 특징을 재사용할 수 있게 한다. 이를 통해 연산 복잡도를 선형적으로 유지한다.

### 3. 전체 수식 요약

단계 $t$에서의 동작은 다음과 같이 정의된다.

1. 마스크 특징 추출: $F_t^m = f(B_t^e, F)$
2. 마스크 및 중간 특징 생성: $(M_t, F_t^-) = m_t(F_t^m \oplus w_t^m a(F_{t-1}^-, B_{t-1}^e, B_t^e))$
3. 박스 헤드용 특징 추출 (SgRoIAlign): $F_{t+1}^b = f^s(B_t^e, F, M_t)$
4. 바운딩 박스 정제: $B_{t+1} = b_t(B_t, F_{t+1}^b \oplus w_{t+1}^b F_t^-)$

## 📊 Results

### 1. 실험 설정

- **데이터셋:** COCO 2017 (val, test-dev)
- **지표:** Box AP ($AP_b$), Mask AP ($AP_m$)
- **비교 대상:** HTC, Mask R-CNN, Cascade Mask R-CNN 등

### 2. 정량적 결과

- **COCO val:** 다양한 백본(ResNet-50, 101, ResNeXt-101)에서 HTC 대비 평균적으로 $2.1$ box AP와 $1.5$ mask AP 향상을 보였다. 특히 더 엄격한 지표인 $AP_{75}$에서 큰 폭의 향상이 있었다.
- **COCO test-dev:** 최신 모델들과 비교했을 때, 동일 백본 기준 DSC가 가장 높은 성능을 기록하였다. 특히 DCN과 Multi-scale training을 적용했을 때 $51.8$ box AP와 $45.5$ mask AP를 달성하였다.

### 3. 효율성 분석 (DSC vs F-DSC)

특징 맵 해상도를 $14 \times 14$에서 $7 \times 7$로 줄인 **F-DSC (Fast-DSC)** 버전은 성능 하락을 최소화하면서(약 $0.3$ box AP 감소) 추론 시간을 크게 단축시켰다. F-DSC는 HTC보다 약간만 더 느리면서도 훨씬 높은 정확도를 제공한다.

### 4. Huddled Instances (밀집된 객체) 분석

객체들이 심하게 겹쳐 있는 'huddled instances' 부분 집합에 대해 실험한 결과, 겹침 정도($T_O$)와 비율($T_P$)이 높아질수록 HTC 대비 DSC의 성능 향상 폭이 커졌다(최대 $4.2$ box $AP_{oI}$, $4.7$ mask $AP_{oI}$ 향상). 이는 형태 가이드가 겹쳐진 객체들을 구분해내는 데 매우 효과적임을 입증한다.

## 🧠 Insights & Discussion

### 강점

본 논문은 단순히 네트워크를 깊게 쌓는 것이 아니라, 마스크 $\rightarrow$ 박스 $\rightarrow$ 마스크로 이어지는 **양방향 피드백 루프**를 설계함으로써 인스턴스 분할의 본질적인 관계를 잘 활용하였다. 특히 겹쳐진 객체(huddled instances) 상황에서 기존 모델들이 박스를 잘못 예측하는 문제를 Shape-guided RoIAlign을 통해 효과적으로 해결하였다.

### 한계 및 비판적 해석

- **초기 예측 의존성:** 본 모델은 '형태 가이드'에 기반하므로, mRPN에서 생성된 초기 마스크 예측의 품질이 낮을 경우 최종 결과가 악화될 가능성이 있다. 실제로 저자들은 실패 사례의 약 10%가 낮은 초기 마스크 IoU 때문임을 확인하였다.
- **연산 비용:** F-DSC가 대안을 제시하긴 했으나, 기본 DSC 구조는 추가적인 특징 융합과 가이드 연산으로 인해 기본 캐스케이드 구조보다 추론 시간이 증가한다.

## 📌 TL;DR

- **주요 기여:** 마스크 분할 결과를 바운딩 박스 검출의 가이드로 활용하는 양방향 캐스케이드 구조(DSC) 제안.
- **핵심 기술:** mRPN(초기 가이드), Shape-guided RoIAlign(명시적 가이드), 특징 융합 및 AFA(암시적 가이드).
- **성과:** COCO 데이터셋에서 HTC를 크게 상회하는 성능을 보였으며, 특히 객체가 밀집된 상황에서 탁월한 정밀도를 나타냄.
- **의의:** 인스턴스 분할에서 검출과 분할의 상호 보완적 관계를 수식적으로 구현하여 고정밀 분할의 새로운 방향성을 제시함.
