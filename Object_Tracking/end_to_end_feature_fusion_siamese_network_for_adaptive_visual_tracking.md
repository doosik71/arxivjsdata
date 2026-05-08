# End-to-end feature fusion siamese network for adaptive visual tracking

Dongyan Guo, Jun Wang, Weixuan Zhao, Ying Cui, Zhenhua Wang, Shengyong Chen (2019)

## 🧩 Problem to Solve

본 논문은 비주얼 객체 추적(Visual Object Tracking)에서 발생하는 **특징 표현의 한계와 시나리오별 적응성 부족 문제**를 해결하고자 한다.

일반적으로 추적 대상이 되는 객체는 시나리오에 따라 서로 다른 두드러진 특징(Salient features)을 가지며, 동일한 객체라 할지라도 장기 추적 과정에서 형태와 외관 특징이 지속적으로 변화한다. 기존의 단일 특징 기반 추적 방식은 다음과 같은 명확한 한계를 지닌다.

1. **Hand-crafted features (예: HOG):** 일반적인 외관 특징을 잘 포착하지만, 객체의 급격한 변형(Large deformation)에 매우 민감하여 추적 성능이 저하된다.
2. **Deep features (예: CNN):** 이미지 표현력이 뛰어나고 변형에 강건하지만, 훈련 데이터가 부족하거나 특정 장면이 포함되지 않은 경우 성능이 급격히 하락하며, 때로는 객체의 세부적인 변별 정보가 부족한 경우가 있다.

따라서 본 연구의 목표는 서로 상보적인 성격의 CNN 특징과 HOG 특징을 효과적으로 융합하여, 다양한 환경 변화(조명, 스케일, 변형, 가림 등)에 적응적으로 대응할 수 있는 **FF-Siam**이라는 End-to-end 특징 융합 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **CNN의 시맨틱 정보(Semantic information)와 HOG의 외관 정보(Appearance information)를 결합하고, 이를 채널 주의 집중(Channel Attention) 메커니즘을 통해 적응적으로 제어**하는 것이다.

구체적인 기여점은 다음과 같다.

- **End-to-End 특징 융합:** 기존의 수동적인 파라미터 설정 방식에서 벗어나, CNN과 HOG 특징의 융합 과정을 신경망을 통해 종단간(End-to-end)으로 학습시켜 최적의 융합 가중치를 도출한다.
- **채널 주의 집중 메커니즘(Channel Attention Mechanism):** 각 특징 채널의 중요도가 시나리오마다 다르다는 점에 착안하여, 현재 상황에 맞는 최적의 채널 가중치를 생성하는 가중치 생성 계층(Weight generation layer)을 도입하였다.
- **Correlation Filter(CF)의 결합:** CNN 특징이 가진 세부 변별력 부족 문제를 해결하기 위해 Siamese 네트워크 구조 내에 상관 필터(Correlation Filter)를 결합하여 보다 정교한 템플릿을 생성한다.

## 📎 Related Works

### Siamese Network 기반 추적기

SiamFC, SINT, CFNet 등 Siamese CNN 구조를 사용하는 추적기들은 오프라인 학습을 통해 보편적인 객체 기술자(Universal object descriptors)를 학습함으로써 실시간성과 성능을 동시에 확보하였다. 특히 CFNet은 Siamese 구조에 Correlation Filter를 도입하여 성능을 높였다.

### 특징 융합(Feature Fusion) 방식

다양한 추적기들이 성능 향상을 위해 여러 특징을 결합해 왔다.

- **앙상블 방식:** 여러 독립적인 추적기의 결과를 HMM 등으로 결합하는 방식.
- **상보적 학습자:** Staple과 같이 HOG와 컬러 히스토그램을 결합하여 변형과 배경 잡음에 대응하는 방식.
- **다층 특징 결합:** 얕은 특징(Shallow feature)과 깊은 특징(Deep feature)을 결합하여 외관 표현과 객체 구별 능력을 동시에 확보하려는 시도가 있었다.

**차별점:** 기존 방식들은 주로 동일 계열의 특징(Hand-crafted $\leftrightarrow$ Hand-crafted 또는 Deep $\leftrightarrow$ Deep)을 융합했으나, FF-Siam은 성격이 완전히 다른 CNN과 HOG 특징을 End-to-end 방식으로 융합하며, 특히 주의 집중 메커니즘을 통해 이를 적응적으로 조절한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

FF-Siam의 파이프라인은 크게 네 가지 계층으로 구성된다.

1. **특징 추출 계층 (Feature Extraction Layer):** 이전 프레임의 타겟 영역($x'$)과 현재 프레임의 탐색 영역($y'$)에서 각각 CNN 특징($f_c$)과 HOG 특징($f_h$)을 추출한다.
2. **가중치 생성 계층 (Weight Generation Layer):** 추출된 특징을 입력받아 각 채널의 중요도를 나타내는 채널 가중치($\phi_c, \phi_h$)를 생성한다.
3. **템플릿 생성 계층 (Template Generation Layer):** 특징과 생성된 채널 가중치를 사용하여 변별력이 높은 CNN 템플릿과 HOG 템플릿을 생성한다.
4. **융합 계층 (Fusion Layer):** 생성된 템플릿과 탐색 영역 특징 간의 컨볼루션을 통해 각각의 응답 맵(Response map)을 생성하고, 이를 융합하여 최종 타겟 위치를 결정하는 최종 응답 맵을 도출한다.

### 주요 방정식 및 학습 절차

#### 1. 응답 맵 생성

CNN과 HOG 특징에 대해 각각 다음과 같이 응답 맵 $g_c, g_h$를 생성한다.
$$g_c(x', y') = (\phi_c(f_c(x')) \otimes W(f_c(x'))) \star f_c(y')$$
$$g_h(x', y') = (\phi_h(f_h(x')) \otimes W(f_h(x'))) \star f_h(y')$$
여기서 $W$는 최적의 템플릿을 얻는 함수이며, $\phi$는 채널 가중치 함수이다.

#### 2. 특징 융합 및 최종 응답

두 응답 맵은 학습된 융합 커널 $k_d$를 통해 다음과 같이 결합된다.
$$m(x', y') = \sum_{d=1}^{D} g_d(x', y') * k_d$$
최종적으로 로지스틱 회귀에 적합하도록 스케일 $s$와 바이어스 $b$를 추가하여 최종 응답 맵 $M$을 생성한다.
$$M(x', y') = s \cdot m(x', y') + b$$

#### 3. 손실 함수 및 학습

네트워크는 오프라인으로 학습되며, 정답 라벨 $L_i$ (타겟 영역이면 1, 아니면 -1)에 대해 요소별 로지스틱 손실 함수(Element-wise logistic loss)를 최소화하도록 학습된다.
$$\arg \min \sum_{i} \ell(g(x', y'), L_i)$$

#### 4. Correlation Filter 템플릿 생성

Ridge Regression 문제를 푸는 것과 동일하게 템플릿 $w$를 최적화하며, 정규화 항을 추가하여 오버피팅을 방지한다.
$$\arg \min_{w} \|w \star x - y\|^2 + \|w\|^2$$
최종 해결책은 $\hat{w} = \frac{\hat{y}^* \circ \hat{x}}{(\hat{x}^* \circ \hat{x}) + \lambda}$ 형태로 도출된다.

#### 5. 채널 가중치 생성 (Attention Module)

특징 맵의 중앙 영역을 Crop한 후 $\text{Average Pooling} \to \text{MLP (2개 FC layers)} \to \text{Sigmoid}$ 과정을 거쳐 각 채널의 중요도 가중치를 산출한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Temple-Color, OTB50, UAV123.
- **지표:** Bounding box overlap ratio (Success rate), Center location error (Precision rate).
- **비교 대상:** KCF, Staple, SAMF, SiamFC, CFNet, MEEM 등 13종의 최신 추적기.

### 주요 결과

1. **CNN 레이어 깊이 영향:** 실험 결과, CNN 특징의 깊이가 **Conv-2**일 때 HOG 특징과의 융합 성능이 가장 좋았다. 너무 깊은 레이어(Conv-5)는 시맨틱 정보는 강하지만 세부 외관 정보가 부족하여 HOG와의 보완 효과가 상대적으로 낮았다.
2. **정량적 성능:**
   - **UAV123:** FF-Siam-CA(Attention 적용 모델)가 AUC 0.505로 가장 높은 성공률을 기록하였다.
   - **OTB50:** FF-Siam-CA가 AUC 0.572로 기존 SOTA 모델들을 상회하는 성능을 보였다.
3. **정성적 분석:** 가림(Occlusion)이 발생하거나 객체의 크기가 매우 작아지는 상황, 혹은 스케일 변화가 급격한 상황에서 다른 추적기들은 타겟을 놓치는 반면, FF-Siam은 안정적으로 추적을 유지하는 모습이 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 단순한 특징 결합이 아니라, **'상황에 맞는 특징의 선택적 활용'**이라는 관점을 성공적으로 구현하였다. CNN 특징은 객체의 범주적 정체성을 파악하는 데 유리하고, HOG 특징은 세부적인 외곽선과 형태를 잡는 데 유리하다.
특히 채널 주의 집중(Channel Attention)의 효과가 뚜렷하게 나타났는데, 예를 들어 탐색 영역에 타겟과 유사한 카테고리의 방해물(Distractor)이 존재할 경우, 시맨틱 정보(CNN)의 가중치를 낮추고 외관 정보(HOG)의 가중치를 높임으로써 오인식을 방지할 수 있음을 시사한다.

### 한계 및 논의사항

- **오프라인 학습의 의존성:** 모델이 오프라인으로 학습되므로, 학습 데이터셋에 포함되지 않은 완전히 새로운 도메인의 객체에 대해서는 일반화 성능이 떨어질 가능성이 있다.
- **온라인 업데이트:** 논문에서 단순한 선형 보간법(Eq. 11)을 통한 템플릿 업데이트를 언급하였으나, 이 업데이트 전략이 최적의 학습률 $\eta$를 어떻게 결정하는지에 대한 구체적인 분석은 부족하다.
- **계산 복잡도:** 두 가지 서로 다른 특징을 추출하고 주의 집중 모듈을 통과시켜야 하므로, 단일 특징 기반 추적기보다 연산 비용이 증가했을 가능성이 크나 이에 대한 구체적인 FPS(Frames Per Second) 수치는 명시되지 않았다.

## 📌 TL;DR

**요약:** FF-Siam은 CNN의 시맨틱 특징과 HOG의 외관 특징을 End-to-end로 융합하고, 채널 주의 집중(Channel Attention) 메커니즘을 통해 상황에 맞게 특징 가중치를 조절하는 적응형 비주얼 추적 프레임워크이다.

**의의:** 본 연구는 서로 다른 성격의 특징들이 어떻게 상보적으로 작용할 수 있는지를 신경망 구조로 증명하였으며, 특히 다양한 환경 변화(가림, 변형, 배경 잡음) 속에서도 강건한 추적이 가능함을 보여주어 향후 멀티-모달 특징 융합 기반의 추적 연구에 중요한 기초를 제공한다.
