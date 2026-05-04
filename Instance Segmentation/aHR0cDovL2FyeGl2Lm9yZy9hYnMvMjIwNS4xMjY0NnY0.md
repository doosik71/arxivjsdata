# UniInst: Unique Representation for End-to-End Instance Segmentation

Yimin Ou, Rui Yang, Lufan Ma, Yong Liu, Jiangpeng Yan, Shang Xu, Chengjie Wang, Xiu Li (2022)

## 🧩 Problem to Solve

본 논문은 인스턴스 세그멘테이션(Instance Segmentation) 분야에서 공통적으로 발생하는 **중복 표현(Redundant Representations)** 문제를 해결하고자 한다. 기존의 주류 방법론들(RoI 기반, 그리드 기반, 앵커 포인트 기반 등)은 하나의 인스턴스에 대해 여러 개의 바운딩 박스, 그리드 또는 앵커 포인트가 할당되는 'Many-to-One' 할당 방식을 사용한다.

이러한 중복 표현은 추론 단계에서 동일한 객체에 대해 여러 개의 중복된 예측 결과를 생성하며, 이를 해결하기 위해 수작업으로 설계된 **NMS(Non-Maximum Suppression)** 후처리 단계에 의존하게 만든다. NMS의 의존성은 다음과 같은 문제를 야기한다:

1. **End-to-End 학습 저해**: 후처리 단계가 필수적이므로 전체 네트워크를 완전한 종단간(end-to-end) 방식으로 학습시키는 데 제약이 있다.
2. **폐색(Occlusion) 및 밀집 장면에서의 취약성**: 객체들이 서로 겹쳐 있는 경우, NMS가 정답인 예측치까지 제거하여 성능이 저하되는 문제가 발생한다.

따라서 본 논문의 목표는 바운딩 박스와 NMS 없이, 각 인스턴스당 단 하나의 고유한 표현(Unique Representation)만을 생성하는 완전한 end-to-end 인스턴스 세그멘테이션 프레임워크인 **UniInst**를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스당 하나의 표현만을 생성하도록 강제하는 **일대일 할당(One-to-One Assignment)** 체계와 예측 결과의 품질을 교정하는 **재순위화(Re-ranking)** 전략이다.

1. **OYOR (Only Yield One Representation)**: 예측값과 정답(Ground Truth) 간의 매칭 품질을 기반으로, 각 인스턴스에 단 하나의 최적 표현만을 동적으로 할당하는 인스턴스 인식 기반의 일대일 할당 스킴을 제안한다.
2. **Prediction Re-ranking Strategy**: 분류 점수(Classification Score)와 실제 마스크 품질(Mask Quality) 사이의 불일치 문제를 해결하기 위해, 두 지표의 곱을 이용해 최종 예측치를 결정하는 재순위화 전략을 도입한다.
3. **Box-free & NMS-free Framework**: FCN(Fully Convolutional Network) 기반으로 탐지기(Detector)나 NMS 없이 직접 마스크를 예측하는 완전한 end-to-end 파이프라인을 구현하여, 특히 폐색이 심한 환경에서 강건한 성능을 입증하였다.

## 📎 Related Works

인스턴스 세그멘테이션 연구는 크게 두 가지 패러다임으로 나뉜다:

- **Two-stage 방법론**: Mask R-CNN과 같이 객체 탐지기로 제안 영역(Proposal)을 먼저 생성하고 RoI-Align을 통해 마스크를 예측한다.
- **One-stage 방법론**: SOLO, SOLOv2, CondInst와 같이 RoI 크롭 없이 단일 FCN을 통해 마스크를 예측한다.

**기존 방식의 한계**:
위의 모든 방법론은 기본적으로 **Many-to-One 할당** 방식을 따른다. 예를 들어, CondInst는 인스턴스 중심 영역 내의 여러 앵커 포인트를 양성 샘플로 지정한다. 이로 인해 추론 시 중복 예측이 발생하며, 이를 제거하기 위해 반드시 NMS가 필요하다. 최근 DETR나 DeFCN 같은 end-to-end 탐지기를 활용하려는 시도가 있었으나, 이들은 여전히 탐지 결과에 의존하여 마스크를 예측하는 구조이므로 완전한 end-to-end 인스턴스 세그멘테이션 프레임워크라고 보기 어렵다.

UniInst는 이러한 기존 접근법과 달리, 마스크 수준의 세밀한 정보를 할당 단계에서 직접 활용함으로써 탐지기와 NMS를 완전히 제거한 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

UniInst는 CondInst를 기반으로 하되, 바운딩 박스 회귀(Box Regression) 및 Center-ness 브랜치를 제거한 구조이다. 전체 파이프라인은 다음과 같이 구성된다:

- **Backbone & FPN**: ResNet-50/101-FPN을 통해 다중 스케일 특징 맵 $\{P_3, ..., P_7\}$을 추출한다.
- **Head Network**: 각 FPN 레벨에서 세 가지 병렬 브랜치가 동작한다.
  - **Classification Branch**: 클래스 확률을 예측한다.
  - **Re-ranking Branch**: 예측된 마스크의 $\text{IoU}$를 회귀 예측한다.
  - **Controller Branch**: 마스크 FCN 헤드에 전달할 동적 파라미터(Dynamic Weights)를 생성한다.
- **Mask FCN Head**: 컨트롤러에서 생성된 동적 가중치와 FPN 특징 맵을 결합하여 최종 인스턴스 마스크를 생성한다.

### 2. Instance-aware One-to-one Assignment: OYOR

OYOR는 헝가리안 알고리즘(Hungarian Algorithm)을 이용해 예측 집합 $\hat{y}$와 정답 집합 $y$ 사이의 최적의 일대일 매칭 $\hat{\pi}$를 찾는다.

$$\hat{\pi} = \arg \max_{\pi \in \Pi_{N}^{G}} \sum_{i=1}^{G} Q_{\text{match}}(\hat{y}_{\pi(i)}, y_i)$$

여기서 매칭 품질 $Q_{\text{match}}$는 공간적 제약(Spatial Prior), 분류 점수, 그리고 마스크 정확도를 모두 고려하여 다음과 같이 정의된다:

$$Q_{\text{match}}(\hat{y}_{\pi(i)}, y_i) = \mathbb{I}_{\{\pi(i) \in \Psi_i\}} \cdot (\hat{p}_{\pi(i)}(c_i))^{1-\alpha} \cdot (\text{Dice}(\hat{m}_{\pi(i)}, m_i))^\alpha$$

- $\mathbb{I}_{\{\pi(i) \in \Psi_i\}}$: 중심 샘플링 전략을 통한 공간적 사전 제약이다.
- $\hat{p}_{\pi(i)}(c_i)$: 타겟 클래스에 대한 예측 분류 점수이다.
- $\text{Dice}(\hat{m}_{\pi(i)}, m_i)$: 예측 마스크와 정답 마스크 간의 Dice 유사도 계수로, 마스크 정확도를 측정한다.
  $$\text{Dice}(\hat{m}_{\pi(i)}, m_i) = \frac{2 \cdot |\hat{m}_{\pi(i)} \cap m_i|}{|\hat{m}_{\pi(i)}| + |m_i| + \epsilon}$$
- $\alpha$: 분류 점수와 마스크 정확도 사이의 가중치를 조절하는 하이퍼파라미터이다 ($\alpha=0.9$ 기본값).

### 3. Prediction Re-ranking Strategy

분류 점수가 높더라도 마스크 품질이 낮은 'Sub-optimal' 예측이 선택되는 문제를 해결하기 위해 재순위화 전략을 사용한다.

- **학습 단계**: Re-ranking 헤드가 예측 마스크와 정답 마스크 간의 $\text{IoU}$를 예측하도록 학습시킨다. 손실 함수 $L_{\text{rank}}$는 다음과 같다:
  $$L_{\text{rank}} = \mathbb{I}_{\{\pi(i) \in \Psi_i\}} \cdot \|\widehat{\text{IoU}}_{\pi(i)} - \text{IoU}(\hat{m}_{\pi(i)}, m_i)\|_1$$
- **추론 단계**: 최종 랭킹 기준을 $\text{Classification Score} \times \text{Predicted Mask IoU}$로 설정하여, 분류 성능과 마스크 품질이 모두 높은 최적의 예측치만을 선택한다.

### 4. 학습 절차 및 손실 함수

전체 손실 함수 $L$은 다음과 같이 다중 작업 손실의 합으로 정의된다:
$$L = \lambda_{\text{cls}}L_{\text{cls}} + \lambda_{\text{mask}}L_{\text{mask}} + \lambda_{\text{rank}}L_{\text{rank}} + \lambda_{\text{aux}}L_{\text{aux}}$$
특히 $L_{\text{aux}}$는 일대일 할당으로 인해 부족해진 학습 감독 신호를 보완하기 위해, 많은 수의 양성 샘플을 사용하는 Many-to-One 방식의 보조 손실을 추가하여 특징 학습을 강화한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: COCO test-dev2017 (일반 성능 평가), OCHuman (심한 폐색 환경 평가).
- **지표**: mask AP (Average Precision).
- **백본**: ResNet-50-FPN, ResNet-101-FPN.

### 2. 정량적 결과

- **COCO 데이터셋**: UniInst는 ResNet-50-FPN 기준 **39.0 mask AP**, ResNet-101-FPN 기준 **40.2 mask AP**를 달성하였다. 이는 NMS를 사용하는 Mask R-CNN 및 CondInst보다 우수하거나 경쟁력 있는 수치이다.
- **OCHuman 데이터셋**: 폐색이 심한 환경에서 UniInst는 CondInst 대비 **+12.6 mask AP**라는 압도적인 성능 향상을 보였으며, 이는 NMS-free 구조와 인스턴스 인식 할당 방식이 중첩된 객체를 구분하는 데 매우 효과적임을 입증한다.
- **추론 속도**: ResNet-50-FPN 기준 약 **21.1 FPS** (47.5ms/img)를 기록하여, 기존 Mask R-CNN이나 SOLOv2보다 빠르거나 유사한 속도를 보여 실용성을 입증하였다.

### 3. 주요 분석 결과

- **NMS 제거 영향**: 기존 Many-to-One 방식(CondInst 등)은 NMS 제거 시 AP가 19.4포인트나 급락하지만, UniInst는 단 0.2포인트만 감소하여 중복 표현 문제가 근본적으로 해결되었음을 보여준다.
- **$\alpha$ 값의 영향**: 분류 점수만 고려($\alpha=0$)하거나 마스크 정확도만 고려($\alpha=1$)하는 것보다, 적절한 융합($\alpha=0.9$)이 가장 높은 성능을 낸다.

## 🧠 Insights & Discussion

### 강점

UniInst는 인스턴스 세그멘테이션에서 고질적이었던 NMS 의존성을 제거함으로써 파이프라인을 단순화하였고, 특히 폐색이 심한 장면에서 기존 방식들이 겪던 '객체 소실' 문제를 획기적으로 개선하였다. 이는 바운딩 박스라는 중간 단계 없이 마스크의 기하학적 특성(Dice coefficient)을 할당 단계에 직접 반영한 결과로 해석된다.

### 한계 및 향후 과제

1. **작은 객체 세그멘테이션**: 실험 결과, UniInst는 작은 인스턴스에 대한 세그멘테이션 성능이 다소 낮게 나타났다. 이에 대한 손실 함수 설계나 할당 전략 수정이 필요하다.
2. **하이퍼파라미터 최적화**: 현재 손실 함수의 가중치가 동일하게 설정되어 있으나, AutoML 등을 통해 최적의 가중치를 찾는 연구가 필요하다.
3. **모델 배포**: TensorRT 등의 포맷 변환을 통한 추가적인 가속화 가능성이 남아 있다.

### 비판적 해석

본 논문은 OYOR와 Re-ranking을 통해 NMS-free를 달성했지만, 사실상 헝가리안 알고리즘을 통한 일대일 매칭이 NMS의 역할을 학습 단계에서 대신 수행하도록 유도한 것으로 볼 수 있다. 다만, 이를 위해 도입된 보조 손실($L_{\text{aux}}$)이 실제 성능 향상에 어느 정도 기여하는지에 대한 더 세밀한 분석이 필요하다.

## 📌 TL;DR

UniInst는 **OYOR(일대일 할당 스킴)**와 **Prediction Re-ranking(마스크 품질 기반 재순위화)**을 통해, 바운딩 박스와 NMS 후처리 없이 각 인스턴스당 단 하나의 고유한 표현만을 생성하는 완전한 end-to-end 인스턴스 세그멘테이션 프레임워크이다. COCO 데이터셋에서 경쟁력 있는 성능을 보였으며, 특히 **폐색이 심한 OCHuman 데이터셋에서 기존 방법론을 압도하는 강건함**을 입증하여, 향후 FCN 기반 end-to-end 세그멘테이션 연구의 새로운 방향성을 제시하였다.
