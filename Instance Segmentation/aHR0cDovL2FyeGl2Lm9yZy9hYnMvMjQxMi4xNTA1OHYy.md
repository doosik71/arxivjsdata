# MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance

Hallee E. Wong, Jose Javier Gonzalez Ortiz, John Guttag, Adrian V. Dalca (2024)

## 🧩 Problem to Solve

의료 영상 분석 분야에서 새로운 데이터셋에 대해 세그멘테이션(segmentation) 작업을 수행하는 것은 매우 노동 집약적인 과정이다. 기존의 접근 방식은 크게 두 가지로 나뉜다. 첫째, 대화형 세그멘테이션(interactive segmentation) 방식은 사용자가 클릭이나 스크리블(scribble) 등의 힌트를 제공하여 결과를 얻지만, 모든 개별 이미지에 대해 이 과정을 독립적으로 반복해야 하므로 데이터셋 전체를 처리하는 데 드는 노력이 이미지 수에 비례하여 선형적으로 증가한다. 둘째, In-Context Learning 방식은 몇 개의 예시(labeled examples)를 모델에 제공하여 새로운 작업을 수행하게 하지만, 대개 충분한 크기의 기존 레이블링된 데이터셋이 미리 존재해야 하며 사용자가 예측 결과를 수정할 수 있는 메커니즘이 부족하다.

본 논문의 목표는 기존의 레이블링된 데이터가 없는 상태에서도 사용자가 몇 개의 이미지를 세그멘테이션하면, 그 결과가 다음 이미지들의 가이드가 되어 점진적으로 필요한 사용자 상호작용 횟수를 줄여나가는 확장 가능한(scalable) 대화형 세그멘테이션 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **대화형 세그멘테이션과 In-Context Learning을 결합**하는 것이다. 사용자가 데이터셋의 첫 번째 이미지를 대화형으로 세그멘테이션하면, 이 이미지-마스크 쌍이 '컨텍스트 세트(context set)'에 추가된다. 이후의 이미지들을 세그멘테이션할 때 모델은 현재 이미지의 상호작용 정보뿐만 아니라 이 컨텍스트 세트를 함께 입력으로 받아 정보를 활용한다. 결과적으로 데이터셋을 처리함에 따라 컨텍스트 세트가 커지게 되고, 모델은 해당 작업의 특성을 학습하여 새로운 이미지를 세그멘테이션하는 데 필요한 사용자 클릭이나 스크리블 횟수를 획기적으로 줄일 수 있게 된다.

## 📎 Related Works

1. **Interactive Segmentation**: SAM(Segment Anything Model), MedSAM 등은 개별 이미지에 대해 효율적인 세그멘테이션을 가능케 하지만, 데이터셋 전체를 처리할 때는 모든 이미지에 대해 상호작용을 반복해야 하는 한계가 있다.
2. **In-Context Learning**: UniverSeg 등은 추론 시 예시 데이터를 제공하여 새로운 작업을 수행하지만, 대개 많은 양의 컨텍스트 데이터가 필요하며 예측 오류를 수정하는 인터페이스가 없다.
3. **Continual Learning & Annotation-Efficient Learning**: 일부 연구는 온라인 학습을 통해 모델 가중치를 업데이트하며 적응하려 하지만, 이는 매번 재학습(retraining)이나 미세 조정(fine-tuning)을 요구하며 계산 비용이 크다.

MultiverSeg는 모델 가중치를 업데이트하는 재학습 과정 없이, 추론 단계에서 컨텍스트 세트를 입력으로 받는 것만으로도 작업에 적응하는 방식을 취함으로써 위 방법론들의 한계를 극복한다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

MultiverSeg는 인코더-디코더 구조의 UNet 기반 아키텍처를 사용한다. 모델의 입력은 크게 두 가지다.

- **Target Image Inputs**: 세그멘테이션 대상이 되는 이미지 $x_i$, 사용자의 상호작용(클릭, 박스, 스크리블) $u_{i,j}$, 그리고 이전 단계의 예측 결과 $\hat{y}_{i,j-1}$이 포함된다. 이들은 총 5개의 채널로 스택(stack)되어 입력된다.
- **Context Set Inputs**: 이전에 세그멘테이션이 완료된 이미지-마스크 쌍들의 집합 $S = \{(x_l, y_l)\}_{l=1}^m$이다.

### CrossBlock 메커니즘

모델의 핵심은 일반적인 합성곱 블록 대신 **CrossBlock**을 사용하는 것이다. CrossBlock은 타겟 이미지의 특징 맵 $q$와 가변적인 크기의 컨텍스트 세트 특징 맵 $V = \{v_i\}_{i=1}^n$ 사이의 정보를 상호작용시킨다.

1. **Cross-Convolution**: 타겟 특징 $q$와 각 컨텍스트 특징 $v_i$를 결합(concatenation)하여 합성곱을 수행한다.
   $$\text{CrossConv}(q, V; \theta_z) = \{z_i\}_{i=1}^n, \quad z_i = \text{Conv}(q \| v_i; \theta_z)$$
2. **Feature Update**: 위에서 얻은 $z_i$를 사용하여 타겟 특징 $q'$와 컨텍스트 특징 $v_i'$를 업데이트한다.
   $$z_i = \text{LN}(A(\text{CrossConv}(q, v_i; \theta_z)))$$
   $$q' = \text{LN}(A(\text{Conv}(\frac{1}{n} \sum_{i=1}^n z_i; \theta_q)))$$
   $$v_i' = \text{LN}(A(\text{Conv}(z_i; \theta_v)))$$
   여기서 $A(\cdot)$는 비선형 활성화 함수이며, $\text{LN}(\cdot)$은 LayerNorm을 의미한다. 이 과정을 통해 모델은 컨텍스트 세트의 정보 중 타겟 이미지와 관련이 높은 정보를 추출하여 세그멘테이션에 활용한다.

### 훈련 절차 및 손실 함수

모델은 다양한 생의학 데이터셋과 합성 데이터를 사용하여 한 번만 훈련된다.

- **손실 함수**: Soft Dice Loss와 Focal Loss의 합을 최소화한다.
  $$L(\theta; T) = \mathbb{E}_{t \in T} \left[ \mathbb{E}_{(x_i, y_i; S) \in t} \left[ \sum_{j=1}^k L_{\text{seg}}(y_i, \hat{y}_{i,j}) \right] \right]$$
- **Prompt Simulation**: 훈련 시 사용자의 상호작용을 시뮬레이션한다. 첫 단계에서는 무작위로 클릭/스크리블을 생성하고, 이후 단계에서는 정답($y_i$)과 예측값($\hat{y}_{i,j-1}$) 사이의 오차 영역에서 수정 상호작용을 생성한다.
- **Synthetic Task Generation**: 일반화 성능을 높이기 위해 실제 이미지에 Superpixel 알고리즘을 적용하여 가상의 레이블을 생성하고, 이를 바탕으로 합성 데이터셋을 만들어 훈련에 활용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 79개의 생의학 데이터셋 중 67개로 훈련하고 12개(187개 작업)로 평가하였다.
- **비교 대상**: 대화형 세그멘테이션 모델(SAM, MedSAM, ScribblePrompt 등), In-Context 모델(UniverSeg), 그리고 이를 결합한 baseline(SP+UVS) 및 상한선으로 nnUNet(완전 지도 학습)을 사용하였다.
- **평가 지표**: Dice Score, 95th percentile Hausdorff distance ($\text{HD}_{95}$), 그리고 90% Dice에 도달하기까지 필요한 상호작용 횟수($\text{NoI}_{90}$)를 측정하였다.

### 주요 결과

1. **상호작용 횟수의 감소**: MultiverSeg는 컨텍스트 세트가 커질수록 90% Dice에 도달하기 위해 필요한 상호작용 횟수가 급격히 감소함을 보였다. (그림 3 참조)
2. **정량적 효율성**: ScribblePrompt와 비교했을 때, 90% Dice 달성을 위해 필요한 전체 클릭 횟수는 약 36.4%, 스크리블 단계는 약 25.3% 감소하였다.
3. **타 모델 대비 우위**: In-Context 모델인 UniverSeg보다 더 높은 Dice 점수를 기록하였으며, 특히 컨텍스트 세트의 크기가 작을 때도 안정적인 성능을 보였다.
4. **미세 조정(Fine-tuning) 대비 효율성**: 몇 개의 이미지를 레이블링한 후 모델을 미세 조정하는 방식보다 MultiverSeg를 사용하는 것이 상호작용 횟수가 적었으며, 특히 추론 시간 면에서 압도적이었다(미세 조정은 이미지당 평균 20분 소요, MultiverSeg 추론은 0.15초 미만).

## 🧠 Insights & Discussion

**강점**: MultiverSeg의 가장 큰 강점은 **'학습 없는 적응(adaptation without retraining)'**이다. 추론 시에 컨텍스트 세트를 입력으로 넣는 것만으로 새로운 도메인의 작업에 빠르게 적응하며, 이는 의료 현장에서 전문가가 소량의 데이터만으로 전체 데이터셋을 빠르게 레이블링해야 하는 실제 요구사항을 정확히 충족한다.

**한계 및 비판적 해석**:

- **데이터 이질성**: 데이터셋 내부의 이미지 간 변동성(heterogeneity)이 매우 큰 경우(예: BUID 데이터셋), 컨텍스트 세트가 일정 규모(약 5개 이상)가 될 때까지는 성능 향상이 더디게 나타나는 경향이 있다.
- **컨텍스트 품질 의존성**: 훈련은 정답(ground truth) 레이블로 수행되었으나, 실제 추론 시에는 이전 단계의 '예측값'이 컨텍스트로 들어간다. 논문에서는 이진화(thresholding)를 통해 품질을 높였다고 언급하지만, 초기 예측의 오류가 컨텍스트 세트에 누적될 경우 발생할 수 있는 '오류 전이' 문제에 대한 심층적인 분석은 부족하다.

## 📌 TL;DR

MultiverSeg는 의료 영상 세그멘테이션에서 **대화형 입력과 In-Context Learning을 통합**하여, 사용자가 이미지를 세그멘테이션할수록 이후 작업에 필요한 노력이 줄어들게 만드는 시스템이다. CrossBlock 구조를 통해 타겟 이미지와 이전 예시들 간의 특징을 효율적으로 융합하며, 기존 SOTA 모델 대비 상호작용 횟수를 25~36% 절감하였다. 이 연구는 특히 재학습 없이 추론 단계에서 즉각적으로 새로운 데이터셋에 적응할 수 있다는 점에서 의료 영상 데이터 구축의 효율성을 극대화할 가능성이 크다.
