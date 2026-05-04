# Perceive, Excavate and Purify: A Novel Object Mining Framework for Instance Segmentation

Jinming Su, Ruihong Yin, Xingyue Chen and Junfeng Luo (2023)

## 🧩 Problem to Solve

본 논문은 인스턴스 분할(Instance Segmentation) 분야에서 여전히 해결되지 않은 두 가지 핵심 과제를 해결하고자 한다.

첫 번째는 **구분하기 어려운 객체(Indistinguishable objects)**의 발견이다. 특히 실제 환경에서는 객체들이 서로 인접해 있거나 겹쳐 있는 경우가 많아, 이를 개별적인 인스턴스로 정확히 위치시키고 분할하는 것이 매우 어렵다.

두 번째는 **인스턴스 간의 관계 모델링(Modeling the relationship between instances)** 부족이다. 서로 다른 인스턴스 사이의 특징 공간(Feature space) 내 거리와 같은 관계 정보는 인스턴스를 구분하는 데 매우 중요함에도 불구하고, 기존 연구에서는 이를 명시적으로 다루는 경우가 드물었다.

따라서 본 논문의 목표는 '인지(Perceiving)', '굴착(Excavating)', '정제(Purifying)'라는 세 단계의 파이프라인을 통해 하드 샘플(Hard samples)을 효과적으로 찾아내고, 인스턴스 간의 관계를 명시적으로 모델링하여 인스턴스 분할 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **PEP(Perceiving, Excavating, and Purifying)**라고 명명된 객체 마이닝 프레임워크를 제안하는 것이다.

1.  **Semantics Perceiving**: 먼저 의미론적 분할을 통해 명확하게 드러나는 객체들을 인지하고, 이를 'Original instance descriptors'로 정의한다.
2.  **Object Excavating**: 인지된 original instance 주변에 숨어 있거나 겹쳐 있어 구분하기 어려운 객체들을 추가로 찾아내는 메커니즘을 도입하여 'Mined descriptors'를 확보한다.
3.  **Instance Purifying**: 확보된 모든 descriptor들을 그래프 구조로 모델링하여, 동일 객체에 속한 descriptor 간의 유사도는 높이고 서로 다른 객체 간의 유사도는 낮춤으로써 인스턴스 간의 변별력을 극대화한다.

## 📎 Related Works

### 기존 인스턴스 분할 접근 방식
논문에서는 기존 방식을 세 가지 범주로 분류한다.
- **Top-down**: 객체 탐지(Object Detection)를 먼저 수행한 후 박스 내부를 분할하는 방식 (예: Mask R-CNN)
- **Bottom-up**: 픽셀 단위의 임베딩을 학습한 후 이를 클러스터링하는 방식 (예: SSAP)
- **Direct**: 탐지나 클러스터링 단계 없이 직접적으로 분할을 수행하는 방식 (예: SOLO)

### 하드 샘플 마이닝 (Hard Example Mining)
OHEM이나 Focal Loss와 같은 기존 기법들이 하드 샘플 학습을 돕지만, 이는 주로 학습 전략 차원의 간접적인 방식이며 객체 수준(Object level)에서 명시적으로 인지하여 마이닝하는 방식이 아니라는 한계가 있다.

### 객체 관계 모델링 (Object Relation)
Relation Network나 IRNet 등이 객체 간의 관계를 학습하여 성능을 높였으나, 관계 모듈 내에서 구체적으로 어떤 정보가 학습되는지에 대한 명시적인 표현력이 부족하다는 점을 지적한다.

## 🛠️ Methodology

### 1. 전체 파이프라인
본 프레임워크는 **Feature Extractor $\rightarrow$ Semantics Perceiving $\rightarrow$ Object Excavating $\rightarrow$ Instance Purifying $\rightarrow$ Mask Learning** 순서로 구성된다.

### 2. 세부 구성 요소 및 절차

#### (1) Feature Extractor
Feature Pyramid Network(FPN)를 사용하여 공통 특징을 추출한다. ResNet-101을 백본으로 사용하며, 픽셀 단위 예측을 위해 마지막 분류 및 전역 평균 풀링 레이어는 제거한다.

#### (2) Semantics Perceiving (인지)
먼저 의미론적 분할(Semantic Segmentation)을 통해 각 픽셀의 클래스 확률을 계산한다.
- **분류 및 위치 파악**: 카테고리 신뢰도가 높은 픽셀들을 선택하여 **Original instance descriptors** $\mathcal{D} = \{I_{ind}\}_{ind=1}^{N_{ori}}$를 생성한다.
- **손실 함수**: 예측된 맵 $P^s$와 정답 맵 $G^s$ 사이의 Cross-Entropy(CE) 손실을 최소화한다.
$$L_P = \sum_{s=1}^{5} CE(P^s, G^s)$$

#### (3) Object Excavating (굴착)
Original instance 주변에 겹쳐 있거나 가려진 '구분하기 어려운 객체'를 찾는다.
- **작동 방식**: Original descriptor를 입력으로 하여 해당 주변 픽셀이 새로운 인스턴스의 중심인지 판별하는 인스턴스 레벨 의미론적 분할을 수행한다.
- **Mined descriptors 생성**: 중심이라고 판단된 key pixel들에 대해 CoordConv를 사용하여 좌표 정보를 결합하고, 이를 통해 새로운 **Mined descriptors**를 학습한다.
- **손실 함수**: 중심점 탐지 손실 $L_E$와 굴착된 객체의 분류 손실 $L_{PE}$를 사용한다.
$$L_E = \sum_{ind=1}^{N_{ori}} CE(E_{ind}, G_{E_{ind}})$$
$$L_{PE} = \sum_{ind=1}^{N_{ori}} \sum_{e=1}^{N_{E_{ind}}} CE(P_{E_{e_{ind}}}, G_{E_{e_{ind}}})$$

#### (4) Instance Purifying (정제)
모든 descriptor(Original + Mined) 간의 관계를 그래프로 모델링하여 중복을 제거하고 변별력을 높인다.
- **관계 그래프**: 각 노드는 instance descriptor이며, 엣지의 가중치는 특징 공간에서의 거리(유사도)를 나타낸다.
- **목표**: 동일 객체 descriptor 간에는 높은 유사도를, 서로 다른 객체 간에는 낮은 유사도를 갖도록 인접 행렬 $M$을 학습시킨다.
- **손실 함수**: 예측 유사도 행렬 $M$과 정답 행렬 $G_M$ 사이의 CE 손실을 최소화한다.
$$L_M = CE(M, G_M)$$

#### (5) Mask Learning (마스크 학습)
최종적으로 정제된 descriptor $I_{ind}$와 공유 특징 맵 $B$를 컨볼루션 연산($\otimes$)하여 최종 마스크 $M_{ind}$를 생성한다.
$$M_{ind} = I_{ind} \otimes B$$
이 결과물은 정답 마스크 $G_{M_{ind}}$와의 CE 손실 $L_M$을 통해 학습된다.

### 3. 전체 학습 목표
모든 손실 함수를 가중 합산하여 최종 목적 함수를 정의한다.
$$\min \mathcal{L} = L_P + \alpha L_E + \beta L_{PE} + \gamma L_M + \delta L_M$$

## 📊 Results

### 실험 설정
- **데이터셋**: COCO dataset (train2017, val2017, test-dev2017)
- **지표**: AP, $AP_{50}$, $AP_{75}$, $AP_S$, $AP_M$, $AP_L$
- **환경**: ResNet-101 + FPN 백본 사용, SGD 옵티마이저, 36 epoch 학습.

### 정량적 결과
제안 방법은 26개의 최신(SOTA) 알고리즘과 비교했을 때 전반적으로 우수한 성능을 보였다.
- **전체 AP**: Two-stage 방법 중 2위인 DCT-Mask R-CNN(40.1%)보다 0.8%p 높은 **40.9%**를 달성했다.
- **크기별 성능**: 특히 대형 객체에 대한 성능($AP_L$)에서 BCNet+FCOS 대비 8.7%p라는 압도적인 향상을 보였으며, $AP_M$에서도 DCT-Mask R-CNN 대비 1.7%p 향상되었다.
- **소형 객체($AP_S$)**: 일부 Two-stage 방법보다 낮았으나, 연산 오버헤드가 더 적음에도 불구하고 경쟁력 있는 수치를 기록했다.

### 절제 연구 (Ablation Study)
- **Excavating의 효과**: Baseline(38.5% AP) 대비 Object Excavating 모듈 추가 시 39.8%로 1.3%p 상승하여 하드 샘플 마이닝의 유효성을 입증했다.
- **Purifying의 효과**: Baseline 대비 Instance Purifying 모듈 추가 시 40.1%로 상승하여 관계 모델링의 중요성을 확인했다.
- **결합 효과**: 두 모듈을 모두 적용한 PEP 프레임워크는 40.6%~40.9%의 최고 성능을 내어 두 메커니즘이 서로 보완적임을 보여주었다.

## 🧠 Insights & Discussion

### 강점
본 논문은 단순히 네트워크의 깊이를 더하거나 학습 전략을 수정하는 대신, **'인지 $\rightarrow$ 굴착 $\rightarrow$ 정제'**라는 명시적인 단계별 파이프라인을 제안함으로써 인스턴스 분할의 고질적인 문제인 객체 겹침과 구분 문제를 효과적으로 해결했다. 특히, Descriptor 기반의 그래프 정제 과정은 계산 복잡도가 낮으면서도 인스턴스 간의 변별력을 크게 높였다.

### 한계 및 비판적 해석
- **Descriptor 수의 가정**: 논문에서는 인스턴스 descriptor의 수가 보통 30개 이하로 제한적이기에 그래프 연산이 효율적이라고 언급했다. 하지만 매우 밀집된 환경(예: 수백 명의 사람이 모인 군중 씬)에서는 이 가정이 흔들릴 수 있으며, 이때의 계산 비용과 성능 변화에 대한 분석이 부족하다.
- **하이퍼파라미터 민감도**: 최종 손실 함수에서 $\alpha, \beta, \gamma, \delta$를 모두 1로 설정했다고 명시했으나, 각 손실 함수의 스케일이 다를 가능성이 높음에도 불구하고 최적의 가중치 탐색 과정에 대한 상세한 설명이 없다.

## 📌 TL;DR

본 논문은 인스턴스 분할에서 겹쳐 있거나 인접한 객체를 찾기 어려운 문제와 인스턴스 간 관계 모델링 부족 문제를 해결하기 위해 **PEP(Perceiving, Excavating, Purifying)** 프레임워크를 제안한다. 명확한 객체를 먼저 인지하고, 그 주변의 숨은 객체를 '굴착'하며, 최종적으로 그래프를 통해 인스턴스 간 유사도를 '정제'하는 과정을 거친다. COCO 데이터셋 실험 결과, 특히 대형 및 중형 객체 분할에서 SOTA 성능을 기록하며 제안한 객체 마이닝 관점의 유효성을 입증했다.