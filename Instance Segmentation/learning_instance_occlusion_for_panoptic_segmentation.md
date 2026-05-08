# Learning Instance Occlusion for Panoptic Segmentation

Justin Lazarow, Kwonjoon Lee, Kunyu Shi, Zhuowen Tu (2020)

## 🧩 Problem to Solve

Panoptic Segmentation은 이미지 내의 'things'(개수 산출이 가능한 개별 객체 인스턴스)와 'stuff'(형태가 불분명한 배경 영역)를 동시에 분할하여 하나의 출력으로 통합하는 작업이다. 기존의 일반적인 접근 방식은 인스턴스 분할(Instance Segmentation) 결과와 시맨틱 분할(Semantic Segmentation) 결과를 융합(Fusion)하여 겹치지 않는 픽셀 할당을 수행한다.

이 과정에서 발생하는 핵심 문제는 인스턴스 간의 겹침(Overlap)을 해결하는 방식이다. 기존의 baseline(Kirillov et al.)은 객체 탐지 신뢰도(Detection Confidence)가 높은 인스턴스를 우선적으로 배치하고, 그 뒤에 오는 낮은 신뢰도의 인스턴스는 겹치는 부분을 제거하는 greedy한 방식을 사용한다. 그러나 **탐지 신뢰도는 실제 물리적인 가려짐(Occlusion) 관계와 상관관계가 낮다**. 예를 들어, 배경에 있는 커다란 객체가 매우 높은 신뢰도를 가질 경우, 그 앞에 위치하여 실제로 가려야 할 작은 객체가 신뢰도가 낮다는 이유로 제거되는 현상이 발생한다. 본 논문의 목표는 인스턴스 간의 가려짐 관계를 명시적으로 학습하여 융합 과정에서의 오류를 줄이고 panoptic segmentation의 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스 간의 상대적인 가려짐 관계를 이진 분류(Binary Classification) 문제로 정의하고, 이를 예측하는 **Occlusion Head**를 추가하는 것이다.

1. **OCFusion 제안**: 두 인스턴스 마스크가 상당 부분 겹칠 때, 어떤 마스크가 앞(Top)에 위치해야 하는지를 판단하는 경량화된 모델을 제안하였다.
2. **명시적 가려짐 모델링**: 단순히 신뢰도에 의존하지 않고, 두 인스턴스의 특징과 마스크 형태를 직접 입력받아 가려짐 관계를 추론함으로써 더 정확한 융합이 가능하게 하였다.
3. **인스턴스 수준의 가려짐 해결**: 클래스 간(Inter-class) 가려짐뿐만 아니라, 동일 클래스 내(Intra-class) 인스턴스 간의 가려짐 관계까지 해결할 수 있도록 설계하였다.

## 📎 Related Works

**Panoptic Segmentation** 분야에서는 Panoptic FPN, UPSNet, AUNet 등 다양한 아키텍처가 제안되었다. 이들은 주로 공유 backbone을 사용하거나 시맨틱-인스턴스 분할 간의 일관성을 높이는 방향으로 발전해 왔으나, 인스턴스 간의 명시적인 가려짐 추론(Occlusion Reasoning) 문제는 다루지 않았다.

**Occlusion Ordering** 연구는 과거부터 존재해 왔으며, 최근 OANet과 같은 연구가 가려짐 문제를 해결하려 시도하였다. 하지만 OANet은 클래스 수준에서 어떤 클래스가 다른 클래스보다 앞에 오는지를 학습하는 방식인 반면, OCFusion은 **개별 인스턴스 간의 관계**를 모델링한다. 이는 "두 명의 사람 중 누가 앞에 있는가?"와 같은 동일 클래스 내 가려짐 문제를 해결할 수 있다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

OCFusion은 Panoptic FPN 아키텍처를 기반으로 하며, 공유된 Feature Pyramid Network(FPN) backbone을 통해 시맨틱 분할 브랜치와 인스턴스 분할 브랜치(Mask R-CNN)를 운영한다. 여기에 추가적으로 두 마스크의 관계를 판단하는 **Occlusion Head**가 탑재된다.

### Occlusion Head Formulation

두 인스턴스 마스크 $M_i$와 $M_j$의 교집합을 $I_{ij} = M_i \cap M_j$라고 할 때, 각 마스크의 교집합 비율을 다음과 같이 정의한다.
$$R_i = \frac{\text{Area}(I_{ij})}{\text{Area}(M_i)}, \quad R_j = \frac{\text{Area}(I_{ij})}{\text{Area}(M_j)}$$
두 마스크 중 어느 하나의 비율이라도 임계값 $\rho$보다 크면($R_i \ge \rho \text{ or } R_j \ge \rho$), 두 인스턴스는 '상당한 겹침(Appreciable Overlap)'이 있다고 판단하며, 이때 Occlusion Head가 다음의 이진 관계를 예측한다.
$$\text{Occlude}(M_i, M_j) = \begin{cases} 1 & \text{if } M_i \text{ should be placed on top of } M_j \\ 0 & \text{if } M_j \text{ should be placed on top of } M_i \end{cases}$$

### Occlusion Head 아키텍처

Occlusion Head는 Mask R-CNN의 추가적인 head로 구현된다.

- **입력**: 두 개의 (soft) 마스크 $M_i, M_j$와 각각의 RoI feature가 입력된다.
- **처리 과정**:
  - 마스크는 max pooling을 통해 $14 \times 14$ 크기로 축소된다.
  - 축소된 마스크와 FPN feature를 결합(Concatenate)한다.
  - 3개의 $3 \times 3$ convolution 레이어(512 feature maps)와 1개의 stride 2 convolution 레이어를 통과한다.
  - Flatten 후 1024 차원의 Fully Connected(FC) 레이어를 거쳐 최종적으로 하나의 logit 값을 출력한다.

### 융합 절차 (Fusion with Occlusion Head)

기존의 confidence 기반 greedy 융합 방식이 다음과 같이 수정된다.

1. 인스턴스 마스크들을 순회하며 픽셀을 할당한다.
2. 현재 마스크 $M_i$가 이미 할당된 픽셀을 가진 마스크 $M_j$와 상당한 겹침($\rho$ 기준)이 있는지 확인한다.
3. 만약 $\text{Occlude}(M_i, M_j) = 1$이라면, $M_i$가 $M_j$보다 앞에 있으므로 $M_j$가 점유하고 있던 겹침 영역($I_{ij}$)의 픽셀을 $M_i$가 뺏어온다.
4. 가려짐 관계가 모두 해결된 후, 최종적으로 남은 픽셀의 비율이 $\tau$보다 크면 panoptic 맵에 할당한다.

### 학습 방법 및 손실 함수

- **Ground Truth 생성**: Panoptic GT와 Instance GT를 비교하여, 교집합 영역에서 더 많은 픽셀을 점유하고 있는 인스턴스를 '앞에 있는 것'으로 정의하여 가려짐 행렬을 미리 계산한다.
- **학습 절차**:
  - 다른 파라미터들은 동결(frozen)시킨 채 Occlusion Head만 fine-tuning 한다.
  - 데이터 불균형을 막기 위해 모든 쌍을 반전시켜 $\text{Occlude}(M_i, M_j) = 0 \iff \text{Occlude}(M_j, M_i) = 1$이 되도록 학습 데이터를 구성한다.
- **손실 함수**: 이진 분류 문제이므로 Binary Cross-Entropy (BCE) 손실 함수 $L_o$를 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: COCO와 Cityscapes panoptic benchmark를 사용하였다.
- **지표**: Panoptic Quality (PQ) 및 이를 세분화한 $PQ_{Th}$(Things), $PQ_{St}$(Stuff)를 사용하였다.
- **비교 대상**: Baseline (Panoptic FPN), UPSNet, OANet, AdaptIS 등.

### 주요 결과

1. **성능 향상**: COCO 데이터셋에서 ResNet-50 기반 baseline 대비 PQ가 39.5에서 41.3으로 향상되었으며, 특히 $PQ_{Th}$에서 46.5 $\rightarrow$ 49.4로 뚜렷한 상승을 보였다. $PQ_{St}$는 변하지 않았는데, 이는 OCFusion이 인스턴스 간의 문제만 다루기 때문이다.
2. **SOTA 달성**: ResNeXt-101 backbone과 deformable convolution을 적용했을 때, COCO validation set에서 45.7 PQ를 달성하며 매우 경쟁력 있는 성능을 보였다.
3. **가려짐 예측 정확도**: Occlusion Head 자체의 분류 정확도는 COCO 91.58%, Cityscapes 93.60%로 매우 높게 나타났다.
4. **추론 시간**: 연산 복잡도는 $O(n^2)$이지만, 신뢰도 필터링과 겹침 임계값 $\rho$를 통해 실제 계산 대상 쌍을 대폭 줄였다. 결과적으로 COCO에서 baseline 대비 2.0%, Cityscapes에서 4.7%의 아주 미미한 시간 증가만 발생하였다.

## 🧠 Insights & Discussion

**강점 및 효과**
본 연구는 panoptic segmentation의 고질적인 문제였던 '잘못된 인스턴스 제거' 문제를 명시적인 가려짐 모델링으로 해결하였다. 특히 시각적 비교 결과, 기존 방식에서는 가려짐 관계로 인해 사라졌어야 할 인스턴스가 사라지거나, 반대로 앞에 있어야 할 객체가 뒤로 숨는 현상이 OCFusion을 통해 자연스럽게 해결됨을 확인하였다.

**한계 및 분석**
가장 주목할 점은 **Intra-class occlusion(동일 클래스 내 가려짐)** 처리 능력이다. 실험 결과, 클래스 간 가려짐만 처리했을 때보다 동일 클래스 내 가려짐까지 처리했을 때 $PQ_{Th}$가 유의미하게 상승하였다. 이는 현실 세계에서 동일 종류의 객체들이 겹쳐 있는 경우가 많음을 시사하며, 단순한 클래스 우선순위 기반 방식(OANet 등)의 한계를 극복한 지점이다. 다만, 본 연구는 'things' 간의 관계에만 집중하였으며, 'stuff'와 'things' 간의 가려짐 관계나 'stuff' 내부의 관계는 다루지 않았다.

## 📌 TL;DR

이 논문은 Panoptic Segmentation의 융합 과정에서 탐지 신뢰도에만 의존해 객체를 제거하던 기존 방식의 문제를 지적하고, **두 인스턴스의 가려짐 관계를 예측하는 경량화된 Occlusion Head**를 제안하였다. 이를 통해 동일 클래스 내의 객체 가려짐까지 정확하게 해결함으로써, 연산 비용의 증가를 최소화하면서도 객체 분할 품질($PQ_{Th}$)을 유의미하게 향상시켰다. 이 연구는 향후 더 복잡한 씬 이해를 위한 레이아웃 추론 및 가려짐 분석 연구에 중요한 기초가 될 것으로 보인다.
