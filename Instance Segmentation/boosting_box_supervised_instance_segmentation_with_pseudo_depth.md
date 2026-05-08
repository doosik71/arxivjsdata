# Boosting Box-supervised Instance Segmentation with Pseudo Depth

Xinyi Yu, Ling Yan, Pengtao Jiang, Hao Chen, Bo Li, Lin Yuanbo Wu, Linlin Ou (2024)

## 🧩 Problem to Solve

본 논문은 Bounding Box 감독 하의 인스턴스 분할(Box-supervised Instance Segmentation)에서 발생하는 한계를 해결하고자 한다. 일반적으로 인스턴스 분할을 위해서는 픽셀 수준의 정교한 Mask 주석이 필요하지만, 이는 비용이 매우 높기 때문에 Bounding Box만을 이용한 약지도 학습(Weakly Supervised Learning) 방식이 연구되어 왔다.

그러나 Bounding Box 감독의 근본적인 문제는 박스 내부에 포함된 전경(Foreground)과 배경(Background)을 명확하게 구분할 수 있는 정보가 부족하다는 점이다. 이로 인해 네트워크가 박스 내부의 배경 영역을 전경으로 오인하여 예측하는 배경 노이즈(Background noise) 문제가 발생하며, 이는 완전 지도 학습(Fully Supervised Learning) 방식과 비교했을 때 상당한 성능 격차를 만드는 원인이 된다. 본 연구의 목표는 별도의 추가 주석 없이 가용 가능한 Pseudo Depth 정보를 활용하여 객체의 형태와 전경-배경 간의 구분을 명확히 함으로써 분할 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전경과 배경 사이의 깊이(Depth) 차이가 뚜렷하며, 동일한 객체 내부에서는 깊이 값이 연속적으로 변화한다는 직관에 기반한다. 이를 위해 다음과 같은 세 가지 핵심 설계 전략을 제안한다.

첫째, Mask 예측 헤드에 깊이 예측 레이어를 통합한 **DG-MaskHead**를 설계하여 네트워크가 Mask와 Depth를 동시에 예측하도록 함으로써, 깊이 특징을 Mask 생성 과정에 직접 활용하게 한다.

둘째, 유사한 깊이 값을 가진 인접 픽셀들이 동일한 예측 결과(전경 또는 배경)를 갖도록 강제하는 **Depth Consistency Loss**를 도입하여 배경 노이즈를 억제하고 Mask의 정교함을 높인다.

셋째, 학습 후반부에 수행되는 Self-distillation 과정에서 **Depth-aware Hungarian algorithm**을 제안한다. 단순히 IoU 점수만으로 Pseudo Mask를 선택하는 대신, 깊이 일관성 점수(Depth consistency score)를 매칭 비용에 포함시켜 훨씬 더 신뢰할 수 있는 Pseudo Mask를 선택하여 학습에 활용한다.

## 📎 Related Works

**Box-supervised Instance Segmentation** 분야에서는 GrabCut을 이용해 Pseudo Mask를 생성하는 SDI나, 픽셀 간의 친화도(Affinity) 관계를 이용하는 BBTP, BoxInst 등이 제안되었다. 최근에는 BoxTeacher와 같이 Self-training 프레임워크를 통해 Pseudo Mask의 질을 높이려는 시도가 있었으나, 여전히 박스 감독만으로는 객체의 정확한 형상을 파악하는 데 한계가 있다.

**Depth and Segmentation**의 관계에 관한 기존 연구들은 RGB와 Depth 정보가 상호 보완적이라는 점을 밝혀왔다. 일부 연구에서는 RGB-D 데이터를 직접 입력으로 사용하거나 Multi-task learning을 통해 성능을 높였으나, 실제 환경에서 모든 이미지에 대해 Ground-truth Depth를 확보하는 것은 불가능에 가깝다. 본 논문은 이 문제를 해결하기 위해 기학습된(Off-the-shelf) 깊이 예측 모델을 통해 생성된 coarse한 Pseudo Depth를 학습 과정에서만 효율적으로 활용하는 전략을 취함으로써, 추론 단계에서는 추가적인 깊이 정보 없이도 성능을 유지하도록 설계하여 기존 연구와 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 파이프라인

시스템은 크게 두 단계로 구성된다. 첫 번째 단계에서는 Bounding Box 주석과 Pseudo Depth Map을 사용하여 Student 네트워크를 초기 학습시킨다. 두 번째 단계에서는 EMA(Exponential Moving Average) 방식으로 업데이트되는 Teacher 네트워크를 통해 Pseudo Mask를 생성하고, 이를 다시 Student 네트워크가 학습하는 Self-distillation 과정을 거친다.

### 2. Depth-guided Mask Prediction (DG-MaskHead)

본 연구는 CondInst를 베이스라인으로 하며, Mask 예측 헤드를 수정하여 **DG-MaskHead**를 제안한다. 이 헤드는 Mask와 Depth를 동시에 예측하는 Multi-task 구조를 가진다.

상세 과정은 다음과 같다. 먼저 FPN에서 추출된 특징 맵 $F_0$에 상대 좌표 맵을 결합한 후, 두 개의 레이어를 거쳐 강화된 특징 $F_1$을 생성한다. 이후 깊이 예측 레이어 $M_d$와 Mask 예측 레이어 $M_m$이 다음과 같이 작동한다.

$$F_1 = M_2(M_1(F_0))$$
$$P_{depth} = \sigma(M_d(F_1))$$
$$P_{mask} = \sigma(M_m(F_1) \cdot P_{depth})$$

여기서 $\sigma(\cdot)$는 Sigmoid 함수이다. 특히 최종 Mask 예측 시 예측된 깊이 맵 $P_{depth}$를 곱해줌으로써 깊이 정보가 Mask 생성에 직접적인 가이드 역할을 하게 한다.

### 3. 손실 함수 (Loss Functions)

학습을 위해 세 가지 손실 함수를 결합하여 사용한다.

**가. Instance Depth Estimation Loss ($L_{depth}$):**
전체 이미지 대신 각 인스턴스 박스 영역($B$) 내에서만 Pseudo Depth($P_{true\_depth}$)와의 차이를 계산한다.
$$L_{depth} = B \cdot \|P_{depth} - P_{true\_depth}\|^2$$

**나. Pairwise Depth Consistency Loss ($L_{cons}$):**
인접한 두 픽셀 $(x,y)$와 $(i,j)$ 사이의 깊이 일관성 $S_d$를 다음과 같이 정의한다.
$$S_d = \exp(-|d_{x,y} - d_{i,j}|)$$
만약 $S_d$가 임계값 $\tau_d$보다 크면 두 픽셀은 같은 클래스(전경 혹은 배경)여야 한다고 가정하며, 이에 대한 손실 함수는 다음과 같다.
$$L_{cons} = -\sum \mathbb{1}\{S_d > \tau_d\} \log P(y=1)$$
여기서 $P(y=1) = m_{x,y} \cdot m_{i,j} + (1-m_{x,y}) \cdot (1-m_{i,j})$이며, $m$은 픽셀의 Mask 예측값이다.

**다. 최종 학습 손실:**
$$L_{mask} = L_{boxinst} + L_{cons} + L_{depth}$$

### 4. Pseudo Mask Matching using Depth

Self-distillation 단계에서 Teacher 네트워크가 생성한 수많은 Mask 중 가장 적절한 것을 선택하기 위해 **Depth-aware Hungarian algorithm**을 사용한다. 기존의 IoU 기반 매칭에 깊이 일관성 점수 $S_{dcons}$를 추가하여 매칭 비용을 계산한다.

깊이 일관성 점수 $S_{dcons}$는 예측된 Mask 영역 내에서 깊이 일관성이 임계값을 넘는 비율로 정의된다. 최종 매칭 점수 $S_{match}$는 다음과 같다.
$$S_{match} = \alpha \text{IoU} + \beta S_{dmatch} + (1-\alpha-\beta)S^T_{pred}$$

이렇게 선택된 Pseudo Mask 중 매칭 점수가 임계값 $\tau_m$을 넘는 신뢰할 수 있는 Mask에 대해서만 Dice Loss를 적용하여 Student 네트워크를 최적화한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** COCO (80개 카테고리), Cityscapes (8개 도로 장면 카테고리)
- **백본:** ResNet-50, ResNet-101, Swin-Base
- **지표:** Mask AP (Average Precision)

### 2. 정량적 결과

**COCO 데이터셋:**

- ResNet-101 백본 기준, $3\times$ 학습 일정에서 36.0% Mask AP를 달성하여 BoxInst(33.2%) 및 BoxLevelSet(33.4%)보다 높은 성능을 보였다.
- 특히 Swin-Base 백본과 Box Regression branch의 품질 지표를 'centerness'에서 'IoU' score로 변경했을 때, 최고 **41.0% Mask AP**를 기록하며 Box-supervised 방식 중 매우 강력한 성능을 입증했다.

**Cityscapes 데이터셋:**

- ResNet-50 사용 시 24.4% Mask AP를 달성하여 기존 SOTA(BoxTeacher, 21.7%) 대비 2.7%p 향상되었다.
- Swin-Tiny 백본 및 COCO 사전 학습 가중치를 사용했을 때는 최고 28.9% Mask AP를 기록했다.

### 3. 정성적 결과

시각화 결과, 제안 방법은 복잡한 폐쇄(Occlusion) 상황에서 더 강건한 성능을 보였으며, 전경 객체와 유사한 색상을 가진 배경 노이즈를 효과적으로 억제하는 모습을 보였다.

## 🧠 Insights & Discussion

본 논문은 Pseudo Depth라는 보조 정보를 통해 Bounding Box의 정보 부족 문제를 효과적으로 해결했다. 특히 깊이 정보가 단순히 입력 데이터로 들어가는 것이 아니라, **예측 헤드의 구조적 결합(DG-MaskHead)**과 **일관성 손실 함수($L_{cons}$)**를 통해 네트워크가 능동적으로 깊이 특징을 학습하게 만든 점이 주효했다.

**주요 분석 및 통찰:**

- **Depth Consistency의 중요성:** Ablation study를 통해 $\tau_d$ 값이 너무 낮으면 노이즈가 유입되어 성능이 떨어짐을 확인했으며, 적절한 임계값(0.5) 설정이 필수적임을 보였다.
- **Self-distillation의 최적화:** Teacher 네트워크의 입력 이미지 크기를 800px로 키웠을 때 더 정교한 Pseudo Mask가 생성되어 Student의 성능이 유의미하게 향상되었다.
- **Dice Loss 가중치:** Dice Loss 계수 $\gamma$를 높여 Projection Loss보다 크게 설정했을 때, 네트워크가 배경 노이즈에 덜 민감해지고 Pseudo Mask의 가이드에 더 집중하게 됨을 발견했다.

한계점으로는 Pseudo Depth 생성 모델(DPT)의 정확도에 어느 정도 의존한다는 점이 있으나, 본 논문은 이를 'coarse'한 정보로 정의하고 일관성(Consistency) 관점에서 접근함으로써 모델의 부정확성을 완화하려 노력했다.

## 📌 TL;DR

이 논문은 Bounding Box 주석만으로 인스턴스 분할을 수행할 때 발생하는 배경 노이즈 문제를 해결하기 위해 **Pseudo Depth**를 도입했다. Depth 예측 레이어를 통합한 **DG-MaskHead**, 깊이 연속성을 강제하는 **Depth Consistency Loss**, 그리고 깊이 정보를 활용해 신뢰도 높은 Pseudo Mask를 선택하는 **Depth-aware Hungarian algorithm**을 제안했다. 결과적으로 COCO와 Cityscapes 데이터셋에서 기존 Box-supervised 방법론들을 상회하는 성능을 달성했으며, 이는 추가적인 정밀 주석 없이도 기하학적 정보(Depth)를 활용해 약지도 학습의 한계를 극복할 수 있음을 시사한다.
