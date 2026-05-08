# Dual Relation Knowledge Distillation for Object Detection

Zhen-Liang Ni, Fukui Yang, Shengzhao Wen, Gang Zhang (2023)

## 🧩 Problem to Solve

본 논문은 객체 탐지(Object Detection) 작업에 지식 증류(Knowledge Distillation, KD)를 적용할 때 발생하는 두 가지 핵심적인 문제점을 해결하고자 한다.

첫째는 **전경(Foreground)과 배경(Background) 특징 사이의 심각한 불균형** 문제이다. 일반적인 객체 탐지 이미지에서 전경 픽셀은 배경 픽셀보다 훨씬 적다. 기존의 KD 방법들은 모든 픽셀 특징을 동일한 우선순위로 학습하기 때문에, 학생 모델(Student model)이 배경 특징에 더 많은 주의를 기울이게 되어 정작 중요한 전경 특징의 학습이 제한되는 결과가 초래된다.

둘째는 **소형 객체(Small Object)의 특징 표현 부족** 문제이다. 소형 객체는 특징 맵에서 차지하는 영역이 작아 충분한 정보를 추출하기 어렵고, 이로 인해 기존의 KD 방식으로는 소형 객체 탐지 성능을 유의미하게 끌어올리는 데 한계가 있다.

따라서 본 논문의 목표는 픽셀 단위 및 인스턴스 단위의 관계(Relation)를 효과적으로 증류함으로써 전경 특징 학습을 강화하고, 특히 소형 객체의 탐지 성능을 향상시키는 **Dual Relation Knowledge Distillation (DRKD)** 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단순한 특징 값(Feature value)의 모방을 넘어, 특징들 간의 **'관계(Relation)'**를 증류하는 것이다.

1. **Pixel-wise Relation Distillation**: Graph Convolution을 통해 전역적인 픽셀 관계를 캡처한다. 이를 통해 학생 모델이 전경과 배경의 관계를 학습하게 하여, 모델이 자연스럽게 전경 영역에 집중하도록 유도한다.
2. **Instance-wise Relation Distillation**: 서로 다른 인스턴스 간의 유사도를 계산하여 관계 행렬을 생성한다. 특히 소형 객체는 다른 크기의 객체들과 더 풍부한 관계를 가진다는 점에 착안하여, 인스턴스 간 관계 정보를 통해 소형 객체의 부족한 특징 표현을 보완한다.
3. **Relation Filter Module**: 모든 관계가 동일하게 중요하지 않으므로, 유의미한 인스턴스 관계만을 강조하여 증류 효율을 높이는 필터 모듈을 설계하였다.

## 📎 Related Works

객체 탐지 프레임워크는 크게 후보 영역을 먼저 생성하는 Two-stage detector(예: Faster R-CNN, Cascade R-CNN)와 직접 결과를 도출하는 One-stage detector(예: SSD, YOLOv3, RetinaNet, FCOS)로 나뉜다. 이러한 모델들의 연산 비용을 줄이기 위해 Pruning, Quantization, Knowledge Distillation 등의 모델 압축 기술이 사용된다.

기존의 KD 연구들은 다음과 같은 접근 방식을 취했다.

- **FGFI**: 앵커와 타겟 박스의 교집합을 이용해 세밀한 특징 모방을 수행한다.
- **DeFeat**: 전경과 배경 특징을 분리하여 각각 증류함으로써 불균형 문제를 해결하려 했다.
- **GID**: 일반적인 인스턴스 선택 모듈을 통해 인스턴스 간의 관계 지식을 모델링했다.
- **NLD**: Non-local 모듈을 사용하여 전역 픽셀 관계를 캡처하고 이를 증류했다.

본 논문은 이러한 기존 방식과 달리, Graph Convolution을 활용한 더 효율적인 전역 픽셀 관계 추출(GloRe)과 소형 객체에 특화된 인스턴스 관계 증류 및 필터링 메커니즘을 도입하여 차별성을 갖는다.

## 🛠️ Methodology

DRKD는 픽셀 단위 관계 증류, 인스턴스 단위 관계 증류, 그리고 직접적인 인스턴스 특징 증류의 세 가지 경로로 구성된다.

### 1. Pixel-wise Relation Distillation

전경-배경 불균형 문제를 해결하기 위해 **GloRe(Graph-based global reasoning networks)** 모듈을 사용한다. 이는 Attention 메커니즘보다 전역 문맥을 더 효율적으로 캡처한다.

**절차 및 수식:**

- **Graph Embedding**: 입력 특징 $X \in \mathbb{R}^{C \times W \times H}$를 선형 층을 통해 $X \in \mathbb{R}^{C_1 \times HW}$로 변환한 후, 학습 가능한 투영 행렬 $B \in \mathbb{R}^{C_2 \times HW}$를 통해 그래프 노드 특징 $V$를 생성한다.
$$V = XB^T$$
- **Graph Convolution**: 인접 행렬(Adjacency matrix) $A$와 상태 업데이트 행렬 $W$를 사용하여 노드 간의 관계를 캡처한 관계 인식 특징 $Z$를 얻는다.
$$Z = ((I - A)V)W$$
- **Reprojection**: 관계 인식 특징 $Z$를 다시 원래의 좌표 공간으로 투영하여 픽셀 단위 관계 특징 $F$를 생성한다.
$$F = ZB$$

**손실 함수:**
교사 모델과 학생 모델의 GloRe 출력값 사이의 $L_2$ 거리를 계산한다. 이때 학생 모델 쪽에는 특징 차이를 최소화하기 위한 Adaptive Convolution $f$가 추가된다.
$$L_{PR} = \frac{1}{k} \sum_{i=1}^{k} \| \phi(t_i) - f(\phi(s_i)) \|^2$$
여기서 $\phi$는 GloRe 모듈, $t_i$와 $s_i$는 각각 교사와 학생의 특징을 의미한다.

### 2. Instance-wise Relation Distillation

소형 객체의 표현력을 높이기 위해 인스턴스 간의 유사도를 모델링한다.

**절차 및 수식:**

- **Instance Extraction**: Ground Truth 좌표를 기반으로 ROI Align ($\xi$)을 통해 인스턴스 특징을 추출하고 동일한 크기로 리사이징한다.
$$bx = \xi(x, c, o)$$
- **Relation Modeling**: Embedded Gaussian 함수를 사용하여 인스턴스 $s_i$와 $s_j$ 사이의 관계 $\psi(s_i, s_j)$를 계산한다. 이때 Relation Filter를 통해 얻은 가중치 $w_{ij}$가 적용된다.
$$\psi(s_i, s_j) = \frac{1}{\tau} e^{w_{ij} g_1(s_i) g_2(s_j)}, \quad \tau = \sum \forall i e^{w_{ij} g_1(s_i) g_2(s_j)}$$
- **Instance-wise Relation Loss**: 교사와 학생의 관계 행렬 간의 차이를 학습한다.
$$L_{IR} = \sum_{(i,j) \in N} \| \psi(t_i, t_j) - f(\psi(s_i, s_j)) \|^2$$

### 3. Instance Distillation

전경-배경 불균형을 추가로 해결하고 수렴 속도를 높이기 위해, 추출된 전경 특징 자체를 직접 증류한다.
$$L_{INS} = \frac{1}{n} \sum_{i=1}^{n} \| t_i - f(s_i) \|^2$$

### 4. Overall Loss Function

최종 손실 함수는 탐지 작업의 기본 손실($L_{det}$)과 세 가지 증류 손실의 가중 합으로 정의된다.
$$L = L_{det} + \lambda_1 L_{PR} + \lambda_2 L_{IR} + \lambda_3 L_{INS}$$

## 📊 Results

### 실험 설정

- **데이터셋**: COCO 2017 (120k 이미지, 80 클래스)
- **평가 지표**: mAP, $AP_{50}$, $AP_{75}$, $AP_S$, $AP_M$, $AP_L$
- **모델 구성**: 교사 모델(ResNeXt101), 학생 모델(ResNet50) 기반의 RetinaNet(One-stage) 및 Faster R-CNN(Two-stage)

### 주요 결과

1. **성능 향상**:
    - **Faster R-CNN**: mAP 38.4% $\rightarrow$ 41.6%로 3.2%p 상승.
    - **RetinaNet**: mAP 37.4% $\rightarrow$ 40.3%로 2.9%p 상승.
2. **소형 객체 탐지($AP_S$) 효과**:
    - RetinaNet 기준, $L_{PR}$과 $L_{IR}$을 동시에 적용했을 때 $AP_S$가 20.0%에서 23.8%로 3.8%p 크게 향상되었다. 이는 인스턴스 관계 증류가 소형 객체 표현력 보완에 핵심적임을 시사한다.
3. **Ablation Study**:
    - GloRe 기반의 픽셀 관계 모델링이 기존의 Non-local 방식보다 우수한 성능을 보였다 (Table 2).
    - Relation Filter 모듈을 적용했을 때 mAP가 추가로 상승하여, 유의미한 관계를 선별하는 것이 효과적임을 확인했다 (Table 4).
4. **SOTA 비교**:
    - Faster R-CNN, Grid R-CNN, Dynamic R-CNN, RetinaNet, RepPoints 등 다양한 프레임워크에 적용한 결과, NLD, GKD, FGD 등 최신 KD 방법론보다 우수하거나 대등한 성능을 기록하며 State-of-the-art(SOTA)를 달성하였다.

## 🧠 Insights & Discussion

**강점 및 유효성:**
본 논문은 단순한 특징값의 복제가 아니라 '관계'라는 고차원적인 정보를 증류함으로써 객체 탐지 KD의 고질적인 문제인 전경-배경 불균형과 소형 객체 인식 문제를 동시에 해결하였다. 특히 GloRe 모듈을 통한 전역 관계 학습이 모델의 시야를 전경으로 유도하는 효과가 있음을 시각화(Figure 1)와 수치로 증명한 점이 뛰어나다. 또한, One-stage와 Two-stage 디텍터 모두에 적용 가능한 일반적인(General) 프레임워크라는 점이 큰 강점이다.

**한계 및 논의사항:**

- **연산 오버헤드**: 관계 행렬을 생성하고 Graph Convolution을 수행하는 과정이 학습 단계에서 추가적인 연산량을 요구한다. 다만, 이는 학습 시에만 적용되는 KD 과정이므로 추론(Inference) 속도에는 영향을 주지 않는다.
- **하이퍼파라미터 민감도**: $\lambda_1, \lambda_2, \lambda_3$ 세 가지 가중치 파라미터에 따라 성능 차이가 발생하며, 특히 Two-stage와 One-stage 모델 간에 최적의 가중치 값이 다르다는 점은 모델마다 세밀한 튜닝이 필요함을 의미한다.
- **데이터셋 의존성**: COCO 데이터셋에서만 검증되었으므로, 매우 극단적으로 작은 객체가 많거나 배경이 매우 복잡한 특수 도메인 데이터셋에서도 동일한 성능 향상이 있을지는 추가 확인이 필요하다.

## 📌 TL;DR

본 연구는 객체 탐지 모델의 경량화를 위한 **Dual Relation Knowledge Distillation (DRKD)**를 제안한다. 전경-배경 불균형 해결을 위한 **Pixel-wise Relation Distillation(GloRe 기반)**과 소형 객체 성능 향상을 위한 **Instance-wise Relation Distillation**을 결합하여, 학생 모델이 교사 모델의 전역적/국소적 관계 지식을 학습하게 한다. 실험 결과, Faster R-CNN과 RetinaNet 등 다양한 디텍터에서 SOTA 성능을 달성했으며, 특히 소형 객체 탐지 능력을 유의미하게 개선하여 실제 모델 압축 및 배포 환경에서 높은 활용 가능성을 보여주었다.
