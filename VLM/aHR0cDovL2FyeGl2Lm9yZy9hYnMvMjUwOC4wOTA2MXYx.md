# VLM-3D: End-to-End Vision-Language Models for Open-World 3D Perception

Fuhao Chang, Shuxin Li, Yabei Li, Lei He (2025)

## 🧩 Problem to Solve

본 논문은 자율주행 시스템이 복잡한 도로 환경에서 이전에 학습하지 않은 새로운 객체 범주(unseen object categories)를 식별하고 탐지해야 하는 **Open-set Perception(개방형 집합 인식)** 문제를 해결하고자 한다.

자율주행에서 롱테일(long-tail) 시나리오나 이상 상황이 발생했을 때, 정해진 카테고리만을 인식하는 기존의 Closed-set 방식은 미탐지(missed detection)나 오탐지(false alarm)를 빈번하게 일으켜 안전에 심각한 위협이 된다. 이를 해결하기 위해 풍부한 세계 지식과 시맨틱 추론 능력을 갖춘 Vision-Language Models(VLMs)가 대안으로 제시되었으나, 기존의 VLM 기반 접근 방식들은 주로 VLM을 단순한 시각 특징 추출기로 사용하고 이를 전통적인 객체 탐지기와 결합하는 **다단계 파이프라인(multi-stage pipeline)** 구조를 취한다. 이러한 구조는 단계별로 오차가 누적되는 **Error Propagation** 문제를 야기하며, 자율주행의 핵심인 정밀한 3D 공간 추론 능력을 제한하고 실시간 배포 효율성을 떨어뜨리는 한계가 있다.

따라서 본 논문의 목표는 VLM이 직접 3D 기하학적 인식을 수행할 수 있도록 하는 **최초의 End-to-End 3D Open-set Perception 프레임워크**인 VLM-3D를 제안하는 것이다.

## ✨ Key Contributions

VLM-3D의 핵심 아이디어는 거대 시각-언어 모델의 강력한 제로샷(zero-shot) 능력을 유지하면서, 이를 3D 공간의 좌표계로 직접 매핑하는 End-to-End 구조를 설계하는 것이다.

주요 기여 사항은 다음과 같다:

1. **효율적인 모델 적응(Adaptation):** Qwen2-VL 모델에 **LoRA(Low-Rank Adaptation)**를 적용하여, 매우 적은 파라미터 업데이트만으로도 자율주행 작업에 특화된 튜닝이 가능하게 하여 임베디드 플랫폼 배포 가능성을 높였다.
2. **Joint Semantic-Geometric Loss 설계:** 시맨틱 정렬과 기하학적 정밀도를 동시에 잡기 위해, 학습 초기에는 토큰 수준의 시맨틱 손실을 통해 안정적인 수렴을 도모하고, 이후 단계에서 3D IoU 손실을 도입하여 바운딩 박스의 정밀도를 높이는 2단계 최적화 전략을 제안하였다.
3. **통합 End-to-End 아키텍처:** 데이터 전처리부터 모달리티 융합, 3D 바운딩 박스 예측까지 하나의 프레임워크 내에서 처리함으로써 정보 손실을 최소화하고 시스템의 강건성을 향상시켰다.

## 📎 Related Works

### 기존 연구 및 한계

1. **Closed-set 3D Detection:** VoxelNet, PointPillars, PointRCNN, CenterPoint 등이 대표적이다. 이들은 포인트 클라우드나 이미지를 기반으로 높은 정확도를 보이지만, 학습 단계에서 정의되지 않은 새로운 객체에 대해서는 일반화 능력이 부족하다.
2. **Open-set Object Detection:**
   - **OV3D:** 이미지-텍스트 정렬을 통해 3D 시맨틱 세그멘테이션을 수행하지만, 실내 환경 위주이며 End-to-End 구조가 아니다.
   - **VL-SAM:** 어텐션 맵을 이용해 학습 없이 탐지를 수행하지만, 추론 속도가 느리고 VLM의 환각(hallucination) 현상에 취약하다.
   - **PointCLIP:** CLIP을 포인트 클라우드로 확장하여 3D Open-set 탐지를 수행하지만, 다단계 처리 과정으로 인해 자율주행의 실시간 요구사항을 충족하기 어렵다.
3. **Multimodal Large Models:** Qwen2-VL, CLIP, LLaVA 등이 강력한 표현력을 갖추고 있으나, 3D 공간의 연속적인 좌표값을 직접 출력하는 능력이 부족하거나 연산 비용이 너무 높아 임베디드 장치 적용이 어렵다.

### VLM-3D의 차별점

VLM-3D는 기존의 다단계 파이프라인을 제거하고, LoRA를 통해 경량화된 Qwen2-VL이 직접 LiDAR 좌표계의 3D 바운딩 박스 파라미터를 예측하게 함으로써 실시간성과 정확도를 동시에 추구한다.

## 🛠️ Methodology

### 전체 시스템 구조

VLM-3D는 크게 네 가지 구성 요소로 이루어져 있다: **멀티모달 입력 전처리 $\rightarrow$ LoRA 기반 특징 융합 $\rightarrow$ 2단계 손실 최적화 $\rightarrow$ Open-set 전략**.

### 1. Multimodal Input Preprocessing

- **텍스트 입력:** 사용자의 프롬프트(예: "이미지 내의 보행자를 탐지하여 LiDAR 좌표계의 3D 바운딩 박스로 출력하라")를 BERT 기반 토크나이저로 처리하여 Vocab ID로 변환한 뒤, Qwen2-VL 텍스트 인코더를 통해 텍스트 특징 $F_t \in \mathbb{R}^{d_t}$를 추출한다.
- **이미지 입력:** 차량 카메라의 RGB 프레임($224 \times 224$ 해상도)을 Qwen2-VL 내부의 CNN/ViT 백본을 통해 시각 특징 $F_v \in \mathbb{R}^{d_v}$로 추출한다.
- **데이터 통합:** 두 특징을 결합하여 $F = [F_v; F_t] \in \mathbb{R}^{d_v + d_t}$ 형태의 통합 입력을 구성하고 투영 층(projection layer)을 통해 차원을 맞춘다.

### 2. Multimodal Feature Fusion with LoRA

모델의 모든 파라미터를 튜닝하는 대신, Qwen2-VL의 self-attention 모듈에 LoRA를 적용한다.

- **LoRA 원리:** 원래 가중치 행렬 $W \in \mathbb{R}^{d \times d}$에 대해, 낮은 랭크의 두 행렬 $A \in \mathbb{R}^{r \times d}$와 $B \in \mathbb{R}^{d \times r}$ ($r \ll d$, 본 논문에서는 $r=16$)의 곱을 더해 업데이트한다.
- **수식:** 업데이트된 가중치 $W'$는 다음과 같이 계산된다.
  $$W' = W + \alpha \cdot \Delta W, \quad \text{where } \Delta W = BA$$
  (여기서 $\alpha=32$는 스케일링 인자이다.)
- **출력 생성:** 융합된 특징은 3개의 은닉층(512, 256, 128 유닛)을 가진 MLP를 통과하여 7차원 벡터 $[x, y, z, l, w, h, \theta]$를 예측한다. 이는 각각 3D 중심 좌표, 크기, Yaw 각도를 의미한다.

### 3. Two-Stage Loss Optimization

시맨틱 정렬과 기하학적 정밀도를 단계적으로 학습시키기 위해 다음과 같은 전략을 사용한다.

**Stage 1: Preliminary Feature Alignment (1~50 Epochs)**

- 목표: 입력 프롬프트와 이미지의 시맨틱 관계를 빠르게 학습하여 초기 수렴을 안정화하는 것이다.
- 방법: 예측된 바운딩 박스와 정답(Ground-truth) 바운딩 박스를 각각 투영 헤드를 통해 시맨틱 특징 공간 $\mathbb{R}^{128}$으로 보낸 후 MSE(Mean Squared Error) 손실을 계산한다.
  $$L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} \| f_{pred}^i - f_{gt}^i \|^2$$

**Stage 2: Refined Bounding Box Alignment (51~100 Epochs)**

- 목표: 3D 바운딩 박스의 위치와 크기를 정밀하게 조정하는 것이다.
- 방법: 3D IoU(Intersection over Union) 손실을 도입하여 예측 박스와 정답 박스의 겹침 정도를 직접 최적화한다.
  $$IoU_i = \frac{P_i \cap G_i}{P_i \cup G_i}, \quad L_{IoU} = \frac{1}{N} \sum_{i=1}^{N} (1 - IoU_i)$$

**최종 손실 함수:**
$$L = \lambda_1 L_{MSE} + \lambda_2 L_{IoU}$$

- **Stage 1:** $\lambda_1 = 1.0, \lambda_2 = 0.0$
- **Stage 2:** $\lambda_1 = 0.2, \lambda_2 = 0.8$

### 4. Open-Set Strategy

학습 데이터에 포함되지 않은 새로운 카테고리(유모차, 동물, 건설 장비 등)에 대해서도 VLM의 일반화된 표현 능력을 활용하여 3D 바운딩 박스를 생성하도록 설계하였다. 평가 시에는 학습되지 않은 클래스들에 대해 mIoU를 측정하여 일반화 성능을 검증한다.

## 📊 Results

### 실험 설정

- **데이터셋:** nuScenes (보스턴 및 싱가포르 도시 환경, 1000개 씬).
- **평가 지표:** Accuracy, Recall, F1 Score, mIoU.
- **하드웨어:** AMD EPYC 7642 CPU, NVIDIA A100 GPU (40GB).

### 주요 결과

1. **정량적 성능 향상:** 제안된 Joint Semantic-Geometric Loss 설계가 기존 방식 대비 **인식 정확도를 12.8% 향상**시켰음을 확인하였다.
2. **학습 단계별 분석 (Table 1):**
   - Stage 1에서는 Accuracy가 30.2%에서 최대 70.5%까지 빠르게 상승하지만, 33 Epoch 이후부터 성능 변동이 발생한다.
   - Stage 2(Joint Loss) 도입 후, Accuracy는 최대 70.3%까지 다시 상승하며 안정화되었고, 특히 Recall이 개선되어 False Negative(미탐지)가 감소하였다.
3. **카테고리별 IoU (Table 2):**
   - 훈련 세트 mIoU 0.2344, 테스트 세트 mIoU 0.2257로 일관된 성능을 보였다.
   - 휠체어(wheelchair), 건설 노동자(construction worker) 등 희소 카테고리에서도 강건한 탐지 성능을 보였으나, 경찰차(police)나 트레일러(trailer)와 같이 기하학적으로 복잡하거나 데이터가 매우 적은 객체는 상대적으로 낮은 IoU를 기록하였다.
4. **정성적 분석 (Figure 2):** 건설 장비, 동물, 성인 등 학습되지 않은 Open-set 카테고리에 대해서도 적절한 3D 바운딩 박스를 생성하는 것을 확인하여 일반화 능력을 입증하였다.

## 🧠 Insights & Discussion

**강점 및 성과:**
본 연구는 VLM을 단순한 보조 도구가 아닌 3D 인식의 주체로 사용하여 End-to-End 구조를 구현했다는 점에서 큰 의미가 있다. 특히 LoRA를 통해 모델의 0.1% 파라미터만 튜닝함으로써, 거대 모델의 지식을 유지하면서도 자율주행이라는 특수 도메인에 효율적으로 적응시켰다. 또한, MSE 손실로 시맨틱 뼈대를 잡고 IoU 손실로 기하학적 정밀도를 다듬는 2단계 전략은 VLM이 연속적인 3D 공간 값을 예측할 때 겪는 불안정성을 효과적으로 해결하였다.

**한계 및 논의사항:**

- **기하학적 복잡성:** 일부 객체(트레일러 등)에서 IoU가 낮게 나타난 것은, VLM이 이미지의 2D 정보만으로는 매우 긴 객체의 정확한 3D 깊이와 길이를 추론하는 데 여전히 한계가 있음을 시사한다.
- **데이터 의존성:** nuScenes 데이터셋을 사용하였으나, 실제 도로의 훨씬 더 다양한 엣지 케이스(edge cases)에 대해 어느 정도의 강건성을 가질지는 추가 검증이 필요하다.
- **실시간성:** LoRA로 경량화하였음에도 불구하고 Qwen2-VL과 같은 거대 모델을 임베디드 보드에서 실시간(예: 10Hz 이상)으로 구동하기 위해서는 추가적인 양자화(Quantization)나 가속화 기법이 필수적일 것이다.

## 📌 TL;DR

VLM-3D는 Qwen2-VL을 기반으로 **End-to-End 3D Open-set Perception**을 수행하는 프레임워크이다. **LoRA**를 통해 효율적으로 모델을 튜닝하고, **시맨틱 정렬(MSE) $\rightarrow$ 기하학적 정밀화(IoU)**로 이어지는 2단계 손실 함수를 통해 학습되지 않은 객체에 대해서도 강건한 3D 바운딩 박스 예측 능력을 갖추었다. 이 연구는 기존의 다단계 파이프라인에서 발생하는 오차 누적 문제를 해결하고, VLM의 제로샷 능력을 3D 자율주행 인식에 직접적으로 통합했다는 점에서 향후 실시간 오픈월드 인식 시스템 연구에 중요한 이정표가 될 가능성이 높다.
