# S4M: Boosting Semi-Supervised Instance Segmentation with SAM

Heeji Yoon, Heeseong Shin, Eunbeen Hong, Hyunwook Choi, Hansang Cho, Daun Jeong, Seungryong Kim (2025)

## 🧩 Problem to Solve

본 논문은 레이블링된 데이터가 매우 부족한 상황에서 인스턴스 분할(Instance Segmentation) 성능을 높이기 위한 준지도 학습(Semi-supervised Learning, SSL) 문제를 다룬다. 

일반적인 Teacher-Student 프레임워크는 Teacher 네트워크가 생성한 Pseudo-label을 통해 Student 네트워크를 학습시킨다. 그러나 학습 데이터가 제한적일 경우, Teacher 네트워크가 생성하는 Pseudo-label의 품질이 낮아 전체적인 성능 향상이 제한되는 문제가 발생한다. 특히 저자들은 분석을 통해, 기존 방식의 병목 현상이 클래스 분류(Classification)보다는 마스크의 품질, 즉 객체의 정확한 위치를 찾아내는 Localization 능력의 부족에 있음을 발견하였다. 구체적으로는 여러 인스턴스를 하나의 마스크로 묶어버리는 **Under-segmentation** 문제가 주요 원인으로 지목된다.

따라서 본 논문의 목표는 강력한 세그멘테이션 능력을 가진 Segment Anything Model(SAM)을 준지도 인스턴스 분할 프레임워크에 효율적으로 통합하여, SAM의 정밀한 Localization 능력을 활용하면서도 SAM의 단점인 클래스 정보 부재(Class-agnostic)와 **Over-segmentation**(객체를 너무 잘게 쪼개는 현상) 문제를 극복하는 것이다.

## ✨ Key Contributions

본 논문은 SAM을 직접 적용하는 대신, 지식 증류(Knowledge Distillation)와 Pseudo-label 정제, 그리고 데이터 증강 기법을 통해 SAM의 이점만을 취하는 $S^4M$ 프레임워크를 제안한다. 핵심 아이디어는 다음과 같다.

1.  **Structural Distillation (SD):** SAM을 Meta-teacher로 설정하여 Teacher 네트워크에 SAM의 공간적 구조 이해 능력을 전수한다. 이때 단순한 특징값(Feature) 전이가 아닌, Self-similarity matrix를 통한 구조적 레이아웃을 학습시켜 Over-segmentation을 방지하고 Localization 능력을 강화한다.
2.  **Pseudo-label Refinement (PR):** Teacher가 생성한 노이즈 섞인 Pseudo-label을 SAM의 정밀한 마스크 생성 능력을 이용해 정제한다. 이때 확률 기반의 stochastic point sampling을 사용하여 SAM의 Over-segmentation 경향을 억제한다.
3.  **Augmentation with Refined Pseudo-labels (ARP):** 정제된 고품질 Pseudo-label을 활용하여 인스턴스 단위로 이미지를 합성하는 ARP(Augmentation with Refined Pseudo-label) 기법을 도입, Student 네트워크의 강건성(Robustness)을 높인다.

## 📎 Related Works

- **Semi-supervised Instance Segmentation:** 기존의 Noisy Boundaries, Polite Teacher, PAIS, GuidedDistillation 등은 주로 Teacher-Student 구조와 일관성 규제(Consistency Regularization)를 사용하였다. 하지만 이들은 모두 제한된 레이블 데이터로 인해 Pseudo-label의 품질이 낮다는 근본적인 한계를 가지고 있다.
- **Segment Anything Model (SAM):** SAM은 프롬프트 기반의 강력한 제로샷 세그멘테이션 성능을 보여주지만, 클래스 정보를 제공하지 않으며 객체를 너무 세분화하여 분할하는 경향이 있어 인스턴스 분할 작업에 직접 적용하기에는 무리가 있다.
- **Knowledge Distillation (KD):** 기존 KD 연구들은 주로 세만틱 세그멘테이션에서 공간적 관계를 전수하는 데 집중했다. 인스턴스 분할에서는 각 객체를 구분해야 하므로 더 복잡한 구조적 정보의 전수가 필요하며, 본 논문은 SAM의 특성을 고려해 '무엇을 배울 것인가'와 '무엇을 배제할 것인가'를 설계함으로써 차별점을 둔다.

## 🛠️ Methodology

$S^4M$ 프레임워크는 크게 두 단계로 구성된다. 1단계에서는 Teacher 네트워크를 사전 학습시키고, 2단계에서는 Student 네트워크를 준지도 학습 방식으로 훈련시킨다.

### 1. Structural Distillation (SD)을 통한 Teacher 강화
Teacher 네트워크가 Under-segmentation 문제를 해결하도록 SAM의 구조적 지식을 증류한다. 특징 맵의 단순 거리 최소화는 SAM의 Over-segmentation 성향까지 학습시킬 위험이 있으므로, 본 논문은 **Self-similarity matrix**를 활용한다.

Teacher 모델과 SAM 모델의 디코더에서 각각 특징 맵 $F_T, F_{SAM} \in \mathbb{R}^{d \times H' \times W'}$를 추출한 후, 각 특징 간의 코사인 유사도를 계산하여 다음과 같이 Self-similarity matrix를 생성한다.

$$C_{SAM} = \frac{F_{SAM} \cdot F_{SAM}^T}{\|F_{SAM}\| \|F_{SAM}^T\|}, \quad C_T = \frac{F_T \cdot F_T^T}{\|F_T\| \|F_T^T\|}$$

이후, 두 행렬 간의 차이를 Huber 함수 $\rho$를 이용한 손실 함수로 정의하여 학습한다.

$$L_{SD} = \frac{1}{H'W'} \sum_{i} \rho(C_{SAM}(i) - C_T(i))$$

최종 Teacher 학습 목적 함수는 $L_T = L_{lb} + L_{SD}$가 된다. 여기서 $L_{lb}$는 레이블 데이터에 대한 지도 학습 손실이다.

### 2. Pseudo-label Refinement (PR)
Teacher가 생성한 Pseudo-label $\hat{m}_u^k$가 부정확할 경우, 이를 SAM의 프롬프트로 입력하여 정제한다. 단순히 마스크의 중심점을 사용하는 대신, 마스크 내 픽셀 확률 분포 $\tilde{p}(a,b)$에서 $K$개의 점을 무작위로 샘플링하여 SAM에 입력함으로써 더 정교하고 안정적인 마스크를 얻는다.

$$\tilde{p}(a,b) = \frac{p(a,b)}{\sum p(a,b)}, \quad \text{where } p(a,b) = \begin{cases} \tilde{m}_u^k, & \text{if } \hat{m}_u^k(a,b)=1 \\ 0, & \text{if } \hat{m}_u^k(a,b)=0 \end{cases}$$

### 3. Augmentation with Refined Pseudo-labels (ARP)
정제된 Pseudo-label을 사용하여 두 이미지 $x_A, x_B$ 사이에서 인스턴스를 서로 교차하여 붙이는(paste) 합성 이미지를 생성한다.

$$x_{AB} \leftarrow M_B \odot x_B + (1 - M_B) \odot x_A$$
$$x_{BA} \leftarrow M_A \odot x_A + (1 - M_A) \odot x_B$$

여기서 $M_A, M_B$는 정제된 Pseudo-mask들의 합집합이다. 이를 통해 Student 네트워크는 다양한 컨텍스트와 폐색(Occlusion) 상황을 학습하여 일반화 성능을 높인다.

## 📊 Results

### 실험 설정
- **데이터셋:** Cityscapes (5%, 10%, 20%, 30% 레이블 사용), COCO (1%, 2%, 5%, 10% 레이블 사용).
- **지표:** mask-AP (Average Precision).
- **베이스라인:** Mask2Former 기반의 GuidedDistillation 등 최신 준지도 학습 모델.

### 정량적 결과
- **Cityscapes:** 모든 레이블 비율에서 기존 SOTA 모델들을 상회한다. 특히 5% 레이블 설정에서 기존 최고 성능 대비 **6.9 point AP**라는 압도적인 성능 향상을 보였다.
- **COCO:** 1%와 2%의 매우 적은 레이블 설정에서 각각 **1.9, 1.8 point AP** 향상을 기록하며 SOTA를 달성하였다. (단, 10% 설정에서는 Pseudo-label의 원래 품질이 이미 높기 때문에 SAM 정제 과정의 무작위성으로 인해 약간의 성능 하락이 관찰되었다.)

### 정성적 결과
시각화 결과, 기존 GuidedDistillation 방식이 동일 클래스의 여러 객체를 하나의 마스크로 묶는 경향이 강했던 반면, $S^4M$은 개별 인스턴스를 훨씬 더 명확하게 구분하고 경계선을 정밀하게 추출함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 SAM이라는 거대 모델을 준지도 학습에 통합할 때 발생할 수 있는 **'Under-segmentation(Teacher의 문제)'**과 **'Over-segmentation(SAM의 문제)'** 사이의 균형을 맞추는 것이 핵심임을 보여주었다.

- **구조적 증류의 효과:** 단순한 Feature Distillation보다 Self-similarity matrix를 통한 Structural Distillation이 더 효과적이었으며, 특히 Encoder보다 Decoder 특징을 사용할 때 Localization 성능이 더 좋았다. 이는 SAM의 디코더가 프롬프트를 통해 더 구체적인 객체 구조를 생성하기 때문이다.
- **상호 보완적 설계:** 실험 결과 SD, PR, ARP 세 가지 요소가 결합되었을 때 최고의 성능이 나타났다. 특히 ARP는 정제되지 않은 Pseudo-label을 사용할 때는 오히려 성능을 떨어뜨렸으나, PR을 통해 정제된 레이블을 사용할 때는 시너지 효과를 내어 성능을 크게 끌어올렸다.
- **한계점:** 레이블 비율이 높아질수록(예: COCO 10%) SAM을 통한 정제 과정이 오히려 노이즈로 작용할 수 있다는 점이 발견되었다. 이는 데이터가 충분할 때는 Teacher의 예측이 이미 충분히 정확하므로, SAM의 개입 정도를 조절하는 적응형 메커니즘이 필요함을 시사한다.

## 📌 TL;DR

$S^4M$은 SAM의 강력한 Localization 능력을 준지도 인스턴스 분할에 도입한 프레임워크이다. **Structural Distillation**으로 Teacher의 Under-segmentation을 해결하고, **Pseudo-label Refinement**와 **ARP 증강**을 통해 Student의 학습 효율을 극대화하였다. 결과적으로 매우 적은 양의 레이블 데이터만으로도 기존 SOTA를 뛰어넘는 성능을 달성하였으며, 이는 Foundation Model의 지식을 효율적으로 전이하는 효과적인 방법론을 제시한 연구라고 평가할 수 있다.