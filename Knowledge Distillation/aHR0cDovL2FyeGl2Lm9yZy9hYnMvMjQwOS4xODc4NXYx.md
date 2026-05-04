# Student-Oriented Teacher Knowledge Refinement for Knowledge Distillation

Chaomin Shen, Yaomin Huang, Haokun Zhu, Jinsong Fan, Guixu Zhang (2024)

## 🧩 Problem to Solve

본 논문은 지식 증류(Knowledge Distillation, KD) 과정에서 발생하는 **교사 네트워크(Teacher Network)와 학생 네트워크(Student Network) 사이의 모델 용량 및 아키텍처 설계 차이**로 인한 성능 저하 문제를 해결하고자 한다. 

기존의 KD 방식들은 대부분 '교사 중심(Teacher-oriented)' 패러다임을 따른다. 이는 교사가 가진 복잡하고 정교한 지식을 학생이 그대로 학습하도록 강제하는 방식이다. 그러나 학생 네트워크는 용량이 작기 때문에 교사의 복잡한 인식 패턴을 완전히 이해하기 어렵고, 이로 인해 전송 효율이 떨어지며 결국 최적의 성능(sub-optimal performance)을 달성하지 못하는 한계가 있다.

따라서 본 연구의 목표는 관점을 '학생 중심(Student-oriented)'으로 전환하여, **교사의 지식을 학생의 학습 능력과 구조적 요구에 맞게 정제(Refine)**함으로써 지식 전송의 효율성을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 교사의 지식을 그대로 전달하는 것이 아니라, 학생이 수용 가능한 형태로 동적으로 조정하여 전달하는 것이다. 이를 위해 다음과 같은 두 가지 핵심 구성 요소를 제안한다.

1.  **Differentiable Automatic Feature Augmentation (DAFA):** 신경망 구조 탐색(NAS) 개념을 도입하여, 학생 네트워크의 요구에 가장 적합한 특징 증강(Feature Augmentation) 전략을 자동으로 탐색하고 적용한다. 이를 통해 교사의 지식을 학생이 이해하기 쉬운 형태로 변형한다.
2.  **Distinctive Area Detection Module (DAM):** 교사와 학생 네트워크가 공통적으로 관심을 가지는 '상호 관심 영역'을 식별한다. 불필요한 정보의 전송을 배제하고 핵심적인 영역에 집중하여 지식 증류를 수행함으로써 학습 효율을 높인다.

이 방식은 기존의 다양한 지식 증류 방법론에 결합하여 사용할 수 있는 **플러그인(Plug-in)** 형태로 설계되었다.

## 📎 Related Works

### 기존 지식 증류 연구
지식 증류는 크게 로짓(Logits) 기반과 특징(Feature) 기반 방식으로 나뉜다.
- **Logits-based KD:** 교사와 학생의 출력 분포(soft labels) 간의 KL-Divergence를 최소화하는 방식이다. 하지만 클래스 수준의 정보만 제공하며 입력 데이터의 세부 구조적 정보를 결합하지 못한다는 한계가 있다.
- **Feature-based KD:** 중간 레이어의 특징 맵(Feature map)에 픽셀 수준의 제약 조건을 적용하여 구조적 정보를 전달한다. 그러나 교사와 학생 간의 성능 격차가 클 경우, 학생이 교사의 복잡한 특징을 그대로 복제하는 것이 불가능하다는 점이 한계로 지적된다.

### 데이터 및 특징 증강(Augmentation)
입력 단계의 데이터 증강(Rotation, Mixup 등)은 데이터 분포를 확장하여 일반화 성능을 높인다. 최근에는 잠재 공간(Latent space)에서의 특징 증강이 더 높은 타당성을 가진다는 연구가 진행되었다. 하지만 수동으로 설계된 증강 전략은 많은 그리드 서치(Grid search) 시간이 소요되며, 모든 학생 네트워크에 최적인 전략을 보장할 수 없다.

본 논문은 이러한 한계를 극복하기 위해 **미분 가능한 자동 탐색 방식**을 통해 학생 맞춤형 증강 전략을 찾음으로써 기존의 수동적 접근 방식과 차별화를 둔다.

## 🛠️ Methodology

### 전체 파이프라인
SoKD의 전체 구조는 교사 네트워크의 특징을 DAFA를 통해 학생 맞춤형으로 정제하고, DAM을 통해 핵심 영역을 추출하여 학생 네트워크에 전달하는 흐름을 가진다.

### 1. Differentiable Automatic Feature Augmentation (DAFA)
DAFA는 교사의 고정된 지식을 학생의 필요에 맞게 동적으로 조정하는 모듈이다.

- **특징 탐색 공간(Feature Search Space):** 마스킹(Masking), 노이즈 추가(Adding noise) 등 특징 표현의 강건성을 높이는 일련의 연산들로 구성된다.
- **탐색 전략:** 이산적인 연산 선택 과정을 연속적인 공간으로 변환하기 위해 **Softmax relaxation**을 사용한다. 특정 서브 정책 $\text{s}$가 선택될 확률을 다음과 같이 정의한다.
  $$\bar{\text{s}}(F) = \sum_{\text{s} \in S} \frac{\exp(\alpha_{\text{s}})}{\sum_{\text{s}' \in S} \exp(\alpha_{\text{s}'})} \text{s}(F)$$
  여기서 $\alpha$는 학습 가능한 파라미터 벡터이다.
- **미분 가능성 확보:** Bernoulli 분포를 통한 연산 실행 여부 결정과 $\alpha$의 선택 과정에서 발생하는 비미분 문제를 해결하기 위해 **Gumbel-Softmax reparameterization trick**을 사용한다. 또한, 미분 불가능한 연산의 크기(magnitude) $m$에 대해서는 **Straight-through gradient estimator**를 적용하여 기울기를 추정한다.

### 2. Distinctive Area Detection Module (DAM)
DAM은 교사와 학생이 모두 중요하게 생각하는 영역을 찾아내어 효율적인 전송을 돕는다.

- **구조:** 어댑터(Adapter)를 통해 교사와 학생의 특징을 동일한 세만틱 공간으로 매핑한 후, 3개의 브랜치(Heatmap, Size, Offset 예측)로 구성된 검출 헤드를 통과시킨다.
- **학습 방식:** 교사와 학생이 각각 동일한 파라미터를 공유하는 DAM을 통과하며, 두 네트워크가 예측한 영역이 일치하도록 $L_2$ 손실 함수를 통해 감독 학습한다.
  $$\mathcal{L}_{D} = \left\| \text{conv}_i(\Phi_{\text{s}}(F_{\text{s}})) - \text{conv}_i(\bar{\text{s}}(F_{\text{t}})) \right\|^2$$
- **영역 필터링:** 식별된 핵심 영역 $\mathcal{A}_i$에 대해 마스크 연산 $M$을 적용하여, 해당 영역의 특징만을 증류 과정에 사용한다.
  $$\mathcal{L}_{DA}(F_{\text{s}}, F_{\text{t}}) = \sum_{i=1}^{P} \left\| M(\mathcal{A}_i)\Phi_{\text{s}}(F_{\text{s}}) - M(\mathcal{A}_i)\bar{\text{s}}(F_{\text{t}}) \right\|^2$$

### 3. 최적화 목표 및 학습 절차
본 연구는 Bi-level 최적화 문제를 푼다. 증강 전략 파라미터 $\chi = \{\alpha, \beta, m\}$와 학생 네트워크 가중치 $\theta$를 동시에 최적화한다.

- **증강 손실($\mathcal{L}_{aug}$):** 정제된 교사의 특징이 학생의 특징과 최대한 가까워지도록 유도하는 일관성 손실을 사용한다.
  $$\mathcal{L}_{aug} = \frac{1}{2} \left( \text{s}(f_{\text{t}}(x)) - f_{\text{s}}(x) \right)^2$$
- **최종 손실 함수:** 학생 네트워크는 태스크 손실, 일반 특징 증류 손실, 그리고 DAM 기반의 영역 특화 손실을 함께 최소화한다.
  $$\min_{\theta} (\mathcal{L}_{task} + \lambda \mathcal{L}_{D} + \beta \mathcal{L}_{DA})$$

## 📊 Results

### 실험 설정
- **데이터셋:** CIFAR-100, ImageNet, MS-COCO.
- **비교 대상:** FitNet, CRD, AT, ReviewKD 등의 대표적인 Feature-based KD 방법론.
- **평가 지표:** Top-1/Top-5 Accuracy (분류), Average Precision (AP) (객체 검출).

### 주요 결과
1.  **CIFAR-100:** 동일 구조(Homogeneous) 및 서로 다른 구조(Heterogeneous)의 네트워크 쌍 모두에서 성능 향상을 보였다. 특히 FitNet의 경우 ResNet50-MobileNetV2 쌍에서 Top-1 정확도가 3.96%p 상승하는 등 큰 폭의 개선이 있었다.
2.  **ImageNet:** 대규모 데이터셋에서도 유효함이 입증되었다. ResNet34(교사)-ResNet18(학생) 조합에서 SOTA 모델인 ReviewKD의 성능을 71.61%에서 72.53%로 끌어올렸다.
3.  **MS-COCO (Object Detection):** Faster-RCNN-FPN 백본을 사용하여 실험한 결과, 모든 baseline 방법론에 SoKD를 추가했을 때 AP 및 $AP_{50}$ 지표가 일관되게 상승하였다.

### 분석 결과
- **Ablation Study:** DAFA와 DAM, 그리고 $\mathcal{L}_{aug}$가 각각 독립적으로 성능 향상에 기여함을 확인하였다.
- **시각화 (Grad-CAM):** SoKD를 적용한 학생 네트워크가 기존 방법보다 교사의 인식 패턴(Attention region)을 훨씬 더 유사하게 학습함을 확인하였다.
- **특징 분포 분석:** PCA 차원 축소와 KS-test(p-value: 0.26) 결과, 특징 증강 이후에도 전체적인 분포는 유지되면서 특징의 다양성은 증가하고 이상치(Outlier)는 감소하여 학습이 용이해졌음을 보였다.

## 🧠 Insights & Discussion

본 논문은 기존의 지식 증류가 가졌던 '강제적 학습'의 한계를 '학생 맞춤형 정제'라는 관점으로 해결하였다는 점에서 강점이 있다. 특히, 교사의 파라미터를 수정하지 않고 특징 공간에서의 증강(Feature Augmentation)과 자동 탐색(NAS)을 결합하여, 교사의 원본 지식을 훼손하지 않으면서도 학생이 수용 가능한 형태로 지식을 변형했다는 점이 매우 영리한 설계이다.

또한, DAM 모듈을 통해 불필요한 정보 전송을 막고 상호 관심 영역에 집중한 것은, 모델 간의 용량 차이가 클 때 발생할 수 있는 '노이즈 전송' 문제를 효과적으로 억제한 것으로 해석된다.

다만, Bi-level 최적화와 Gumbel-Softmax 등의 기법이 도입됨에 따라 단순한 KD보다 학습 복잡도가 증가했을 가능성이 있으며, 탐색 에포크(Search Epochs) 수에 따라 오버피팅 위험이 존재한다는 점이 언급되었다. 이는 실제 적용 시 적절한 탐색 주기 설정이 중요함을 시사한다.

## 📌 TL;DR

본 논문은 교사의 지식을 학생의 능력에 맞게 조정하는 **학생 중심(Student-Oriented) 지식 증류 방법론인 SoKD**를 제안한다. **DAFA**를 통해 최적의 특징 증강 전략을 자동으로 탐색하여 교사의 지식을 정제하고, **DAM**을 통해 핵심 관심 영역만을 선별적으로 전송한다. 이 방법은 기존 KD 알고리즘에 플러그인 형태로 추가 가능하며, CIFAR, ImageNet, COCO 등 다양한 벤치마크에서 성능 향상을 입증하였다. 모델 간의 용량 차이가 큰 상황에서 효율적인 지식 전송을 가능케 하여, 향후 경량 모델 설계 및 배포 연구에 중요한 기여를 할 것으로 보인다.