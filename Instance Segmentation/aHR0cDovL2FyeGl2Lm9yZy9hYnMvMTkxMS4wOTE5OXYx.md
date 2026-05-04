# Object-Guided Instance Segmentation for Biological Images

Jingru Yi, Hui Tang, Pengxiang Wu, Bo Liu, Daniel J. Hoeppner, Dimitris N. Metaxas, Lianyi Han, Wei Fan (2020)

## 🧩 Problem to Solve

생물학적 이미지(Biological Images)에서 개체 분할(Instance Segmentation)은 세포의 상호작용, 핵의 치료 반응, 식물 표현형 분석 등 객체의 행동과 속성을 연구하는 데 필수적인 단계이다. 그러나 생물학적 이미지는 다음과 같은 특유의 난제들을 가지고 있다.

1. **객체의 밀집 및 접착(Clustering, Adhesion, Occlusion):** 객체들이 서로 겹쳐 있거나 밀접하게 붙어 있어 개별 인스턴스를 구분하기 어렵다.
2. **세밀한 구조의 중요성:** 세포의 돌출부(Protrusions)나 잎의 줄기(Leaf stalking)와 같은 매우 미세한 디테일을 정확하게 캡처해야 한다.

기존의 **Box-free 방식**은 로컬 픽셀 정보에 의존하여 전역적인 시야가 부족하므로 과분할(Over-segmentation) 또는 과소분할(Under-segmentation)이 발생하기 쉽다. 반면, **Box-based 방식**은 객체 탐지를 결합하여 개별 인스턴스 식별 능력은 좋으나, 고정된 크기의 RoI(Region of Interest) 패치를 사용함으로써 세밀한 디테일을 손실하거나 앵커 박스(Anchor box)의 불균형 문제로 인해 학습 효율 및 작은 객체 탐지 성능이 떨어지는 한계가 있다.

본 논문의 목표는 작은 객체까지 효과적으로 탐지하면서, 밀집된 객체들 사이에서 타겟 객체의 세밀한 디테일을 유지하며 분할할 수 있는 새로운 Box-based 인스턴스 분할 방법을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **중심점 기반의 객체 탐지**와 **객체 가이드 기반의 분할 브랜치**를 결합하는 것이다.

- **중심점 기반 위치 추정:** 복잡한 앵커 박스나 다수의 키포인트 그룹화 대신, 객체의 중심점(Center point)만을 탐지하여 작은 객체에 대한 탐지 성능을 높이고 계산 효율성을 확보하였다.
- **Object-Guided Segmentation:** 탐지 브랜치에서 얻은 객체 특징(Object features)을 분할 브랜치의 가이드로 재사용하여, RoI 패치 내에서 밀집된 인스턴스들을 효과적으로 분리한다.
- **Instance Normalization (IN) 적용:** Instance Normalization을 통해 RoI 내의 주변 객체 분포를 억제하고 타겟 객체의 분포를 회복함으로써, 인접한 객체의 간섭을 최소화하고 형태적 디테일을 보존한다.

## 📎 Related Works

논문에서는 인스턴스 분할 방법을 크게 두 가지 카테고리로 분류하여 설명한다.

1. **Box-free Instance Segmentation:**
   - **DCAN, Deep Watershed:** 객체의 윤곽선이나 에너지 맵을 분석하여 분할하지만, 경계가 불분명한 경우 과분할 문제가 발생하며 세밀한 디테일을 잃기 쉽다.
   - **Cosine Embedding:** 픽셀 임베딩 클러스터링을 사용하지만, 클러스터링 실패 시 조각난 형태의 분할 결과가 생성되는 경향이 있다.
   - **StarDist:** Star-convex 다각형을 사용하여 형태 정보를 활용하지만, 볼록한(Convex) 형태의 객체에만 적용 가능하다는 한계가 있다.

2. **Box-based Instance Segmentation:**
   - **Mask R-CNN:** FPN과 RoIAlign을 사용하지만, 고정된 크기의 RoI 패치(예: $14 \times 14$)로 인해 세포 돌출부와 같은 미세한 디테일을 캡처하지 못하며, 앵커 박스의 양성/음성 불균형 문제로 학습이 느리고 성능이 최적화되지 않는 문제가 있다.
   - **Keypoint Graph:** 4개의 모서리와 1개의 중심점을 탐지하여 그룹화한다. 하지만 작은 객체의 경우 키포인트 간의 거리가 가까워 원형 영역이 겹치게 되므로 탐지에 실패하는 경우가 많다.

## 🛠️ Methodology

제안된 프레임워크는 ResNet50을 인코더로 사용하며, 크게 **객체 탐지 브랜치(Object Detection Branch)**와 **객체 가이드 분할 브랜치(Object-Guided Segmentation Branch)**의 두 부분으로 구성된다.

### 1. Object Detection Branch
객체를 중심점(Center point)을 통해 직접 위치 추정한다. 출력값은 세 가지 맵으로 구성된다.

- **Center Heatmap:** 객체의 중심 위치를 예측한다. 배경 픽셀에 대한 페널티를 줄이기 위해 가우시안 원형 주변의 페널티를 완화한 Variant Focal Loss를 사용한다.
  $$L_{hm} = -\frac{1}{N} \begin{cases} (1-p_i)^\alpha \log(p_i) & \text{if } y_i=1 \\ (1-y_i)^\beta (p_i)^\alpha \log(1-p_i) & \text{otherwise} \end{cases}$$
  여기서 $\alpha=2, \beta=4$를 사용하며, $3 \times 3$ max-pooling을 통한 NMS(Non-Maximum Suppression)로 최종 중심점을 결정한다.
- **Offset Map:** 다운샘플링된 맵에서 원래 이미지 좌표로 복원하기 위한 오프셋을 예측하며, 이상치에 강건한 $L_1$ loss를 사용한다.
  $$o_i = \left( \frac{x_i}{n} - b_{x_i n e}, \frac{y_i}{n} - b_{y_i n e} \right)$$
- **Width-Height Map:** 중심점에서 객체의 너비와 높이를 직접 회귀(Regression)하며, 역시 $L_1$ loss를 사용한다.

### 2. Object-Guided Segmentation Branch
탐지 브랜치에서 예측된 Bounding Box를 이용하여 인코더의 특징 맵에서 RoI 패치를 크롭(Crop)한 후 분할을 수행한다.

- **특징 재사용:** 인코더의 얕은 층(Layer 0-1)에서는 형태적 디테일을, 깊은 층(Layer 2-4)에서는 객체 가이드 정보를 가져온다. 객체 특징(Object features)은 RoI 내에서 밀집된 객체들을 분리하는 가이드 역할을 한다.
- **Instance Normalization (IN):** 객체 특징만 사용할 경우 마스크가 불완전해지는 문제가 발생한다. 이를 해결하기 위해 IN을 적용하여 주변 객체의 통계적 분포를 제거하고 타겟 객체의 특성을 강조한다.
  $$x'_{h,w} = \gamma \left( \frac{x_{h,w} - \mu}{\sigma} \right) + \beta$$
  여기서 $\mu, \sigma$는 RoI 패치의 평균과 표준편차이며, $\gamma, \beta$는 학습 가능한 스케일링 인자이다.
- **학습 목표:** 분할 작업에는 Binary Cross-Entropy (BCE) loss를 사용하여 모델을 최적화한다.

## 📊 Results

### 실험 설정
- **데이터셋:** DSB2018(세포핵), Plant Phenotyping(식물), Neural Cell(신경세포)의 세 가지 생물학적 데이터셋을 사용하였다.
- **평가 지표:** $\text{AP}_{\text{box}}$ (탐지 성능), $\text{AP}_{\text{mask}}$ (분할 성능), $\text{AIoU}_{\text{mask}}$ (분할 품질)를 측정하였다. $\text{AP}$는 $\text{IoU}$ 임계값 0.5에서 0.95까지 0.05 간격으로 평균을 내어 계산하였다.

### 주요 결과
- **정량적 성능:** 제안 방법($\text{Ours-objBranchIN}$)은 모든 데이터셋에서 비교 대상(DCAN, Mask R-CNN, Cosine Embedding, Keypoint Graph)보다 우수한 $\text{AP}_{\text{mask}}$와 $\text{AIoU}_{\text{mask}}$를 기록하였다. 특히 신경세포 데이터셋처럼 구조가 길고 복잡한 경우 기존 방식 대비 성능 향상이 뚜렷했다.
- **추론 속도:** NVIDIA GeForce GTX 1080 GPU 기준, 기존 Box-based 방식들보다 빠른 FPS를 기록하며 효율성을 입증하였다.
- **정성적 결과:** 제안 방법은 작은 객체를 정확히 식별할 뿐만 아니라, 신경세포의 돌출부나 식물의 잎 줄기와 같은 세밀한 디테일을 보존하면서 서로 붙어 있는 객체들을 효과적으로 분리해냈다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **객체 특징과 IN의 시너지:** Ablation Study를 통해 객체 특징만 사용할 경우(Ours-objBranch) 분할 마스크가 불완전해지고, IN만 사용할 경우(Ours-sepBranchIN) 밀집된 객체 분리 능력이 떨어진다는 것을 확인하였다. 두 요소를 결합했을 때 비로소 주변 노이즈를 억제하고 타겟의 디테일을 완벽하게 회복할 수 있었다.
- **멀티모달 분포 처리:** RoI 내에서 타겟과 주변 객체가 섞여 있을 때, 모델이 IN의 정도와 채널별 가중치를 학습함으로써 타겟의 지배적인 분포를 하이라이트하고 주변 분포를 억제할 수 있음을 시사한다.

### 한계 및 가정
- **BBox 정확도 의존성:** 본 방법은 탐지된 Bounding Box가 타겟 객체를 타이트하게 감싸고 있다는 가정하에 작동한다.
- **잠재적 실패 사례:** 만약 Bounding Box가 너무 크게 예측되어 내부에 동일한 크기의 여러 객체가 포함될 경우, 타겟 객체를 단독으로 분리해내는 데 실패할 가능성이 있다. 다만, 배경과 타겟만 포함된 과추정 BBox의 경우에는 분류 정보가 포함되어 있어 강건하게 작동한다.

## 📌 TL;DR

본 논문은 생물학적 이미지의 특성인 객체 밀집 및 미세 구조 보존 문제를 해결하기 위해 **중심점 기반 탐지**와 **객체 가이드 기반 분할**을 결합한 프레임워크를 제안한다. 특히 **Instance Normalization**을 통해 RoI 내의 주변 객체 간섭을 제거하고 타겟의 디테일을 회복함으로써, 기존 SOTA 모델들보다 정확하고 빠르게 밀집된 생물학적 개체들을 분할해낸다. 이 연구는 정밀한 세포 분석이나 식물 표현형 분석과 같이 고도의 세밀함이 요구되는 바이오 이미지 분석 분야에 매우 유용한 도구가 될 것으로 보인다.