# Object Detection Free Instance Segmentation With Labeling Transformations

Long Jin, Zeyu Chen, Zhuowen Tu (2016)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 기존의 Instance Segmentation 방법론들이 대부분 Object Detection 단계(Bounding Box 예측)에 과도하게 의존하고 있다는 점이다. 저자들은 이러한 Detection 기반 방식이 다음과 같은 세 가지 주요 한계를 가진다고 주장한다.

첫째, Bounding Box 예측과 픽셀 단위의 Masking은 서로 성격이 다른 작업임에도 불구하고 이를 결합함으로써 학습과 테스트 과정의 복잡성이 증가한다. 둘째, Detection 기반 방법은 객체가 직사각형 형태의 Box로 타이트하게 둘러싸여 있다는 강한 가정을 전제로 한다. 따라서 변형이 심한 비정형 객체(예: 의료 영상의 Gland)의 경우 성능이 크게 저하된다. 셋째, 여러 객체가 하나의 Bounding Box 내에 겹쳐서 나타날 때 정밀한 분할이 어렵고 시스템의 투명성이 떨어진다.

특히, Instance Segmentation의 본질적인 어려움은 'Quotient Space' 문제에 있다. 이는 서로 다른 인스턴스에 부여된 ID(라벨)를 서로 바꾸더라도 결과적으로는 동일한 분할 결과가 된다는 특성이다. 즉, 인스턴스 ID 자체는 직접적인 의미가 없으므로, 이를 일반적인 분류(Classification)나 회귀(Regression) 문제로 직접 다루기 어렵다. 따라서 본 논문의 목표는 Object Proposal 및 Detection 단계 없이, 이 Quotient Space 문제를 해결할 수 있는 Labeling Transformation 방법을 통해 Instance Segmentation을 수행하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스 라벨링을 직접 예측하는 대신, CNN이 학습 가능한 형태의 새로운 표현으로 변환하는 **Labeling Transformation**을 도입하는 것이다. 이를 통해 Object Detection 과정 없이 Fully Convolutional Network(FCN) 기반의 segmentation만으로 인스턴스를 분리할 수 있는 프레임워크를 제안한다.

저자들은 Quotient Space 문제를 해결하기 위해 세 가지 대안적인 변환 방법을 제시한다:

1. **Pixel-based Affinity Mapping**: 픽셀 간의 국소적 유사성 패턴을 학습하여 인스턴스를 구분한다.
2. **Superpixel-based Affinity Learning**: 슈퍼픽셀 단위로 유사도를 학습하여 계산 복잡도를 줄이고 구조적 정보를 활용한다.
3. **Boundary-based Component Segmentation**: 인스턴스의 경계(Boundary)를 학습하여 동일 클래스의 연결된 컴포넌트를 분리한다.

## 📎 Related Works

기존의 Instance Segmentation 연구는 크게 두 가지 방향으로 나뉜다.

1. **Detection-based Methods**: DeepMask, Mask R-CNN(본 논문 시점의 유사 연구들)과 같이 Bounding Box를 먼저 찾고 그 내부에서 Mask를 생성하는 방식이다. 이는 일반적인 객체 인식에는 강하나, 앞서 언급한 비정형 객체 처리와 시스템 복잡성 문제가 존재한다.
2. **Segmentation-based Methods**: 픽셀 단위의 밀집 특징(Dense per-pixel features)을 사용하는 방식이다. 하지만 기존의 segmentation 기반 방법들은 단일 클래스만 다루거나(foreground/background), 깊이 정보(Depth)와 같은 추가 데이터가 있어야 하는 등 적용 범위가 제한적이었다.

본 연구는 이러한 기존 방식과 달리, Object Proposal 단계 없이 오직 segmentation 기반의 접근법만으로 PASCAL VOC와 같은 일반적인 데이터셋과 Gland와 같은 텍스처 중심의 데이터셋 모두에서 동작하는 범용적인 프레임워크를 지향하며, 특히 라벨의 임의성(Quotient Space)을 처리하는 변환 방식에 집중한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

전체 시스템은 Figure 2에 묘사된 바와 같이 두 개의 경로(Path)로 구성된 FCN 기반 프레임워크이다.

### 1. Semantic Labeling Path

첫 번째 경로는 각 픽셀의 클래스를 예측하는 표준적인 Semantic Segmentation 작업이다. 본 논문에서는 **DeepLab-Large-FOV** 네트워크 구조를 기본으로 사용하며, Atrous Convolution을 통해 수용 영역(Receptive Field)을 확장한다. 이 단계의 결과물인 Semantic Labeling 맵에서 동일 클래스의 연결된 컴포넌트(Connected Component)를 추출하여 잠재적인 인스턴스 후보군을 생성한다.

### 2. Instance Labeling Transformation Path

두 번째 경로는 Quotient Space 문제를 해결하기 위해 인스턴스 라벨을 변환하여 예측하는 경로이다. 세 가지 방법이 제안되었다.

#### Method 1: Pixel-based Affinity Mapping

픽셀 쌍 간의 유사도(Affinity)를 학습하는 방법이다. 전역 유사도 행렬을 계산하는 것은 비용이 너무 크므로, 국소적인 $k \times k$ 패치(본 논문에서는 $5 \times 5$)를 사용한다.

- **변환 과정**: 각 패치 내의 인스턴스 라벨 구성(Configuration)을 기반으로 국소 유사도 행렬을 생성한다. 이때 라벨 ID를 서로 바꾸어도 유사도 행렬은 동일하게 유지되므로 Quotient Space 문제가 해결된다. 생성된 고차원 유사도 행렬을 k-means clustering을 통해 100개의 클래스로 임베딩한다.
- **학습 및 추론**: FCN을 통해 각 픽셀이 어떤 유사도 클래스에 속하는지 분류하도록 학습한다. 추론 시에는 예측된 클래스를 다시 유사도 행렬로 복원하고, 이를 투표(Voting) 방식으로 통합하여 전역 유사도 맵을 구축한다.
- **통합**: Semantic Path에서 얻은 연결 컴포넌트 내 픽셀들에 대해 Normalized Cut(NCuts) 알고리즘을 적용하여 인스턴스를 분리한다.

#### Method 2: Superpixel-based Affinity Learning

픽셀 단위의 계산량을 줄이기 위해 SLIC 알고리즘으로 생성된 슈퍼픽셀(Superpixel) 단위를 사용한다.

- **변환 과정**: 두 슈퍼픽셀이 동일한 인스턴스에 속하면 1, 다르면 0으로 유사도를 정의한다.
- **학습 및 추론**: FCN에서 추출된 각 슈퍼픽셀의 특징 맵(Feature Map)을 결합(Concatenate)하여 두 슈퍼픽셀 간의 유사도를 예측하는 이진 분류기를 학습한다.
- **통합**: Method 1과 마찬가지로, 슈퍼픽셀 간 유사도 행렬을 구축한 후 NCuts 알고리즘을 통해 인스턴스를 분리한다.

#### Method 3: Boundary-based Component Segmentation

라벨 ID를 바꾸어도 인스턴스의 경계선은 변하지 않는다는 점에 착안하여, 경계선(Boundary)을 직접 예측하는 방법이다.

- **변환 과정**: Ground Truth 인스턴스 맵에서 인스턴스 간의 경계선을 추출하여 라벨로 사용한다.
- **학습 및 추론**: **HED(Holistically-Nested Edge Detection)** 모델을 사용하여 인스턴스 경계선을 학습한다. 예측된 경계선 맵에 Non-maximal Suppression(NMS)을 적용하여 얇은 경계선을 얻는다.
- **통합**: Semantic Labeling 결과에서 예측된 경계선 영역에 해당하는 픽셀을 배경(Background)으로 처리한다. 이렇게 하면 하나의 연결된 컴포넌트가 경계선에 의해 여러 개의 분리된 컴포넌트로 쪼개지며, 각각을 개별 인스턴스로 간주한다.

## 📊 Results

실험은 PASCAL VOC 2012(객체 중심)와 MICCAI 2015 Gland Segmentation(텍스처 중심) 데이터셋에서 수행되었다.

### 1. PASCAL VOC 2012

- **평가 지표**: $\text{AP}_r$ (IoU 0.5 기준 평균 정밀도), $\text{AR@N}$ (Recall)
- **결과**:
  - $\text{AP}_r$ 측정 결과, Method 3(Boundary-based)가 가장 우수한 성능($49.9\%$)을 보였으며, 단순 연결 컴포넌트 추출 방식(FE, $45.3\%$)보다 향상되었다.
  - 다만, PFN이나 MPA와 같은 최신 Proposal-based 모델보다는 $\text{AP}_r$ 수치가 낮았다. 저자들은 이를 클러스터링 기반 방법의 인스턴스 개수 추정 어려움과 단순한 scoring 방식 때문이라고 분석한다.
  - $\text{AR@10}$에서는 Proposal 기반 방법들과 경쟁 가능한 수준의 결과를 보였다.

### 2. MICCAI 2015 Gland Segmentation

- **평가 지표**: F1 Score, Object Dice, Object Hausdorff Distance
- **결과**:
  - Detection 기반 방법(MNC 등)보다 월등히 좋은 성능을 보였다. 이는 Gland의 형태가 비정형적이라 Bounding Box 가정이 작동하지 않기 때문이다.
  - 특히 Method 3는 SOTA 방법론들과 대등한 수준의 성능을 달성하였다.
  - 제안된 프레임워크가 객체의 기하학적 형태에 구애받지 않고 유연하게 대응함을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **Object Detection이라는 무거운 제약 조건을 제거하고도 인스턴스 분리가 가능함**을 보여준 점이다. 특히 Boundary-based 방식(Method 3)이 두 데이터셋 모두에서 효율적이었는데, 이는 인스턴스의 고유한 정체성(ID)보다 인스턴스 간의 '분리 지점'을 찾는 것이 학습 관점에서 훨씬 더 명확한 목표가 되기 때문이다.

또한, Texture-centric한 Gland 데이터셋에서의 성과는 Detection-free 접근법의 실용성을 강력하게 뒷받침한다. 기존의 Mask R-CNN 류의 모델들이 정형화된 객체에는 강하지만, 의료 영상과 같이 형태 변형이 심한 도메인에서는 한계가 있을 수 있음을 시사한다.

다만, 한계점으로는 클러스터링 기반 방법(Method 1, 2)에서 최적의 인스턴스 개수를 결정하는 문제가 남아 있으며, PASCAL VOC와 같은 복잡한 배경의 데이터셋에서는 여전히 정교한 Proposal 기반 방법들이 높은 Precision을 유지한다는 점이다.

## 📌 TL;DR

본 논문은 Instance Segmentation에서 발생하는 라벨의 임의성(Quotient Space) 문제를 해결하기 위해, 라벨을 유사도 맵이나 경계선으로 변환하여 학습하는 **Labeling Transformation** 프레임워크를 제안한다. Object Detection 단계 없이 FCN만으로 구현되어 구조가 단순하며, 특히 Bounding Box 가정이 통하지 않는 비정형 객체(Gland 등) 분리에서 기존 Detection 기반 방식보다 뛰어난 성능과 유연성을 보여준다. 이는 향후 비정형 객체 분할이나 단순한 구조의 실시간 인스턴스 분할 연구에 중요한 기초가 될 수 있다.
