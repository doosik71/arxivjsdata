# Circle Representation for Medical Instance Object Segmentation

Juming Xiong, Ethan H. Nguyen, Yilin Liu, Ruining Deng, Regina N Tyree, Hernan Correa, Girish Hiremath, Yaohong Wang, Haichun Yang, Agnes B. Fogo, and Yuankai Huo (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상, 특히 병리 이미지에서 세포, 사구체(glomeruli), 핵(nuclei)과 같이 구형(ball-shaped) 형태를 띠는 개체들을 정밀하게 분할(segmentation)하는 문제를 다룬다.

전통적인 딥러닝 기반의 인스턴스 분할 모델들은 주로 직사각형의 Bounding Box를 사용하여 객체의 위치를 탐색한다. 그러나 의료 영상은 촬영 각도에 따라 객체가 다양한 방향으로 회전되어 나타나는데, 직사각형 박스는 회전 각도에 따라 그 형태와 크기가 가변적이어서 일관성이 떨어진다는 문제가 있다. 특히 구형 객체의 경우, 직사각형 박스보다 원형 표현(circle representation)이 객체의 기하학적 특성을 더 잘 반영하며 회전 불변성(rotation invariance)을 확보하는 데 유리하다.

따라서 본 연구의 목표는 구형 의료 객체에 최적화된 원형 표현 기반의 엔드 투 엔드(end-to-end) 인스턴스 분할 프레임워크인 **CircleSnake**를 제안하여, 분할 성능을 높이고 회전 각도에 관계없이 일관된 결과를 얻는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 Bounding Box 기반 제안 방식을 **Bounding Circle** 기반으로 대체하여 구형 객체에 최적화된 초기 컨투어(contour)를 생성하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **원형 표현 파이프라인 구축**: 원형 탐지(circle detection), 원형 컨투어 제안(circle contour proposal), 그리고 원형 컨볼루션(circular convolution)을 하나의 통합된 프레임워크로 연결하였다.
2.  **자유도(Degrees of Freedom, DoF) 감소**: 기존 DeepSnake 등이 사용하는 팔각형(octagon) 표현은 8의 자유도를 가지지만, 제안된 원형 표현은 중심점과 반지름이라는 2의 자유도만으로 객체를 정의한다. 이는 모델의 강건성(robustness)을 높이고 계산 효율성을 향상시킨다.
3.  **회전 일관성(Rotation Consistency) 확보**: 원형은 본질적으로 회전에 불변하는 특성을 가지므로, 이미지의 회전 각도와 상관없이 일관된 초기 컨투어를 제공하여 최종 분할 결과의 재현성을 높였다.

## 📎 Related Works

### 1. Instance Segmentation
마스크 기반(mask-based) 방법론은 크게 Mask R-CNN과 같은 2단계(two-stage) 방식과 YOLACT와 같은 1단계(one-stage) 방식으로 나뉜다. 2단계 방식은 정밀도가 높지만 구조가 복잡하고 추론 시간이 길며, 1단계 방식은 효율적이지만 정밀도가 상대적으로 낮을 수 있다.

### 2. Medical Object Segmentation
의료 영상 분할에서는 픽셀 기반의 이진 마스크(binary mask) 방식이 널리 사용되어 왔다. 하지만 최근에는 DeepSnake와 같은 컨투어 기반(contour-based) 방법론이 등장하여 더 빠르고 단순한 분할 가능성을 보여주었다.

### 3. 기존 접근 방식의 한계 및 차별점
DeepSnake는 Bounding Box로부터 극점(extreme points)을 찾아 팔각형 컨투어를 생성한 뒤 이를 변형(deformation)시키는 방식을 사용한다. 그러나 이러한 방식은 구형 객체에 대해 불필요하게 복잡하며, 회전 시 컨투어 제안의 일관성이 떨어진다는 한계가 있다. CircleSnake는 이를 **Bounding Circle $\rightarrow$ Circle Contour** 과정으로 단순화하여 구형 객체에 특화된 최적화를 달성하였다.

## 🛠️ Methodology

CircleSnake는 CircleNet 기반의 원형 탐지와 DeepSnake 스타일의 컨투어 변형 과정을 결합한 구조이다. 전체 파이프라인은 다음과 같다.

### 1. Circle Object Detection
입력 이미지 $I \in \mathbb{R}^{W \times H \times 3}$에 대해 Center Point Localization 네트워크를 통해 중심점 히트맵 $\hat{Y}$를 생성한다.

-   **중심점 타겟 설정**: 정답 중심점 $(\tilde{p}_x, \tilde{p}_y)$를 중심으로 2D 가우시안 커널을 생성하여 히트맵 $Y$를 정의한다.
    $$Y_{xyc} = \exp \left( -\frac{(x-\tilde{p}_x)^2 + (y-\tilde{p}_y)^2}{2\sigma_p^2} \right)$$
-   **손실 함수**: 중심점 탐지를 위해 Focal Loss가 적용된 픽셀 단위 로지스틱 회귀 손실 $L^k$를 사용하며, 위치 정밀도를 위해 $\ell^1$-norm 오프셋 손실 $L_{off}$를 함께 사용한다.
-   **반지름 예측**: 반지름 예측 헤드를 통해 각 픽셀의 반지름 $\hat{R}$을 예측하며, 손실 함수는 다음과 같은 $\ell^1$ 손실을 따른다.
    $$L_{radius} = \frac{1}{N} \sum_{k=1}^{N} |\hat{R}_{p_k} - r_k|$$
-   **최종 탐지 손실**: $L_{det} = L^k + \lambda_{radius} L_{radius} + \lambda_{off} L_{off}$

### 2. Circle Contour Proposal
탐지된 중심점 $\hat{p}$와 반지름 $\hat{r}$을 이용하여 Bounding Circle을 형성한다. 이 원의 둘레에서 $N=128$개의 점을 균일하게 샘플링하여 초기 컨투어 $x^{circle}_i$를 생성한다. 이는 기존의 복잡한 팔각형 생성 과정을 대체하여 매우 단순하고 일관된 초기화를 가능하게 한다.

### 3. Contour Deformation via Circular Convolution
초기 원형 컨투어를 실제 객체의 경계에 맞게 정밀하게 조정하는 단계이다.

-   **특징 벡터 구성**: 각 정점 $x^{circle}_i$에서 CNN 백본의 특징 맵 $F$와 정점의 좌표를 결합하여 특징 벡터 $[F(x^{circle}_i); x^{circle}_i]$를 생성한다.
-   **원형 컨볼루션(Circular Convolution)**: 컨투어를 1차원 이산 신호로 간주하고, 원형으로 연결된 구조에서 컨볼루션을 수행하여 주변 정점들의 정보를 통합한다.
    $$(f^{circle}_N * k)_i = \sum_{j=-r}^{r} (f^{circle}_N)_{i+j} k_j$$
    여기서 $k$는 학습 가능한 커널이며, 커널 크기는 9로 설정되었다.
-   **반복적 정교화**: 예측된 정점 오프셋을 적용하여 컨투어를 변형시키며, 이 과정을 3회 반복한다. 변형 손실 함수는 정답 경계 점 $x^{gt}_i$와의 $\ell^1$ 거리로 정의된다.
    $$L_{iter} = \frac{1}{N} \sum_{i=1}^{N} \ell^1(\tilde{x}^{circle}_i - x^{gt}_i)$$

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: 사구체(Glomeruli), 세포핵(Nuclei), 호산구(Eosinophils) 데이터셋의 세 가지 벤치마크를 사용하였다.
-   **비교 대상**: Faster-RCNN, Mask-RCNN, CenterNet, DeepSnake, StarDist.
-   **평가 지표**: Average Precision (AP, $AP_{50}, AP_{75}, AP_S, AP_M$), Dice Score, 그리고 회전 일관성 점수(Rotation Consistency Score).

### 2. 정량적 결과
-   **사구체 데이터셋**: CircleSnake-DLA 모델이 모든 지표에서 최상위 성능을 기록하였다. 특히 Segmentation AP(0.623)와 $AP_{50}$(0.894)에서 기존 모델들을 압도하였다.
-   **세포핵 데이터셋**: Detection AP(0.485)에서 우수한 성능을 보였으나, 중간 크기 객체($AP_M$)에서는 상대적으로 낮은 수치를 보였다.
-   **호산구 데이터셋**: 모든 모델에서 성능 하락이 관찰되었는데, 이는 호산구의 특징이 모호하고 배경과의 대비가 낮기 때문으로 분석된다. 그럼에도 CircleSnake는 $AP, AP_{75}, AP_M$에서 경쟁력 있는 성능을 유지하였다.
-   **회전 일관성**: 사구체 데이터셋에서 0.796, 세포핵 데이터셋에서 0.799의 회전 일관성 점수를 기록하며 Mask-RCNN 및 DeepSnake보다 훨씬 높은 회전 불변성을 입증하였다.

### 3. 정성적 분석 및 Ablation Study
-   **시각적 결과**: 원형 제안 방식이 팔각형 제안 방식보다 회전 후에도 더 일관된 초기 컨투어를 제공함을 확인하였다.
-   **StarDist 비교**: 세포핵 데이터셋에서 StarDist(Dice 0.618, Rotation Consistency 0.621)보다 CircleSnake(Dice 0.800, Rotation Consistency 0.799)가 월등히 높은 성능을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
본 연구의 가장 큰 강점은 **객체의 기하학적 특성(원형)을 모델 아키텍처에 직접 반영**했다는 점이다. 분석 결과, 객체의 원형도(ellipticity ratio, 1에 가까울수록 원형)가 높을수록 CircleSnake와 DeepSnake 간의 성능 격차가 커지는 경향이 확인되었다. 이는 원형 표현이 구형 객체 분할에 본질적으로 더 적합함을 시사한다.

### 2. 한계 및 미해결 과제
-   **비원형 객체에 대한 취약성**: 타원형이나 길쭉한 형태의 객체에 대해서는 원형 컨투어의 초기화가 부적절하여, 탐지(Detection) 성능에 비해 분할(Segmentation) 성능이 낮게 나타나는 경향이 있다.
-   **단일 스케일 특징 맵**: 현재 모델은 단일 스케일 특징 맵을 사용하므로, 크기 변화가 극심한 객체를 처리하는 데 한계가 있다. 저자들은 향후 multi-scale feature map 도입이 필요하다고 언급하였다.

### 3. 비판적 해석
제안된 방법은 매우 효율적이지만, '구형'이라는 강한 가정을 전제로 한다. 따라서 일반적인 객체 분할보다는 특정 의료 도메인(병리 영상의 세포 등)에 특화된 도구로서의 가치가 크다. 또한, 호산구 데이터셋에서 나타난 성능 저하는 모델의 구조적 문제보다는 데이터 자체의 모호함(low contrast)에 기인한 것이므로, 이를 해결하기 위해서는 단순한 표현 방식의 변경보다는 더 강력한 특징 추출기나 전처리 기법이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 구형 의료 객체 분할을 위해 Bounding Box 대신 **Bounding Circle**을 사용하는 **CircleSnake**를 제안한다. 이 방법은 자유도를 8에서 2로 획기적으로 줄이면서도, 원형 컨볼루션을 통한 반복적 변형으로 정밀한 분할을 수행한다. 특히 **회전 불변성(Rotation Invariance)**이 뛰어나 촬영 각도에 민감한 의료 영상 분석에서 매우 효율적이며, 사구체 및 세포핵 분할에서 기존 SOTA 모델들을 상회하는 성능을 입증하였다. 향후 다양한 의료 영상 모달리티(CT, MRI 등)의 구형 병변 탐지에 광범위하게 적용될 가능성이 높다.