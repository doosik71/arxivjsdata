# GLOBALMAMBA: GLOBAL IMAGE SERIALIZATION FOR VISION MAMBA

Chengkun Wang, Wenzhao Zheng, Jie Zhou, Jiwen Lu (2024)

## 🧩 Problem to Solve

최근 Mamba와 같은 State Space Models (SSMs)는 Transformer의 $O(n^2)$ 복잡도를 $O(n)$으로 줄이면서도 강력한 성능을 보여주며 컴퓨터 비전 분야에 도입되고 있다. 그러나 기존의 Vision Mamba 모델들은 이미지를 패치 단위로 토큰화한 후 이를 단순히 1차원 시퀀스로 평탄화(flattening)하여 처리하는 방식을 사용한다.

이러한 접근 방식은 두 가지 핵심적인 문제를 야기한다. 첫째, 이미지의 내재적인 2D 구조적 상관관계와 국소 불변성(local invariance) 특성을 무시하게 되어, 공간적으로 인접한 패치들이 시퀀스 상에서는 멀리 떨어지게 되는 등 인과적 순서(causal order)가 파괴된다. 둘째, 기존의 토큰화 방식은 각 토큰이 국소적인 정보만을 담고 있어, 이미지 전체의 전역적 맥락(global context)을 포착하는 능력이 부족하다. 따라서 본 논문의 목표는 이미지의 전역적 정보를 유지하면서 Mamba의 인과적 아키텍처에 적합한 효율적인 이미지 시리얼라이제이션(serialization) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지를 공간 도메인이 아닌 주파수 도메인(frequency domain)에서 분석하여 시퀀스를 생성하는 Global Image Serialization (GIS) 방법론이다.

핵심 직관은 인간이 이미지를 인식할 때 윤곽선과 같은 저주파 성분을 먼저 파악하고 이후 세부적인 고주파 정보를 통해 보완한다는 점과, 신경망이 저주파 신호를 우선적으로 학습하는 경향(frequency principle)이 있다는 점에 기반한다. 이를 위해 Discrete Cosine Transform (DCT)를 이용하여 이미지를 저주파부터 고주파까지의 순서로 배열함으로써, Mamba 모델이 전역적 특징에서 세부 특징으로 이어지는 자연스러운 인과적 관계를 학습할 수 있도록 설계하였다.

## 📎 Related Works

기존의 Vision Mamba 연구들인 Vim, VMamba, LocalMamba 등은 이미지를 행/열 방향으로 스캔하거나 국소 윈도우 내에서 평탄화하는 전략을 사용하였다. 하지만 이러한 방법들은 이미지의 2D 구조를 강제로 1D로 변환하는 과정에서 인접 픽셀 간의 관계를 훼손하며, 각 토큰이 공간적으로 제한된 정보만을 가지게 된다는 한계가 있다.

또한, Causal Sequence Modeling 관점에서 Mamba는 RNN과 유사하게 은닉 상태(hidden state)를 통해 순차적으로 정보를 처리하므로 입력 데이터의 순서가 매우 중요하다. 하지만 기존 비전 모델들의 평탄화 방식은 논리적인 인과 관계가 결여되어 있어 Mamba의 잠재력을 완전히 활용하지 못하고 있다. 본 논문은 이러한 한계를 극복하기 위해 주파수 분할을 통한 새로운 인과적 순서를 부여함으로써 기존 방식과 차별화한다.

## 🛠️ Methodology

### 1. Frequency-Based Global Image Serialization

제안된 방법론의 핵심인 GIS 프로세스는 다음과 같은 단계로 진행된다.

**가. 주파수 도메인 변환 (DCT)**
입력 이미지 $x \in \mathbb{R}^{h \times w}$를 이차원 Discrete Cosine Transform (DCT)를 통해 주파수 도메인 $F(u,v)$로 변환한다.
$$F(u,v) = \alpha(u)\alpha(v) \sum_{i=0}^{h-1} \sum_{j=0}^{w-1} x(i,j) \cos\left(\frac{(2i+1)u\pi}{2h}\right) \cos\left(\frac{(2j+1)u\pi}{2w}\right)$$
여기서 $\alpha(u), \alpha(v)$는 스케일링 인자이다.

**나. 주파수 세그먼트 분할 (Frequency Segmentation)**
DCT 변환 후의 스펙트럼에서 저주파 성분은 좌상단에, 고주파 성분은 우하단에 집중된다. 본 논문에서는 주 대각선에 수직인 방향으로 스펙트럼을 $K$개의 구간으로 분할한다. 각 분할 지점 $f_k$를 기준으로 저주파부터 고주파까지의 범위를 설정하며, $k$번째 대역에 속하지 않는 주파수 성분의 진폭은 0으로 처리하여 $K$개의 독립적인 스펙트럼 다이어그램 $F^k(u,v)$를 생성한다.
$$F^k(u,v) = I(f_k - f(u,v)) F(u,v), \quad I(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

**다. 공간 도메인 복원 및 다운샘플링 (IDCT & Downsampling)**
생성된 $K$개의 스펙트럼을 Inverse DCT (IDCT)를 통해 다시 공간 도메인의 이미지 $x^k(i,j)$로 복원한다. 이때, 저주파 대역의 이미지는 전역적인 정보를 담고 있으므로 더 낮은 해상도로 다운샘플링하여 토큰 수를 조절한다.
$$x'_k = G(x^k, \frac{h}{2^{K-k}}, \frac{w}{2^{K-k}})$$
여기서 $G(\cdot)$는 다운샘플링 보간 함수이며, $k$가 작을수록(저주파일수록) 해상도가 낮아진다.

**라. 토큰화 및 시퀀스 구성**
복원된 이미지들을 가벼운 CNN 기반의 Tokenizer에 통과시켜 토큰을 추출한다. 최종적으로 저주파에서 고주파 순서로 토큰들을 배열하여 Mamba 모델의 입력으로 사용하는 인과적 시퀀스를 구성한다.

### 2. GlobalMamba 아키텍처

GlobalMamba는 위에서 생성된 시리얼라이즈된 토큰 시퀀스를 입력으로 받는 Vision Mamba 백본이다. 구조적으로는 Vim과 같은 Plain 타입과 VMamba와 같은 Pyramid 타입 모두에 적용 가능하다. 각 블록은 다음과 같은 연산을 수행한다.
$$t_n = z_{n-1} + \text{SSM}(\text{Norm}(z_{n-1}))$$
$$z_n = t_n + \text{MLP}(\text{Norm}(t_n))$$
최종적으로 추출된 특징은 분류기(Classifier)나 다운스트림 태스크의 헤드로 전달된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: ImageNet-1K (분류), MS COCO 2017 (객체 탐지), ADE20K (시맨틱 세그멘테이션)
- **비교 대상**: Vim, VMamba 등 기존 Vision Mamba 모델 및 ResNet, Swin Transformer 등
- **지표**: Top-1 Accuracy, box AP, mask AP, mIoU

### 2. 주요 결과

- **이미지 분류 (ImageNet-1K)**: GlobalMamba는 다양한 모델 크기(Mini, Tiny, Small, Base)에서 베이스라인보다 일관된 성능 향상을 보였다. 특히 Vim 대비 약 $+0.6\%$의 성능 향상을 달성하였다.
- **객체 탐지 및 인스턴스 분할 (COCO)**: Mask-RCNN을 사용하여 평가한 결과, GlobalMamba-S가 VMamba-S보다 $1\times$ 및 $3\times$ 학습 스케줄 모두에서 box AP 및 mask AP 수치가 더 높게 나타났다.
- **시맨틱 세그멘테이션 (ADE20K)**: UPerNet을 사용하여 평가하였으며, GlobalMamba-S가 VMamba-S 대비 mIoU(SS) 기준 $0.3$ 포인트 높은 성능을 기록하였다.

### 3. 분석 실험 (Ablation Study)

- **인과적 순서**: 저주파 $\rightarrow$ 고주파 순서의 배치가 무작위 배치나 고주파 $\rightarrow$ 저주파 배치보다 훨씬 높은 분류 정확도를 보였다. 이는 저주파 우선 학습이라는 신경망의 특성과 일치한다.
- **세그먼트 수 ($K$)**: $K$값이 2에서 6으로 증가함에 따라 성능이 상승하다가 $K=4$ 지점에서 안정화되는 경향을 보였다. 따라서 최적의 값인 $K=4$를 기본 설정으로 사용하였다.
- **Causal Transformer 적용**: 제안한 GIS 방법론을 Mamba가 아닌 Decoder-only Transformer에 적용했을 때도 성능이 향상됨을 확인하여, GIS의 범용적인 우수성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 단순히 모델의 파라미터를 늘리는 것이 아니라, 데이터가 모델에 입력되는 '순서'와 '형태'를 최적화함으로써 성능을 높였다는 점에서 큰 의미가 있다. 특히 저주파 성분을 먼저 배치하고 전역적 정보를 담은 토큰을 생성하는 방식은 Mamba의 인과적 특성을 비전 데이터에 맞게 재해석한 매우 영리한 접근이다.

또한, 저주파 대역에 더 높은 다운샘플링 비율을 적용함으로써, 토큰 시퀀스가 늘어남에도 불구하고 계산 복잡도(FLOPs)의 증가를 최소화한 점이 돋보인다.

다만, 한계점으로 고주파 성분 영역에서는 여전히 일부 평탄화(flattening) 작업이 수반된다는 점을 언급하고 있다. 이는 완전히 2D 구조를 보존하는 시리얼라이제이션을 달성하지 못한 부분으로, 향후 연구에서 해결해야 할 과제로 남아있다.

## 📌 TL;DR

GlobalMamba는 이미지의 2D 구조 파괴와 전역 정보 부족 문제를 해결하기 위해 **DCT 기반의 주파수 분할 시리얼라이제이션(GIS)**을 제안한다. 이미지를 저주파에서 고주파 순서의 토큰 시퀀스로 변환하여 Mamba의 인과적 모델링 능력을 극대화하였으며, 이를 통해 ImageNet 분류, COCO 탐지, ADE20K 세그멘테이션 등 주요 비전 태스크에서 기존 Vision Mamba 모델들보다 우수한 성능을 달성하였다. 이 연구는 비전 데이터의 시퀀스화 전략이 모델의 성능에 결정적인 영향을 미칠 수 있음을 시사한다.
