# End-to-End Video Instance Segmentation with Transformers

Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, Huaxia Xia (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Video Instance Segmentation (VIS)이다. VIS는 비디오 내에서 관심 대상인 객체 인스턴스들에 대해 분류(Classification), 세그멘테이션(Segmentation), 그리고 추적(Tracking)을 동시에 수행해야 하는 고난도 작업이다.

기존의 VIS 접근 방식들은 매우 복잡한 파이프라인을 가지고 있다. Top-down 방식은 주로 Mask R-CNN과 같은 이미지 수준의 인스턴스 세그멘테이션 모델에 의존하며, 프레임 간 인스턴스를 연결하기 위해 사람이 설계한 복잡한 규칙(heuristic rules)을 사용하는 Tracking-by-detection 패러다임을 따른다. 반면 Bottom-up 방식은 픽셀 임베딩을 클러스터링하여 인스턴스를 분리하는데, 이는 마스크를 생성하기 위해 여러 단계의 반복적인 과정이 필요하여 연산 속도가 느리다는 단점이 있다.

따라서 본 논문의 목표는 이러한 복잡한 파이프라인을 제거하고, 단순하면서도 효율적으로 학습 가능한 end-to-end VIS 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 VIS 작업을 **병렬 시퀀스 디코딩/예측 문제(parallel sequence decoding/prediction problem)**로 정의하고 이를 Transformer 구조로 해결하는 것이다.

중심적인 직관은 인스턴스 세그멘테이션과 인스턴스 추적을 모두 **Similarity Learning**의 관점에서 바라보는 것이다. 즉, 인스턴스 세그멘테이션은 픽셀 수준의 유사성을 학습하는 것이고, 인스턴스 추적은 인스턴스 간의 유사성을 학습하는 것이라고 정의한다. 이를 통해 두 작업을 하나의 프레임워크 내에서 통합하여 상호 보완적인 효과를 얻을 수 있도록 설계하였다.

이를 위해 저자들은 Transformer를 도입하여 비디오 클립 전체를 입력으로 받고, 각 인스턴스에 대한 마스크 시퀀스를 직접 출력하는 VisTR(Video Instance Segmentation TRansformer)을 제안한다.

## 📎 Related Works

### 관련 연구 및 한계

1. **Video Object Segmentation (VOS):** VOS는 클래스 구분 없이 전경 객체를 세그멘테이션하는 작업으로, 주로 첫 몇 프레임의 주석(annotation)을 사용하여 추적한다. 반면 VIS는 정해진 카테고리 내의 모든 인스턴스를 분류하고 추적해야 한다는 점에서 더 어렵다.
2. **기존 VIS 방법론:** MaskTrack R-CNN이나 MaskProp 등은 이미지 기반 모델과 외부 메모리, 혹은 복잡한 전파(propagation) 메커니즘을 사용하여 인스턴스를 연결한다. STEm-Seg와 같은 방식은 3D 시공간 볼륨을 모델링하지만, 마스크 생성 과정이 반복적이어서 속도가 느리다.
3. **Transformers:** DETR과 같은 연구를 통해 Transformer가 객체 탐지 파이프라인을 획기적으로 단순화할 수 있음이 증명되었다. VisTR은 이러한 DETR의 철학을 비디오 영역으로 확장하여, Transformer의 self-attention 메커니즘이 비디오의 장거리 의존성(long-range dependencies)과 시간적 정보를 학습하는 데 적합하다는 점을 이용한다.

## 🛠️ Methodology

### 전체 시스템 구조

VisTR의 전체 파이프라인은 크게 네 가지 구성 요소로 이루어진다.

1. **CNN Backbone:** 입력 비디오 클립의 각 프레임에서 특징 맵을 추출한다.
2. **Transformer Encoder-Decoder:** 픽셀 수준 및 인스턴스 수준의 특징 간 유사성을 모델링한다.
3. **Instance Sequence Matching Module:** 예측된 시퀀스와 정답(Ground-truth) 시퀀스를 매칭하여 모델을 감독한다.
4. **Instance Sequence Segmentation Module:** 최종적인 마스크 시퀀스를 생성한다.

### 상세 구성 요소 및 절차

#### 1. 특징 추출 및 인코딩

입력 비디오 클립 $x_{clip} \in \mathbb{R}^{T \times 3 \times H_0 \times W_0}$ ($T$는 프레임 수)가 CNN backbone을 통과하여 클립 레벨 특징 맵 $f^0 \in \mathbb{R}^{T \times C \times H \times W}$가 생성된다. 이후 $1 \times 1$ convolution을 통해 차원을 $d$로 줄인 뒤, 공간 및 시간 차원을 평탄화(flatten)하여 $\mathbb{R}^{d \times (T \cdot H \cdot W)}$ 형태의 시퀀스로 변환하여 Transformer encoder에 입력한다.

Transformer는 위치 불변성(permutation-invariant)을 가지므로, 시간($t$), 가로($x$), 세로($y$)의 3차원 위치 정보를 포함하는 **3D Positional Encoding**을 추가한다. 각 차원에 대해 $\frac{d}{3}$ 크기의 사인 및 코사인 함수를 사용하여 다음과 같이 정의한다.
$$PE(pos,i) = \begin{cases} \sin(pos \cdot \omega_k), & \text{for } i=2k \\ \cos(pos \cdot \omega_k), & \text{for } i=2k+1 \end{cases}$$
여기서 $\omega_k = 1/10000^{2k/(d/3)}$이다.

#### 2. Transformer Decoder 및 Instance Queries

Decoder는 각 프레임의 인스턴스를 대표하는 **Instance level features**를 디코딩한다. 이를 위해 학습 가능한 $N=n \cdot T$개의 **Instance queries**($n$은 프레임당 예측할 인스턴스 수)를 입력으로 사용한다. 출력되는 $N$개의 예측 결과는 입력 프레임 순서를 따르며, 서로 다른 이미지 간의 동일한 인덱스는 동일한 인스턴스로 간주되어 별도의 추적 과정 없이 자연스럽게 Tracking이 이루어진다.

#### 3. Instance Sequence Matching

예측된 시퀀스와 정답 시퀀스 간의 대응 관계를 찾기 위해 **Bipartite graph matching**을 수행한다. 마스크 시퀀스를 직접 비교하는 것은 연산 비용이 크므로, 대리 지표인 **Box sequence**를 사용하여 헝가리안 알고리즘(Hungarian algorithm)으로 최적의 매칭 $\hat{\sigma}$를 찾는다.
$$\hat{\sigma} = \arg \min_{\sigma \in S_n} \sum_{i=1}^n L_{match}(y_i, \hat{y}_{\sigma(i)})$$
여기서 $L_{match}$는 클래스 예측 확률과 박스 유사도를 기반으로 계산된다.

최종 손실 함수인 **Hungarian loss**는 다음과 같이 정의된다.
$$L_{Hung}(y, \hat{y}) = \sum_{i=1}^N [ (-\log \hat{p}_{\hat{\sigma}(i)}(c_i)) + L_{box}(b_i, \hat{b}_{\hat{\sigma}(i)}) + L_{mask}(m_i, \hat{m}_{\hat{\sigma}(i)}) ]$$
$L_{box}$는 $L_1$ loss와 Generalized IoU loss의 선형 결합으로 구성되며, $L_{mask}$는 Dice loss와 Focal loss의 결합으로 정의된다.

#### 4. Instance Sequence Segmentation

최종 마스크를 생성하기 위해 먼저 객체 예측값 $O$와 인코더 특징 $E$ 사이의 유사도 맵을 계산하고, 이를 백본 특징 $B$ 및 변환된 $E$와 융합한다. 마지막 층에는 Deformable convolution layer가 사용된다.

특히, 동일 인스턴스의 여러 프레임 특징들이 서로 보완하도록 **3D Convolution**을 적용한다. 각 인스턴스 $i$에 대해 $T$개 프레임의 특징을 결합하여 $G_i \in \mathbb{R}^{1 \times a \times T \times H_0/4 \times W_0/4}$를 형성하고, 이를 3개의 3D conv layer와 Group Normalization을 거쳐 최종 마스크 시퀀스 $m_i$를 출력한다.

## 📊 Results

### 실험 설정

- **데이터셋:** YouTube-VIS (학습 2238개, 검증 302개, 테스트 343개 클립).
- **평가 지표:** Average Precision (AP) 및 Average Recall (AR).
- **구현 세부사항:** ResNet-50 및 ResNet-101 백본 사용. 프레임당 10개의 객체를 예측하도록 설정하여 총 360개의 쿼리 사용($T=36$ 기준).

### 정량적 결과

VisTR은 단일 모델을 사용하는 방법론 중 가장 우수한 성능과 속도를 기록하였다.

- **ResNet-101 기반 VisTR:** AP $40.1\%$, 속도는 데이터 로딩을 제외할 경우 최대 **57.7 FPS**를 달성하였다.
- **비교 분석:** MaskTrack R-CNN 및 STEm-Seg보다 약 6%p 높은 AP를 보였다. MaskProp보다 AP는 약간 낮으나, MaskProp은 여러 네트워크(FPN, HTC 등)와 복잡한 후처리(Mask Refinement)를 결합한 형태인 반면 VisTR은 단순한 end-to-end 구조라는 점에서 차별점이 있다.

### Ablation Study 결과

1. **시퀀스 길이:** 입력 프레임 수가 18에서 36으로 증가함에 따라 AP가 $29.7\%$에서 $33.3\%$로 단조 증가하여, 더 많은 시간적 정보가 성능 향상에 기여함을 확인하였다.
2. **순서 및 위치 인코딩:** 랜덤 순서보다 시간 순서대로 입력했을 때 성능이 더 좋았으며, 명시적인 Position Encoding을 추가했을 때 AP가 약 5%p 향상되었다.
3. **Instance Queries:** 쿼리를 모든 프레임에 공유하는 'instance level' 설정이 'prediction level' 설정(각 예측마다 고유 쿼리)과 매우 유사한 성능(각각 $32.0\%$ vs $33.3\%$)을 보여, 쿼리 공유를 통한 자연스러운 추적이 가능함을 입증하였다.
4. **3D Convolution:** 3D 인스턴스 시퀀스 세그멘테이션 모듈을 사용했을 때 AP가 $1.1\%$p 향상되어, 시간적 정보의 전파가 마스크 품질을 높임을 확인하였다.

## 🧠 Insights & Discussion

### 강점

VisTR의 가장 큰 강점은 **단순성(Simplicity)**과 **효율성(Efficiency)**이다. 기존의 복잡한 데이터 연관(data association) 규칙이나 반복적인 마스크 최적화 과정 없이, Transformer의 병렬 디코딩을 통해 추적과 세그멘테이션을 한 번에 해결하였다. 특히 후처리가 필요 없는 구조 덕분에 매우 높은 FPS를 달성할 수 있었다.

### 한계 및 비판적 해석

논문에서는 단일 모델로서의 최고 성능을 강조하지만, MaskProp과 같은 다단계 정밀화(multi-stage refinement) 모델보다는 정밀도가 낮다. 이는 "bells and whistles"를 제거하고 단순한 구조를 추구했기 때문이라고 설명하지만, 실제 정밀한 세그멘테이션이 필요한 도메인에서는 이러한 단순함이 한계로 작용할 수 있다. 또한, 입력 시퀀스 길이 $T$가 고정되어 있어, 매우 긴 비디오의 경우 클립을 나누어 처리해야 하며 이때 발생하는 클립 간 연결 문제는 논문에서 명확하게 다루지 않았다.

## 📌 TL;DR

VisTR은 VIS 작업을 **병렬 시퀀스 예측 문제**로 재정의하고 Transformer를 도입하여, 복잡한 추적 알고리즘 없이 end-to-end로 인스턴스 세그멘테이션과 추적을 동시에 수행하는 프레임워크이다. 3D Positional Encoding과 3D Convolution을 통해 시간적 일관성을 학습하였으며, YouTube-VIS 데이터셋에서 단일 모델 기준 최상의 속도(57.7 FPS)와 경쟁력 있는 정확도(40.1% AP)를 달성하였다. 이 연구는 향후 다양한 비디오 이해 작업들을 Transformer 기반의 단순한 시퀀스 예측 구조로 통합할 수 있는 가능성을 제시하였다.
