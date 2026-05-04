# UNETR: Transformers for 3D Medical Image Segmentation

Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger R. Roth, Daguang Xu (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 3D 의료 영상 분할(Medical Image Segmentation)에서 기존의 Fully Convolutional Neural Networks(FCNNs), 특히 U-Net과 같은 구조가 가지는 국소적 수용 영역(localized receptive fields)의 한계이다. FCNNs는 인코더를 통해 지역적 특징과 문맥적 표현을 학습하지만, 합성곱 계층(convolutional layers)의 본질적인 국소성으로 인해 장거리 공간 의존성(long-range spatial dependencies)을 학습하는 능력이 제한된다.

이러한 한계는 크기와 모양이 매우 다양하거나 복잡한 해부학적 구조(예: 다양한 크기의 뇌 병변)를 분할할 때 성능 저하를 야기한다. 따라서 본 연구의 목표는 Transformer의 강력한 전역 문맥 모델링 능력을 3D 의료 영상 분할에 도입하여, 전역적 다중 스케일 정보(global multi-scale information)를 효과적으로 캡처하고 분할 정확도를 높이는 새로운 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 가장 중심적인 기여는 3D 의료 영상 분할 작업을 1D sequence-to-sequence 예측 문제로 재정의하고, 이를 처리하기 위한 UNETR(UNet TRansformers) 아키텍처를 설계한 것이다. 

UNETR의 핵심 직관은 인코더에는 전역적 의존성을 학습하는 Transformer를 배치하고, 디코더에는 국소적 정보를 복원하는 CNN 기반 구조를 사용하는 하이브리드 설계를 채택하는 것이다. Transformer는 이미지 전체의 문맥을 파악하는 데 탁월하지만 국소적인 세부 정보를 캡처하는 데는 취약하므로, CNN 디코더와 Skip Connection을 결합함으로써 전역적 문맥과 지역적 세부 정보를 모두 활용하여 정밀한 분할 결과물을 생성하도록 설계되었다.

## 📎 Related Works

기존의 3D 의료 영상 분할은 주로 U-Net 기반의 CNN 네트워크들이 주도해 왔으며, 최근에는 수용 영역을 넓히기 위해 Atrous Convolution이나 Self-attention 모듈을 추가하는 방식이 제안되었다. 그러나 이러한 시도들 역시 합성곱의 국소성이라는 근본적인 제약을 완전히 극복하지 못했다.

최근 컴퓨터 비전 분야에서는 Vision Transformer(ViT)와 같은 모델들이 등장하여 전역적 문맥 모델링의 가능성을 보여주었으며, SETR나 TransUNet과 같은 2D 이미지 분할 모델들도 제안되었다. UNETR는 이러한 기존 접근 방식과 다음과 같은 차별점을 가진다.

첫째, UNETR는 2D가 아닌 3D 볼륨 데이터를 직접 처리하도록 설계되었다. 둘째, Transformer를 단순한 attention layer나 병목(bottleneck) 지점에 추가하는 것이 아니라, 네트워크의 메인 인코더로 사용하여 전체 구조의 기반으로 삼았다. 셋째, 입력 시퀀스를 생성하기 위해 별도의 CNN backbone을 사용하지 않고, 3D 패치를 직접 토큰화(tokenized patches)하여 Transformer에 입력한다.

## 🛠️ Methodology

### 전체 시스템 구조
UNETR는 전형적인 U-shape 구조를 따르며, Transformer 기반의 인코더와 CNN 기반의 디코더로 구성된다. 3D 입력 볼륨을 1D 시퀀스로 변환하여 Transformer가 처리하게 하고, 그 결과물을 다양한 해상도에서 CNN 디코더로 전달하여 최종 분할 맵을 생성한다.

### 주요 구성 요소 및 절차
1. **입력 및 패치 임베딩**: 입력 3D 볼륨 $x \in \mathbb{R}^{H \times W \times D \times C}$를 겹치지 않는 균일한 3D 패치들로 나눈다. 각 패치의 해상도를 $(P, P, P)$라고 할 때, 총 $N = (H \times W \times D) / P^3$개의 패치가 생성된다. 이 패치들은 선형 층(linear layer)을 통해 $K$차원의 임베딩 공간으로 투영된다.
2. **Positional Embedding**: Transformer는 입력의 순서 정보를 알지 못하므로, 공간 정보를 보존하기 위해 학습 가능한 1D positional embedding $E_{pos} \in \mathbb{R}^{N \times K}$를 더해준다. 최종 입력 시퀀스 $z_0$는 다음과 같이 정의된다.
   $$z_0 = [x^1_{vE}, x^2_{vE}, \dots, x^N_{vE}] + E_{pos}$$
3. **Transformer Encoder**: $L$개의 Transformer 블록으로 구성되며, 각 블록은 Multi-head Self-Attention(MSA)과 Multi-layer Perceptron(MLP) 서브레이어로 이루어져 있다. 
   - MSA는 쿼리($q$), 키($k$), 값($v$) 사이의 유사도를 측정하여 가중 합을 계산한다. 어텐션 가중치 $A$는 다음과 같다.
     $$A = \text{Softmax}\left(\frac{qk^T}{\sqrt{K_h}}\right)$$
     여기서 $K_h = K/n$은 스케일링 인자이다.
4. **CNN Decoder 및 Skip Connection**: Transformer 인코더의 서로 다른 층(예: $i \in \{3, 6, 9, 12\}$)에서 출력된 시퀀스 표현을 추출하여 다시 3D 텐서 형태로 재구성한다. 이 텐서들은 $3 \times 3 \times 3$ 합성곱 층과 정규화 층을 거쳐 디코더로 전달된다.
5. **업샘플링 및 최종 출력**: 인코더의 최하단(bottleneck)에서 deconvolutional layer를 통해 해상도를 2배로 높인 후, 이전 단계의 특징 맵과 결합(concatenate)한다. 이 과정이 입력 해상도에 도달할 때까지 반복되며, 마지막으로 $1 \times 1 \times 1$ 합성곱 층과 softmax 함수를 통해 각 복셀(voxel)별 클래스를 예측한다.

### 손실 함수
본 모델은 Soft Dice Loss와 Cross-Entropy Loss를 결합하여 사용한다. 손실 함수 $L(G, Y)$는 다음과 같다.
$$L(G,Y) = 1 - \frac{2 \sum_{j=1}^J \sum_{i=1}^I G_{i,j} Y_{i,j}}{\sum_{i=1}^I G_{i,j}^2 + \sum_{i=1}^I Y_{i,j}^2} - \frac{1}{I} \sum_{i=1}^I \sum_{j=1}^J G_{i,j} \log Y_{i,j}$$
여기서 $I$는 복셀 수, $J$는 클래스 수, $Y_{i,j}$는 예측 확률, $G_{i,j}$는 원-핫 인코딩된 정답(ground truth)을 의미한다.

## 📊 Results

### 실험 설정
- **데이터셋**: BTCV(복부 CT, 13개 장기 분할) 및 MSD(뇌종양 MRI, 비장 CT 분할)를 사용하였다.
- **평가 지표**: Dice score와 95% Hausdorff Distance (HD95)를 사용하여 정량적으로 평가하였다.
- **구현 세부사항**: PyTorch와 MONAI 프레임워크를 사용하였으며, ViT-B16 아키텍처($L=12, K=768, P=16$)를 기반으로 구축되었다.

### 주요 결과
- **BTCV 데이터셋**: UNETR는 BTCV 리더보드의 Standard 및 Free Competition 섹션 모두에서 새로운 SOTA(State-of-the-art) 성능을 달성하였다. 특히 담낭(gallbladder)과 부신(adrenal glands) 같은 작은 장기 분할에서 기존 모델보다 월등히 높은 Dice score를 기록하였다.
- **MSD 데이터셋**: 뇌종양 분할 및 비장 분할 작업 모두에서 기존 CNN 및 Transformer 기반 방법론보다 우수한 성능을 보였다. 특히 뇌종양의 Tumor Core(TC) 영역 분할에서 강점을 나타냈다.
- **정성적 결과**: 시각적 분석 결과, UNETR는 간(liver)과 위(stomach)의 경계를 명확히 구분하는 등 장거리 의존성 학습 능력을 입증하였으며, 대조도가 낮은 조직 간의 경계 또한 효과적으로 캡처하는 모습을 보였다.

## 🧠 Insights & Discussion

UNETR의 강점은 Transformer의 전역적 문맥 파악 능력과 CNN의 국소적 정밀도를 적절히 결합했다는 점에 있다. 특히 기존 CNN 모델들이 어려워하던 작은 크기의 장기나 복잡한 형태의 구조물을 정확하게 분할해낸 점은, 의료 영상 분석에서 전역적인 해부학적 관계를 파악하는 것이 얼마나 중요한지를 시사한다.

다만, 본 논문에서 제시된 한계점 및 논의 사항은 다음과 같다.
- **패치 해상도와 메모리의 트레이드오프**: 패치 해상도를 낮추면(예: $32 \to 16$) 성능은 향상되지만, 시퀀스 길이가 기하급수적으로 증가하여 메모리 소모가 극심해진다. 이로 인해 더 낮은 해상도에 대한 실험이 불가능했다는 점이 언급되었다.
- **가정 및 제약**: 사전 학습된 가중치(pre-trained weights)를 사용하지 않았음에도 성능이 좋았으나, 이는 데이터셋의 규모나 특성에 따라 다를 수 있다.

비판적으로 해석하자면, UNETR는 Transformer를 인코더로 사용하여 전역 정보를 얻었지만, 디코더는 여전히 전통적인 CNN 구조에 의존하고 있다. 향후 연구에서는 디코더 단계에서도 Transformer의 메커니즘을 어떻게 효율적으로 통합할 수 있을지가 중요한 과제가 될 것으로 보인다.

## 📌 TL;DR

본 논문은 3D 의료 영상 분할을 위해 Transformer 인코더와 CNN 디코더를 결합한 **UNETR** 아키텍처를 제안하였다. 3D 볼륨을 1D 시퀀스로 변환하여 처리함으로써 기존 CNN의 한계였던 장거리 공간 의존성 학습 문제를 해결하였으며, 이를 통해 BTCV 및 MSD 데이터셋에서 SOTA 성능을 달성하였다. 특히 작은 장기 및 복잡한 조직의 경계 분할에서 탁월한 성능을 보여, 향후 의료 영상 분석 분야에서 Transformer 기반 분할 모델의 기초가 될 가능성이 매우 높다.