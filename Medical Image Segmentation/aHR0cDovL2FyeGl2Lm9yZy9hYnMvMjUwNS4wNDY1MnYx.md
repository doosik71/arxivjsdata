# Rethinking Boundary Detection in Deep Learning-Based Medical Image Segmentation

Yi Lin, Dong Zhang, Xiao Fang, Yufan Chen, Kwang-Ting Cheng, Hao Chen (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation, MedISeg)은 의료 영상 분석 및 컴퓨터 비전 분야에서 핵심적인 과제이다. 최근의 딥러닝 기반 방법론들은 관심 영역(Region of Interest)의 주요 부분을 정확하게 분할하는 성과를 거두었으나, 객체의 경계 영역(Boundary areas)을 정밀하게 분할하는 것은 여전히 어려운 과제로 남아 있다. 

기존의 Convolutional Neural Networks(CNNs)는 국소적 특징 추출에 강점이 있지만 전역적인 문맥 파악 능력이 부족하며, Vision Transformer(ViT)는 장거리 의존성(Long-range dependencies)을 포착하는 데 탁월하지만 이동 불변성(Translation invariance)과 국소적 특징 표현 능력이 떨어진다는 한계가 있다. 본 논문의 목표는 CNN, ViT, 그리고 명시적인 경계 검출 연산자를 결합하여, 추가적인 데이터 입력이나 레이블 주입 없이도 경계 영역의 분할 정확도를 높이고 연산 효율성 사이의 균형을 맞춘 새로운 네트워크 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **CTO(Convolution, Transformer, and Operator)**라는 통합 프레임워크를 통해 국소적 특징, 전역적 의존성, 그리고 명시적 객체 경계 정보를 동시에 활용하는 것이다. 

주요 기여 사항은 다음과 같다.
1. **Dual-Stream Encoder 설계**: 국소적 특징을 캡처하는 CNN 스트림과 전역적 의존성을 통합하는 보조적인 StitchViT 스트림을 병렬로 구성하여 정보의 상호 보완성을 극대화하였다.
2. **StitchViT 제안**: 연산 비용을 낮게 유지하면서도 다양한 수용장(Receptive field)을 가지는 패치를 샘플링하는 'Stitch' 연산을 도입하여 전역 및 국소 특징을 효율적으로 캡처한다.
3. **Boundary-Guided Decoder 설계**: Sobel 연산자를 통해 스스로 생성한 이진 경계 마스크(Self-generated boundary mask)를 디코딩 과정에 명시적인 지도 신호로 활용하는 Boundary-Extracted Module(BEM)과 Boundary-Injected Module(BIM)을 제안하였다.

## 📎 Related Works

기존의 의료 영상 분할 연구는 크게 세 가지 범주로 나뉜다.
- **CNN 기반 방법**: U-Net, V-Net 등이 대표적이며 스킵 연결(Skip connections)과 다중 스케일 표현을 사용한다. 하지만 컨볼루션의 국소적 특성으로 인해 분할 마스크가 불완전하게 생성되는 경향이 있다.
- **ViT 기반 방법**: Swin-UNet, MissFormer 등이 있으며 장거리 의존성을 통해 전역 정보를 통합한다. 그러나 의료 영상 데이터의 부족으로 최적화가 어렵고 연산 비용이 매우 높다.
- **CNN-ViT 하이브리드 방법**: TransUNet, UNETR, Swin-UNETR 등이 CNN의 국소 특징과 ViT의 전역 특징을 결합한다. 하지만 이들 역시 높은 계산 오버헤드 문제를 겪고 있다.

본 논문은 이러한 기존 방식들이 경계 정보를 암시적으로만 학습한다는 점에 주목하며, 고전적인 영상 처리 연산자인 Sobel operator를 사용하여 경계 정보를 명시적으로 추출하고 이를 학습 과정에 주입함으로써 차별성을 둔다.

## 🛠️ Methodology

### 전체 파이프라인
CTO는 전형적인 인코더-디코더(Encoder-Decoder) 구조를 따른다. 인코더에서 추출된 CNN과 ViT의 특징 맵을 융합하여 디코더로 전달하며, 이 과정에서 Sobel 연산자로 생성된 경계 마스크가 디코더의 각 층에 주입되어 경계 학습을 가이드한다.

### Dual-Stream Encoder Network
인코더는 두 개의 스트림으로 구성된다.
1. **CNNs Stream**: Res2Net을 백본으로 사용하여 짧은 범위의 특징 의존성(Local context)을 캡처한다. Res2Net은 split attention 메커니즘을 통해 다중 스케일 특징 표현을 생성한다.
2. **Auxiliary StitchViT Stream**: 전역적 의존성을 캡처하기 위해 StitchViT를 사용한다. 입력 특징 맵 $F^c$를 서로 다른 stride $s \in \{2, 4, 8, 16\}$로 샘플링하여 패치 $P_i$를 생성하는 'Stitch' 연산을 수행한다.
   - 샘플링 예시: $P_1 = F^c[1::2, 1::2, :]$
   - 이후 Multi-Head Self-Attention(MHSA)을 적용하여 $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$를 계산하고, Feed-Forward Network(FFN)를 거쳐 최종 특징을 출력한다.

### Boundary-Guided Decoder Network
경계 학습 능력을 강화하기 위해 두 가지 모듈을 도입하였다.
1. **Boundary-Extracted Module (BEM)**: 
   - Sobel 연산자 $K_x, K_y$를 사용하여 수평 및 수직 그래디언트 맵 $M_x, M_y$를 생성한다.
   - 경계 강화 특징 맵 $F_e$는 다음과 같이 계산된다:
     $$F_e = F_c \odot \sigma(M_{xy})$$
     여기서 $\odot$은 요소별 곱셈이며, $M_{xy}$는 $M_x$와 $M_y$의 결합이다. 이를 통해 노이즈가 제거된 고품질의 이진 경계 마스크 $F_b$를 생성한다.
2. **Boundary-Injected Module (BIM)**: 
   - Boundary Injection Operation(BIO)을 통해 전경(Foreground)과 배경(Background)의 특징 표현을 각각 강화한다.
   - 배경 경로(Background path)에서는 다음과 같은 배경 주의 집중 맵(Background attention map)을 사용한다:
     $$F_{bg} = \text{Convs}((1 - \sigma(F_{d}^{j-1})) \odot F_c)$$
     여기서 $(1 - \sigma(F_{d}^{j-1}))$는 이전 디코더 층의 출력을 이용해 계산된 배경 영역에 대한 가중치이다.

### 손실 함수 (Loss Function)
CTO는 시맨틱 분할과 경계 검출을 동시에 수행하는 멀티태스크 학습 모델이다.
- **분할 손실 ($L_{seg}$)**: Cross-Entropy Loss($L_{CE}$)와 mean IoU Loss($L_{mIoU}$)의 가중 합으로 정의된다.
- **경계 손실 ($L_{bnd}$)**: 클래스 불균형 문제를 해결하기 위해 Dice Loss($L_{Dice}$)를 사용한다.
- **전체 손실 ($L$)**:
  $$L = \sum_{i=1}^{L} (L_{CE} + L_{mIoU}) + \alpha L_{Dice}$$
  여기서 $L=3$ (BIM의 수), $\alpha=3$ (가중치 계수)이다.

## 📊 Results

### 실험 설정
- **데이터셋**: ISIC 2016, ISIC 2018, PH2 (피부 병변), CoNIC (핵 분할), LiTS17 (간 종양), MSD BraTS (뇌 종양), BTCV (복부 장기) 등 총 7개의 챌린징한 데이터셋에서 검증하였다.
- **측정 지표**: Dice Coefficient, IoU, average Hausdorff Distance (HD), Panoptic Quality (PQ) 및 모델 효율성(FLOPs, Params)을 측정하였다.

### 주요 결과
1. **정량적 성능**: 
   - ISIC 2018에서 Dice 90.6%, IoU 84.0%를 달성하여 TransUNet 등 기존 SOTA 모델을 능가하였다.
   - 2D 데이터셋(ISIC, PH2, CoNIC)뿐만 아니라 3D 데이터셋(LiTS17, BraTS, BTCV)에서도 경쟁력 있는 성능을 보였다. 특히 BTCV 데이터셋에서 경계가 모호한 비장(Spleen)과 위(Stomach)의 분할 성능이 크게 향상되었다.
2. **효율성**: 모델 파라미터 62.22M, 22.70G FLOPs를 기록하며, 높은 정확도를 유지하면서도 계산 복잡도 면에서 경쟁력을 확보하였다.
3. **대형 모델과의 비교**: 최근 주목받는 MedSAM과 비교했을 때, 특정 데이터셋에서 CTO가 더 우수한 성능을 보였으며, 특히 훨씬 더 경량화된 구조임을 확인하였다.

## 🧠 Insights & Discussion

### 이론적 배경: MRMR 원칙
본 논문은 **Minimum Redundancy Maximum Relevance (MRMR)** 원칙을 통해 CTO의 정당성을 설명한다. 
- **CNN 스트림**: 장기의 표면과 같은 국소적 문맥 정보 제공.
- **Transformer 스트림**: 장기의 위치와 같은 전역적 시맨틱 정보 제공.
- **Edge Operator**: 장기와 배경 사이의 명확한 대비(Contrast) 정보 제공.
이 세 가지 구성 요소는 서로 매우 다르면서(Low redundancy) 목표 작업에는 매우 관련성이 높은(High relevance) 특징들을 추출하므로, 이를 결합했을 때 모델의 일반화 성능과 표현 능력이 극대화된다는 해석이다.

### 한계점 및 비판적 해석
논문에서 명시한 한계점은 다음과 같다.
- **3D 컨볼루션 미적용**: 현재 2D 기반 구조를 3D 데이터에 적용하고 있으므로, 실제 3D 컨볼루션을 도입한다면 공간적 관계를 더 잘 포착하여 성능을 높일 수 있을 것이다.
- **도메인 시프트 문제**: 서로 다른 의료 센터 간의 데이터 차이에 따른 일반화 성능 연구가 부족하다.
- **실시간성 부족**: 임상 현장에서 즉각적으로 사용하기 위한 실시간 최적화가 아직 이루어지지 않았다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 고질적인 문제인 **'경계 영역의 부정확함'**을 해결하기 위해 **CNN(국소), ViT(전역), Sobel Operator(경계)**를 결합한 **CTO** 아키텍처를 제안한다. 특히 추가 레이블 없이 스스로 경계 마스크를 생성하여 디코더를 가이드하는 방식과 효율적인 StitchViT 구조를 통해, 7개의 의료 영상 데이터셋에서 SOTA 수준의 정확도와 우수한 연산 효율성을 입증하였다. 이 연구는 명시적인 경계 정보 주입이 의료 영상 분할의 정밀도를 높이는 데 결정적인 역할을 함을 시사하며, 향후 3D 프레임워크로의 확장 가능성이 높다.