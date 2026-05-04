# LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation

Weibin Liao, Yinghao Zhu, Xinyuan Wang, Chengwei Pan, Yasha Wang, and Liantao Ma (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 UNet과 그 변형 모델들은 널리 사용되어 왔다. 하지만 기존의 CNN 기반 모델들은 합성곱 연산의 고유한 지역성(locality)으로 인해 전역적인 문맥 정보와 장거리 공간 의존성(long-range spatial dependencies)을 포착하는 데 한계가 있다. 이를 해결하기 위해 Transformer 아키텍처를 통합한 모델들이 제안되었으나, self-attention 메커니즘의 특성상 이미지 크기에 따라 연산 복잡도가 이차적으로 증가($quadratic\ complexity$)하는 문제가 발생한다.

이러한 높은 파라미터 수와 계산 비용은 실제 의료 현장, 특히 모바일 헬스케어 애플리케이션과 같이 컴퓨팅 자원이 제한된 환경에서 모델을 배포하는 데 큰 걸림돌이 된다. 따라서 본 논문은 추가적인 파라미터 증가나 계산 부담 없이 UNet에 장거리 의존성 모델링 능력을 부여하여, 경량화된 의료 영상 분할 모델을 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최근 등장한 State Space Models (SSMs)의 일종인 Mamba를 UNet의 CNN 및 Transformer 구조를 대체하는 경량 대체제로 사용하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **LightM-UNet 제안**: Mamba를 UNet 아키텍처에 통합하여 파라미터 수를 약 1M 수준으로 획기적으로 줄인 경량 분할 모델을 제안한다. 이는 기존의 nnU-Net이나 U-Mamba 대비 파라미터 수를 수십 배에서 수백 배까지 절감하면서도 동등하거나 더 우수한 성능을 보인다.
2.  **Residual Vision Mamba Layer (RVM Layer) 설계**: 순수 Mamba 방식으로 이미지의 깊은 시맨틱 특징을 추출하기 위해 RVM Layer를 제안한다. 특히 residual connection과 adjustment factor를 도입하여 추가적인 연산량 증가 없이 SSM의 장거리 공간 모델링 능력을 강화하였다.
3.  **경량 최적화 전략의 제시**: 기존 연구들이 CNN-SSM 하이브리드 구조를 채택하여 여전히 높은 계산 비용을 초래한 것과 달리, 본 연구는 Mamba를 CNN과 Transformer의 완전한 경량 대체제로 사용하여 실제 의료 환경의 자원 제한 문제를 해결하고자 하였다.

## 📎 Related Works

의료 영상 분할의 표준인 UNet은 대칭적인 U자형 구조와 skip connection을 통해 우수한 성능을 보이지만, CNN 기반의 한계로 인해 전역 정보 포착이 어렵다. 이를 극복하기 위해 atrous convolution, self-attention, image pyramids 등이 시도되었으나 여전히 장거리 의존성 모델링에는 제약이 있었다.

이후 등장한 Transformer 기반 모델(UNETR, SwinUNETR 등)은 이미지를 패치 시퀀스로 처리하여 전역 정보를 효과적으로 캡처하지만, 연산 복잡도가 매우 높아 모바일 환경 적용이 어렵다. 최근 제안된 U-Mamba와 같은 SSM 기반 모델은 선형 복잡도로 장거리 의존성을 모델링할 수 있다는 장점이 있으나, 여전히 많은 수의 파라미터(약 173M)와 높은 GFLOPs를 사용하여 경량 모델이라고 보기 어렵다. LightM-UNet은 이러한 하이브리드 모델들의 무거운 구조를 지양하고, Mamba를 통한 극단적인 경량화를 추구함으로써 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
LightM-UNet은 입력 이미지 $I \in \mathbb{R}^{C \times H \times W \times D}$를 받아 다음과 같은 파이프라인으로 처리한다.
1.  **초기 특징 추출**: Depthwise Convolution (DWConv) 레이어를 통해 초기 얕은 특징 맵 $F^S$를 생성한다.
2.  **Encoder**: 3개의 Encoder Block이 순차적으로 배치되어 특징을 추출한다. 각 블록을 거칠 때마다 채널 수는 2배로 증가하고 해상도는 절반으로 감소한다.
3.  **Bottleneck**: 4개의 RVM Layer로 구성된 Bottleneck Block이 해상도를 유지하며 장거리 공간 의존성을 모델링한다.
4.  **Decoder**: 3개의 Decoder Block이 특징을 디코딩하고 해상도를 복원한다. 각 블록 후 채널 수는 절반으로 줄고 해상도는 2배로 증가한다.
5.  **최종 출력**: 마지막 Decoder Block의 출력을 DWConv와 SoftMax 함수에 통과시켜 최종 세그멘테이션 마스크를 생성한다. skip connection은 인코더의 특징 맵을 디코더에 직접 전달하여 다중 레벨 특징을 제공한다.

### 주요 구성 요소 및 방정식

#### 1. Residual Vision Mamba Layer (RVM Layer)
RVM Layer는 SSM 블록을 강화하여 깊은 시맨틱 특징을 추출한다. LayerNorm과 VSSM을 거친 후, adjustment factor $s$가 적용된 residual connection을 통해 성능을 높인다.

$$f^{M_l} = \text{VSSM}(\text{LayerNorm}(M_{in}^l)) + s \cdot M_{in}^l$$

이후 다시 LayerNorm과 projection layer를 거쳐 최종 출력을 생성한다.

$$M_{out}^l = \text{Projection}(\text{LayerNorm}(f^{M_l}))$$

#### 2. Vision State-Space Module (VSS Module)
VSS Module은 입력 $W_{in}$을 두 개의 병렬 브랜치로 나누어 처리한 뒤 결합한다.
- **브랜치 1**: $\text{Linear} \rightarrow \text{DWConv} \rightarrow \text{SiLU} \rightarrow \text{SSM} \rightarrow \text{LayerNorm}$
- **브랜치 2**: $\text{Linear} \rightarrow \text{SiLU}$

두 브랜치의 결과물 $W_1, W_2$는 Hadamard product ($\odot$)로 결합되며, 최종적으로 Linear 레이어를 통해 원래 채널 수로 투영된다.

$$W_1 = \text{LayerNorm}(\text{SSM}(\text{SiLU}(\text{DWConv}(\text{Linear}(W_{in})))))$$
$$W_2 = \text{SiLU}(\text{Linear}(W_{in}))$$
$$W_{out} = \text{Linear}(W_1 \odot W_2)$$

#### 3. Decoder Block
디코더 블록은 skip connection에서 온 $F_D^l$와 이전 블록의 출력 $P_{in}$을 더한 후, DWConv와 adjustment factor $s'$가 포함된 residual connection, ReLU 활성화 함수를 순차적으로 적용한다.

$$P_{out} = \text{ReLU}(\text{DWConv}(P_{in} + F_D^l) + s' \cdot (P_{in} + F_D^l))$$

최종적으로 bilinear interpolation을 통해 원래 해상도로 복원한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 3D CT 이미지인 LiTS 데이터셋과 2D X-ray 이미지인 Montgomery \& Shenzhen 데이터셋을 사용하였다.
- **비교 모델**: CNN 기반(nnU-Net, SegResNet), Transformer 기반(UNETR, SwinUNETR), Mamba 기반(U-Mamba) 모델들과 비교하였다.
- **평가 지표**: Mean Intersection over Union (mIoU)와 Dice similarity score (DSC)를 사용하였다.
- **학습 환경**: SGD 옵티마이저(lr=1e-4), PolyLRScheduler, Cross Entropy 및 Dice loss의 조합을 사용하였으며, 100 epoch 동안 학습하였다.

### 정량적 결과
- **LiTS (3D) 데이터셋**: LightM-UNet은 평균 mIoU $77.48\%$를 달성하여 비교 모델 중 가장 우수한 성능을 보였다. 특히 nnU-Net 대비 파라미터 수는 $47.39\times$, 계산 비용은 $15.82\times$ 감소하였다. U-Mamba와 비교했을 때 평균 mIoU가 $2.11\%$ 향상되었으며, 특히 탐지가 어려운 작은 객체인 Tumor의 mIoU가 $3.63\%$ 향상되는 성과를 거두었다.
- **Montgomery \& Shenzhen (2D) 데이터셋**: LightM-UNet은 $\text{DSC } 96.17\%$, $\text{mIoU } 92.74\%$로 최적의 성능을 기록하였다. 파라미터 수는 $1.09\text{M}$으로 nnU-Net 대비 $99.14\%$, U-Mamba 대비 $99.55\%$ 감소하였다.

### 정성적 결과 및 절제 연구(Ablation Study)
- **시각화**: LightM-UNet은 타 모델 대비 세그멘테이션 경계가 더 매끄러우며, 작은 종양(tumor)에 대한 오인식 없이 정확하게 탐지하는 모습을 보였다.
- **모듈 검증**: VSS Module을 Convolution이나 Self-Attention으로 대체했을 때 성능 저하가 발생하였으며, 특히 연산 비용과 파라미터가 크게 증가하였다. 또한, RVM Layer에서 adjustment factor나 residual connection을 제거했을 때 mIoU가 약 $0.44\% \sim 0.69\%$ 하락함을 확인하여, 추가 파라미터 없이 성능을 높이는 제안 기법의 유효성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 Mamba라는 최신 SSM 아키텍처를 UNet에 도입함으로써 '전역 정보 포착'과 '모델 경량화'라는 상충하는 두 가지 목표를 동시에 달성하였다. 특히 기존의 U-Mamba가 CNN-SSM 하이브리드 구조를 택해 여전히 무거운 모델이었던 반면, LightM-UNet은 Mamba를 CNN/Transformer의 대체제로 사용하여 파라미터 수를 $1\text{M}$ 수준으로 극단적으로 줄인 점이 매우 고무적이다.

실험 결과에서 작은 객체(종양)에 대한 분할 성능이 향상된 것은 Mamba의 선형 복잡도 기반 장거리 의존성 모델링 능력이 의료 영상의 세밀한 구조 파악에 효과적임을 시사한다. 다만, 본 논문은 두 가지 데이터셋에 대해서만 검증을 수행하였으므로, 더 다양한 장기(organ)와 모달리티를 가진 데이터셋에서의 범용성 검증이 추가적으로 필요할 것으로 보인다. 또한, 매우 적은 파라미터 수로 고성능을 낸 원인이 단순히 Mamba의 특성인지, 아니면 제안한 RVM Layer의 adjustment factor와 같은 세부 설계의 영향이 얼마나 지배적인지에 대한 더 심도 있는 분석이 필요하다.

## 📌 TL;DR

LightM-UNet은 UNet의 인코더와 보틀넥에 Mamba(SSM)를 통합하여, Transformer 수준의 전역 문맥 파악 능력을 유지하면서도 파라미터 수를 획기적으로 줄인(약 1M) 경량 의료 영상 분할 모델이다. 기존 SOTA 모델들보다 훨씬 가벼우면서도 더 높은 분할 정확도를 달성함으로써, 컴퓨팅 자원이 제한된 모바일 헬스케어 환경으로의 적용 가능성을 크게 높였다.