# SAM-Mamba: Mamba Guided SAM Architecture for Generalized Zero-Shot Polyp Segmentation

Tapas Kumar Dutta, Snehashis Majhi, Deepak Ranjan Nayak, and Debesh Jha (2024)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 대장내시경 영상에서 폴립(Polyp)을 정확하게 분할(Segmentation)하는 것이다. 폴립 분할은 대장암의 조기 발견과 치료에 있어 필수적이지만, 폴립의 구조, 색상, 크기가 매우 다양하고 주변 조직과의 경계가 불분명하여 자동화된 분할이 매우 어렵다.

기존의 Convolutional Neural Networks (CNN) 기반 모델들은 세부적인 패턴과 전역적인 문맥(Global context)을 동시에 포착하는 데 한계가 있으며, Vision Transformer (ViT) 기반 모델들은 지역적 문맥(Local context) 포착 능력이 부족하고 새로운 데이터셋에 대한 제로샷 일반화(Zero-shot generalization) 성능이 떨어진다는 문제점이 있다. 최근 등장한 Segment Anything Model (SAM)은 뛰어난 제로샷 성능을 보이지만, 의료 영상과 같은 특정 도메인에 대한 지식이 부족하여 폴립 분할에 직접 적용했을 때 성능이 낮게 나타난다. 또한, SAM을 전체 미세 조정(Full fine-tuning)하는 것은 막대한 계산 비용과 메모리를 요구하며, 과적합(Overfitting)의 위험이 있다.

따라서 본 논문의 목표는 SAM의 강력한 범용 표현 능력과 Mamba의 효율적인 전역 문맥 모델링 능력을 결합하여, 적은 계산 비용으로도 높은 일반화 성능을 갖는 폴립 분할 모델인 SAM-Mamba를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 SAM의 이미지 인코더에 폴립 도메인의 특수한 정보(Domain prior)를 주입하는 **Mamba-Prior** 모듈을 도입하는 것이다. 이는 SAM이 가진 일반적인 특징 표현과 폴립 영상 특유의 단서들 사이의 간극을 메우기 위한 설계이다.

핵심 기여는 다음과 같다.
첫째, SAM의 인코더에 Mamba 기반의 Prior 모듈을 결합하여, 멀티스케일 및 전역 문맥 단서를 효과적으로 포착함으로써 일반화된 제로샷 폴립 분할 성능을 향상시켰다.
둘째, **Multi-scale Spatial Decomposition (MSD)**와 **Mamba block**으로 구성된 Mamba-Prior 모듈을 제안하였다. MSD를 통해 다양한 크기의 공간적 특징을 학습하고, Mamba 블록을 통해 특징 맵 내의 광범위한 문맥을 포착하여 복잡한 폴립과 그 경계를 효과적으로 분할하도록 설계하였다.
셋째, 5개의 벤치마크 데이터셋에 대한 광범위한 실험을 통해 기존의 CNN, ViT 및 Adapter 기반 모델보다 우수한 정량적/정성적 성능과 제로샷 일반화 능력을 입증하였다.

## 📎 Related Works

폴립 분할 분야의 기존 연구는 크게 세 가지 방향으로 전개되었다.
첫째, **CNN 기반 접근 방식**이다. U-Net 및 이를 개선한 UNet++, ResUNet++ 등이 대표적이며, 이후 PraNet, CFA-Net, MEGANet 등 경계 정보나 스케일 다양성을 해결하려는 모델들이 제안되었다. 그러나 이들은 전역적인 특징 관계를 학습하는 능력이 부족하여 복잡하거나 매우 작은 폴립을 감지하는 데 한계가 있다.
둘째, **Transformer 기반 접근 방식**이다. TransUNet, UNETR, PVT-Cascade, CTNet 등이 self-attention 메커니즘을 통해 전역 문맥을 모델링함으로써 성능을 높였으나, 지역적 문맥 포착 능력이 부족하고 보지 못한 데이터셋에 대한 일반화 성능이 제한적이다.
셋째, **Foundation Model 기반 접근 방식**이다. SAM은 뛰어난 범용성을 보이지만, 의료 영상 도메인의 특수성이 결여되어 성능이 낮다. 이를 해결하기 위해 Adapter를 사용하는 연구가 진행되었으나, 단순한 Adapter만으로는 폴립의 색상, 모양, 모호한 경계와 같은 결정적인 특징을 학습하기에 부족함이 있다.

본 연구는 Mamba의 State Space Models (SSM)가 가진 선형 계산 복잡도와 장거리 의존성 모델링 능력을 SAM에 결합함으로써, 기존 모델들이 해결하지 못한 효율성과 일반화 성능의 트레이드-오프 문제를 해결하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인 구조
SAM-Mamba는 기본적으로 SAM의 백본 구조를 유지하되, 이미지 인코더(Image Encoder) 내에 **Mamba-Prior 모듈**과 **Adapter**를 추가하여 효율적인 미세 조정을 수행한다. SAM의 기존 파라미터는 동결(Frozen) 상태로 유지하며, 추가된 모듈들만 학습시키는 방식을 취한다.

### Mamba-Prior 모듈
Mamba-Prior 모듈은 폴립 영상의 핵심 단서를 추출하여 SAM 인코더에 주입하며, 세 가지 전략으로 구성된다.

1. **Multi-scale Spatial Decomposition (MSD)**: 입력 영상 $I \in \mathbb{R}^{H \times W \times C}$를 서로 다른 수용 영역(Receptive field)을 가진 병렬 컨볼루션 레이어($k \in \{3, 5, 7\}$)로 처리한다. 생성된 맵들을 채널 방향으로 결합하여 스케일 피라미드 $M^* \in \mathbb{R}^{H \times W \times 3\dot{C}_0}$를 구축함으로써, 미세한 디테일부터 거친 세맨틱까지 계층적으로 분석한다.
2. **Channel Saliency and Context Accumulation**: $M^*$에 대해 Global Max Pooling과 Global Average Pooling을 적용하여, 각각 돌출된 특징(Saliency)과 광범위한 문맥(Context) 정보를 담은 $M^S, M^C \in \mathbb{R}^{1 \times 3\dot{C}_0}$를 추출한다.
3. **Mamba Channel Interaction**: 추출된 $M^S$와 $M^C$를 두 개의 병렬 Mamba 레이어에 입력하여 채널 간의 장거리 의존성을 학습한다. 각 Mamba 레이어의 연산은 다음과 같이 정의된다.
   $$M^S_o = \phi(\text{SSM}(\sigma(\text{Conv}(\phi(M^S)))) \otimes \sigma(\phi(M^S)))$$
   $$M^C_o = \phi(\text{SSM}(\sigma(\text{Conv}(\phi(M^C)))) \otimes \sigma(\phi(M^C)))$$
   여기서 $\phi(\cdot)$는 선형 레이어, $\sigma$는 SiLU 활성화 함수, $\otimes$는 행렬 곱셈을 의미한다.

최종적으로 Mamba의 게이팅 메커니즘으로 인해 소실될 수 있는 세부 정보를 보존하기 위해 skip-connection을 적용하며, 다음과 같이 도메인 Prior 특징 맵 $M^D$를 생성한다.
$$M^D = \text{Concat}(M^S_o \odot M^*, M^C_o \odot M^*)$$

### Adapter 및 SAM Decoder
- **Adapter**: 두 개의 연속적인 Cross-Attention 모듈을 사용하여 $M^D$를 ViT 블록에 주입한다. 이를 통해 사전 학습된 ViT의 특징 분포를 크게 훼손하지 않으면서 폴립 도메인 정보를 융합한다.
- **SAM Decoder 및 Pseudo-mask**: SAM은 원래 포인트나 박스 같은 프롬프트가 필요하지만, 본 모델은 이를 자동화하기 위해 인코더의 출력을 통해 **Pseudo-mask**를 먼저 생성하고, 이를 디코더의 프롬프트로 입력하여 최종 마스크를 정교화한다.

### 학습 절차 및 손실 함수
학습은 Dice loss와 가중치 적용 Binary Cross Entropy (BCE) loss의 조합인 $L_D = L^w_{\text{Dice}} + L^w_{\text{BCE}}$를 사용하며, 두 단계로 진행된다.
- **Stage 1**: 이미지 인코더 내의 Adapter들을 먼저 학습시킨다. 이때 인코더의 보조 출력($S^{\text{up}}_{\text{Encoder}}$)을 이용한 Deep Supervision을 적용하여 $L_{\text{stage1}} = L_D(G, S^{\text{up}}_{\text{Encoder}})$로 최적화한다.
- **Stage 2**: 마스크 디코더를 포함한 전체 모델(Mamba-Prior, Adapter, Decoder)을 학습시킨다. 최종 출력 $S_{\text{Decoder}}$를 이용해 $L_{\text{stage2}} = L_D(G, S_{\text{Decoder}})$로 최적화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Kvasir-SEG, CVC-ClinicDB(학습 및 테스트), CVC-300, CVC-ColonDB, ETIS(제로샷 테스트 전용) 등 총 5개 데이터셋을 사용하였다.
- **평가 지표**: mDice, mIoU, $S^\alpha$, $F^w_\beta$, $E^{\max}_\phi$, MAE의 6가지 지표를 사용하여 정량 평가를 수행하였다.

### 주요 결과
1. **학습 데이터셋 성능**: Kvasir-SEG와 CVC-ClinicDB에서 SAM-Mamba는 기존 SOTA 모델들을 능가하였다. 특히 Kvasir-SEG에서 mDice 92.4%를 기록하며 CTNet 등 경쟁 모델보다 우수한 성능을 보였다.
2. **제로샷 일반화 능력**: 보지 못한 데이터셋인 CVC-300, CVC-ColonDB, ETIS에서 매우 강력한 성능을 입증하였다. 특히 CVC-ColonDB와 ETIS의 mIoU 지표에서 기존 SOTA 모델 대비 각각 +4%, +3.8%의 상당한 성능 향상을 달성하였다.
3. **정성적 분석**: MSD 모듈을 통해 다양한 크기의 폴립을 정확히 감지하였으며, Mamba 레이어의 전역 문맥 포착 능력 덕분에 오탐(False Positive) 비율이 현저히 낮아진 것을 확인하였다.

### 절제 연구 (Ablation Study)
- **구성 요소의 효과**: Adapter만 사용했을 때보다 MSD와 Mamba를 모두 포함했을 때 모든 데이터셋에서 성능이 가장 높았다. 이는 Mamba가 세부적이고 돌출된 특징을 포착하는 데 결정적인 역할을 함을 시사한다. (파라미터는 약 9.5% 증가)
- **커널 크기의 영향**: 단일 스케일(Uni-scale, $3\times3, 5\times5, 7\times7$) 설정보다 멀티스케일(Multi-scale) 설정이 특히 보지 못한 데이터셋(Unseen datasets)에서 훨씬 더 높은 강건함과 일반화 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 SAM이라는 거대 모델을 의료 도메인에 효율적으로 적응시키기 위해, 단순히 파라미터를 추가하는 Adapter 방식에서 나아가 **Mamba-Prior**라는 도메인 특화 지식 주입 구조를 설계했다는 점에서 강점이 있다. 특히 Mamba의 선형 복잡도를 활용하여 전역 문맥을 포착함으로써, Transformer의 무거운 연산량 문제를 피하면서도 ViT의 한계를 극복하였다.

또한, Pseudo-mask를 이용한 프롬프트 생성 방식은 SAM의 최대 단점인 '사용자 입력 의존성'을 제거하여 실제 임상 환경에서 자동화된 도구로 활용될 가능성을 높였다.

다만, 정량적 결과에서 $E^{\max}_\phi$ 지표가 다른 지표에 비해 상대적으로 향상 폭이 적은데, 이는 모델이 전역적인 위치와 형태는 잘 잡지만 아주 미세한 경계(Edge)를 정밀하게 묘사하는 능력에는 아직 개선의 여지가 있음을 의미한다. 또한, MAE 값의 최적화 가능성 역시 향후 과제로 남아 있다.

## 📌 TL;DR

본 연구는 SAM의 일반화 능력과 Mamba의 효율적인 전역 문맥 모델링을 결합한 **SAM-Mamba**를 제안하여 폴립 분할 성능을 극대화하였다. **Mamba-Prior 모듈(MSD + Mamba)**을 통해 폴립 특유의 멀티스케일 특징을 학습하고 이를 SAM 인코더에 주입함으로써, 학습하지 않은 데이터셋에서도 매우 뛰어난 **제로샷 일반화 성능**을 달성하였다. 이 연구는 거대 파운데이션 모델을 특정 의료 도메인에 효율적으로 전이 학습시키는 효과적인 방법론을 제시하였으며, 향후 실시간 임상 진단 보조 시스템 구축에 중요한 기여를 할 것으로 기대된다.