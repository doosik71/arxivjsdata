# SAM-Mamba: Mamba Guided SAM Architecture for Generalized Zero-Shot Polyp Segmentation

Tapas Kumar Dutta, Snehashis Majhi, Deepak Ranjan Nayak, and Debesh Jha (2024)

## 🧩 Problem to Solve

본 논문은 대장내시경 영상에서 용종(Polyp)을 정확하게 분할(Segmentation)하는 문제를 다룬다. 대장암(CRC)의 조기 발견을 위해서는 용종의 정확한 검출이 필수적이지만, 실제 임상 환경에서는 다음과 같은 어려움이 존재한다.

- **용종의 특성**: 용종은 크기, 색상, 형태가 매우 다양하며, 주변 조직과 색상 및 질감이 유사하여 경계가 불분명한 '위장(Camouflage)' 특성을 가진다.
- **기존 모델의 한계**: CNN 기반 모델은 전역적 문맥(Global Context) 파악 능력이 부족하며, Vision Transformer(ViT) 기반 모델은 지역적 문맥(Local Context) 캡처에 어려움이 있고 제로샷(Zero-shot) 일반화 성능이 낮다.
- **SAM(Segment Anything Model)의 한계**: 강력한 제로샷 능력을 갖춘 SAM을 의료 영상에 직접 적용할 경우, 도메인 특화 지식의 부족으로 성능이 저하된다. 또한, SAM의 모든 파라미터를 미세 조정(Full fine-tuning)하는 것은 계산 비용과 메모리 요구량이 지나치게 높으며, 포인트나 박스 같은 프롬프트(Prompt)가 반드시 필요하다는 제약이 있다.

따라서 본 연구의 목표는 SAM의 강력한 일반화 능력을 유지하면서, 적은 비용으로 용종 도메인에 특화된 지식을 주입하여 프롬프트 없이도 작동하는 효율적인 용종 분할 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM의 이미지 인코더에 **Mamba-Prior 모듈**을 도입하고 **Adapter 기반의 미세 조정**을 수행하는 것이다.

1. **Mamba-Prior 모듈 제안**: 다양한 스케일의 공간적 특징을 추출하는 MSD(Multi-scale Spatial Decomposition)와 상태 공간 모델(State Space Model, SSM) 기반의 Mamba 블록을 결합하여, 용종의 전역적 문맥과 다중 스케일 특징을 효율적으로 캡처한다.
2. **효율적인 적응(Adaptation)**: SAM의 거대한 파라미터를 동결(Frozen)시킨 채, 경량화된 Adapter를 통해 용종 도메인에 맞게 최적화함으로써 계산 비용을 줄이고 과적합을 방지한다.
3. **프롬프트 의존성 해결**: 별도의 사용자 입력(포인트, 박스 등) 없이도 작동할 수 있도록, 학습 가능한 네트워크 $f(\theta)$를 통해 생성된 '의사 마스크(Pseudo-mask)'를 SAM 디코더의 프롬프트로 사용하는 파이프라인을 구축하였다.

## 📎 Related Works

- **CNN 기반 접근법**: U-Net, UNet++, ResUNet++ 등이 사용되었으나, 전역적 특징 관계를 파악하는 능력이 부족하여 복잡하거나 아주 작은 용종을 검출하는 데 한계가 있었다.
- **Transformer 기반 접근법**: TransUNet, UNETR 등이 전역 의존성을 모델링하여 성능을 높였으나, 지역적 문맥 캡처 능력이 부족하며 학습 데이터 외의 새로운 데이터셋에 대한 일반화 성능이 낮았다.
- **SAM 및 Adapter**: SAM은 범용 분할 모델로서 뛰어나지만 의료 도메인 지식이 부족하다. 이를 해결하기 위해 일부 연구에서 Adapter를 사용했으나, 여전히 용종의 미세한 경계와 다양한 크기를 모두 잡아내는 데는 한계가 있었다.
- **Mamba (SSM)**: 최근 등장한 Mamba는 선형 계산 복잡도로 긴 시퀀스 데이터를 처리할 수 있어, Transformer의 연산 비용 문제를 해결하면서도 전역적 의존성을 잘 포착하는 대안으로 주목받고 있다.

## 🛠️ Methodology

### 전체 시스템 구조

SAM-Mamba는 크게 **Mamba-Prior 모듈 $\rightarrow$ SAM 이미지 인코더(with Adapters) $\rightarrow$ SAM 마스크 디코더** 순으로 구성된다. SAM의 기본 가중치는 동결된 상태이며, Mamba-Prior와 Adapter만이 학습 가능하다.

### Mamba-Prior 모듈

이 모듈은 SAM 인코더에 도메인 지식을 주입하기 위해 다음 세 단계로 작동한다.

1. **Multi-scale Spatial Decomposition (MSD)**:
   입력 영상 $I \in \mathbb{R}^{H \times W \times C}$를 서로 다른 수용역(Receptive field)을 가진 병렬 컨볼루션 레이어($k \in \{3, 5, 7\}$)에 통과시킨다. 결과물들을 채널 방향으로 결합하여 다중 스케일 특징 피라미드 $M^* \in \mathbb{R}^{H \times W \times 3C_0}$를 생성한다.

2. **Channel Saliency and Context Accumulation**:
   $M^*$에서 Global Max Pooling을 통해 돌출된 특징(Saliency) $M_S$를, Global Average Pooling을 통해 배경 문맥(Context) $M_C$를 추출한다. 두 결과물은 $\mathbb{R}^{1 \times 3C_0}$ 크기의 벡터가 된다.

3. **Mamba Channel Interaction**:
   추출된 $M_S$와 $M_C$를 각각 독립적인 Mamba 레이어에 통과시켜 채널 간의 장거리 의존성을 모델링한다. 이때의 연산 과정은 다음과 같다.
   $$M_{S/C}^o = \phi(\text{SSM}(\sigma(\text{Conv}(\phi(M_{S/C})))) \otimes \sigma(\phi(M_{S/C})))$$
   여기서 $\phi$는 선형 레이어, $\sigma$는 SiLU 활성화 함수, $\otimes$는 행렬 곱셈을 의미한다.

마지막으로, Mamba의 결과물과 원본 다중 스케일 맵 $M^*$를 요소별 곱(Element-wise multiplication) 및 결합(Concatenation)하여 최종 도메인 사전 특징 맵 $M_D$를 생성한다.
$$M_D = \text{Concat}(M_S^o \odot M^*, M_C^o \odot M^*)$$

### Adapter 및 Decoder

- **Adapter**: 두 개의 순차적인 Cross-Attention 모듈을 사용하여 $M_D$에서 추출된 특징을 ViT 블록에 주입한다. 이를 통해 SAM의 사전 학습된 지식을 유지하면서 용종 특화 정보를 반영한다.
- **SAM Decoder**: 사용자의 프롬프트 대신, 인코더의 출력을 통해 생성된 **의사 마스크(Pseudo-mask)**를 입력으로 받아 최종 마스크를 정교화한다.

### 학습 절차 및 손실 함수

손실 함수 $L_D$는 가중치 적용 Binary Cross Entropy(BCE)와 Dice Loss의 합으로 정의된다.
$$L_D = L_{Dice}^w + L_{BCE}^w$$

학습은 2단계로 진행된다.

- **Stage 1**: 이미지 인코더 내의 Adapter들을 먼저 학습시킨다. 인코더의 출력을 업샘플링하여 정답(GT)과 비교하는 Deep Supervision 방식을 사용한다.
- **Stage 2**: 마스크 디코더와 Adapter를 함께 학습시켜 최종 분할 성능을 최적화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 학습에는 Kvasir-SEG, CVC-ClinicDB를 사용하였고, 테스트 및 제로샷 검증을 위해 CVC-300, CVC-ColonDB, ETIS 등 총 5개 데이터셋을 활용하였다.
- **지표**: mDice, mIoU, $F^w_\beta$, $S_\alpha$, $E^\phi_{max}$, MAE 등 6가지 지표를 사용하였다.

### 정량적 결과

- **학습 데이터(Seen)**: Kvasir-SEG에서 mDice 92.4%를 기록하며 CTNet 등 기존 SOTA 모델들을 능가하였다.
- **미학습 데이터(Unseen/Zero-shot)**: 제로샷 일반화 성능이 매우 뛰어나게 나타났다. 특히 CVC-ColonDB와 ETIS 데이터셋에서 mIoU 기준 SOTA 모델 대비 각각 **+4%, +3.8%**의 성능 향상을 보였다.

### 정성적 결과 및 분석

- **MSD 모듈의 효과**: 다양한 크기의 용종을 정확하게 포착하며, 특히 두 개 이상의 용종이 존재하는 경우에도 이를 잘 식별하였다.
- **Mamba의 효과**: 전역적 문맥 파악 능력이 뛰어나 오탐(False Positive) 발생률을 현저히 낮추었다.
- **Ablation Study**: Mamba 구성 요소가 없을 때보다 있을 때 모든 데이터셋에서 성능이 상승하였으며, 단일 스케일(Uni-scale)보다 다중 스케일(Multi-scale) 방식이 제로샷 일반화 성능을 크게 향상시킴을 확인하였다.

## 🧠 Insights & Discussion

### 강점

본 모델은 SAM의 강력한 기초 표현력(Foundation representation)을 활용하면서도, Mamba를 통해 의료 영상 특유의 전역적/지역적 특징을 효율적으로 결합하였다. 특히, 별도의 프롬프트 입력 없이도 높은 성능을 내는 구조를 설계하여 실제 임상 적용 가능성을 높였다.

### 한계 및 비판적 해석

실험 결과에서 $E^\phi_{max}$(경계선 정확도) 지표가 다른 지표에 비해 상대적으로 낮게 나타난 점은, 여전히 용종의 정밀한 경계 추출에 개선의 여지가 있음을 시사한다. 또한, Mamba-Prior 모듈이 추가됨에 따라 파라미터 수가 약 9.5% 증가하였는데, 이는 매우 적은 양이지만 실시간 추론 속도에 미치는 영향에 대한 구체적인 분석은 부족하다.

## 📌 TL;DR

**SAM-Mamba**는 SAM의 제로샷 일반화 능력을 용종 분할에 최적화하기 위해 **Mamba 기반의 도메인 사전 지식(Prior) 모듈**과 **경량 Adapter**를 결합한 아키텍처이다. 다중 스케일 특징 추출과 Mamba의 전역 의존성 모델링을 통해, 학습하지 않은 데이터셋에서도 기존 SOTA 모델들을 압도하는 일반화 성능을 보였으며, 이는 의료 영상 분석에서 기초 모델을 효율적으로 적응시키는 효과적인 방법론을 제시한 것으로 평가된다.
